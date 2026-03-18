"""
shadow_dna.py
-------------
Extracts the "Shadow DNA" of a clinic's communication style:
the behavioral fingerprint used to make the AI agent indistinguishable
from the human team.

Two layers:
  1. Pure-Python extraction (fast, deterministic):
     - emoji_frequency, average_response_length_tokens,
       response_time_metrics, tone_classification (heuristic)

  2. DSPy extraction (LLM, semantic):
     - greeting_example, closing_example, local_entities,
       personality_traits, forbidden_terms, handoff_triggers,
       knowledge_gaps (unresolved queries)

Both are aggregated across all conversations and merged into
a single ShadowDNA dataclass.
"""

import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import dspy

from analyzer.parser import Conversation, Message
from analyzer.dspy_pipeline import get_aggregate_lm

logger = logging.getLogger(__name__)

# Rough token estimate: 1 token ≈ 4 chars (pt-BR average)
CHARS_PER_TOKEN = 4

# Common emoji regex
EMOJI_RE = re.compile(
    "[\U00002600-\U000027BF"
    "\U0001F300-\U0001F9FF"
    "\U00002702-\U000027B0"
    "\U0000FE00-\U0000FE0F"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF]+",
    flags=re.UNICODE,
)


# ------------------------------------------------------------------
# DSPy Signatures
# ------------------------------------------------------------------

class ShadowDNASignature(dspy.Signature):
    """
    Analise o conjunto de conversas de WhatsApp de uma clínica e extraia
    o perfil comportamental ("Shadow DNA") do estilo de comunicação da equipe.
    Responda em português. Seja específico e baseado nos textos reais.
    """
    conversations_sample: str = dspy.InputField(
        desc="Amostra de até 10 conversas concatenadas (início e fim de cada uma)"
    )
    clinic_name: str = dspy.InputField(desc="Nome da clínica")
    payment_mentions_raw: str = dspy.InputField(
        desc=(
            "Mensagens reais que mencionam formas de pagamento, parcelamento, PIX ou descontos. "
            "Extraídas de TODAS as conversas (não apenas da amostra). "
            "Use para preencher local_payment_conditions com precisão."
        )
    )

    tone_classification: str = dspy.OutputField(
        desc="Tom dominante: Formal, Informal, Empático, Direto ou Misto"
    )
    greeting_example: str = dspy.OutputField(
        desc="Exemplo real de saudação usado pela clínica (copie do texto)"
    )
    closing_example: str = dspy.OutputField(
        desc="Exemplo real de encerramento usado pela clínica (copie do texto)"
    )
    personality_traits: list = dspy.OutputField(
        desc="Lista de 3-5 traços de personalidade da comunicação da clínica"
    )
    forbidden_terms: list = dspy.OutputField(
        desc="Lista de termos/frases problemáticos detectados (respostas ruins, inadequadas)"
    )
    handoff_keywords: list = dspy.OutputField(
        desc="Palavras ou frases que indicam necessidade de atendimento humano"
    )
    handoff_situations: list = dspy.OutputField(
        desc="Situações (não palavras-chave) que devem escalar para humano"
    )
    local_procedures: list = dspy.OutputField(
        desc="Procedimentos/serviços específicos desta clínica mencionados nas conversas"
    )
    local_insurances: list = dspy.OutputField(
        desc="Convênios/planos de saúde mencionados"
    )
    local_neighborhoods: list = dspy.OutputField(
        desc="Bairros ou regiões mencionados"
    )
    local_payment_conditions: list = dspy.OutputField(
        desc="Condições de pagamento mencionadas (ex: '12x sem juros')"
    )
    unresolved_queries: list = dspy.OutputField(
        desc="Perguntas ou temas recorrentes que ficaram sem resposta satisfatória"
    )
    agent_suggested_name: str = dspy.OutputField(
        desc="Sugestão de nome para o agente de IA desta clínica"
    )
    common_complaints: list = dspy.OutputField(
        desc=(
            "Lista de reclamações reais feitas pelos pacientes nas conversas. "
            "Foco em: demora no atendimento, informação incorreta recebida, "
            "dificuldade de reagendamento, falta de retorno da clínica, "
            "problema com procedimento ou resultado. "
            "NÃO incluir: objeções de preço (essas são objeções comerciais), "
            "queixas de saúde (são sintomas), perguntas sem resposta (são lacunas). "
            "Se não houver reclamações claras, retorne lista vazia []."
        )
    )
    attendance_flow_steps: list = dspy.OutputField(
        desc=(
            "Fluxo típico de atendimento desta clínica como lista ORDENADA de passos. "
            "Cada passo deve ser um dict com EXATAMENTE duas chaves: "
            "'step' (nome curto do passo em português, ex: 'Saudação') e "
            "'example' (mensagem REAL copiada das conversas que ilustra este passo). "
            "Extraia entre 4 e 6 passos. Exemplos de passos comuns: "
            "Saudação, Identificação do Paciente, Entendimento da Necessidade, "
            "Informações / Agendamento, Confirmação, Encerramento. "
            "Use passos específicos desta clínica, não genéricos. "
            "Formato: [{\"step\": \"Saudação\", \"example\": \"Bom dia! Aqui é...\"}, ...]"
        )
    )
    operating_hours: dict = dspy.OutputField(
        desc=(
            "Horário de funcionamento da clínica inferido das conversas. "
            "Retorne um dict com as chaves 'open' (hora de abertura, ex: '08:00'), "
            "'close' (hora de encerramento, ex: '18:00') e 'days' (lista de dias em inglês, "
            "ex: ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']). "
            "Se não for possível inferir, retorne {}."
        )
    )
    source_signals: dict = dspy.OutputField(
        desc=(
            "Sinais de origem por canal extraídos das conversas. "
            "Dict com chaves de canal ('instagram', 'google', 'referral', etc.) "
            "e valor sendo lista de frases REAIS copiadas das conversas que indicam aquele canal. "
            "Ex: {\"instagram\": [\"vi no post\", \"vi no reels\"], \"google\": [\"pesquisei no google\"]}. "
            "Use apenas canais com evidência real nas conversas. Se nenhum, retorne {}."
        )
    )


class ShadowDNAModule(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(ShadowDNASignature)

    def forward(self, conversations_sample: str, clinic_name: str, payment_mentions_raw: str = ""):
        return self.predict(
            conversations_sample=conversations_sample,
            clinic_name=clinic_name,
            payment_mentions_raw=payment_mentions_raw,
        )


class ReturningPatientPlaybookSignature(dspy.Signature):
    """
    A partir de conversas de WhatsApp de uma clínica que contêm sinais de pacientes
    recorrentes (remarcar, cancelar, retorno), extraia o playbook de atendimento para
    esses cenários. Responda em português. Use mensagens REAIS das conversas.
    Vocabulário canônico de element: greeting, identification, connection,
    active_listening, technical_details, before_after, insurances, pricing_payment,
    objections, scheduling_slots, confirmation, closing.
    """
    conversations_sample: str = dspy.InputField(
        desc="Amostra de conversas com sinais de recorrência (remarcar, cancelar, retorno, 'já fui aí')"
    )

    reschedule_elements: list = dspy.OutputField(
        desc=(
            "Lista de elementos do fluxo de remarcação. Cada item é um dict com: "
            "'element' (do vocabulário canônico), 'initiated_by' ('sofia' ou 'patient'), "
            "'trigger_signals' (lista de frases que ativam), 'blocked_by' (lista de impedimentos), "
            "'real_example' (mensagem real copiada das conversas). "
            "Formato: [{\"element\": \"greeting\", \"initiated_by\": \"sofia\", "
            "\"trigger_signals\": [\"quero remarcar\"], \"blocked_by\": [], "
            "\"real_example\": \"...\"}]"
        )
    )
    cancellation_elements: list = dspy.OutputField(
        desc=(
            "Lista de elementos do fluxo de cancelamento. Mesmo formato de reschedule_elements. "
            "Foco em: como a clínica reage ao cancelamento, tenta reverter, confirma cancelamento."
        )
    )
    followup_elements: list = dspy.OutputField(
        desc=(
            "Lista de elementos do fluxo de retorno/acompanhamento pós-consulta. "
            "Mesmo formato de reschedule_elements. "
            "Foco em: como a clínica agenda retorno, confirma próxima consulta."
        )
    )
    reschedule_example: str = dspy.OutputField(
        desc="Mensagem REAL da clínica ao tratar uma remarcação (copie do texto das conversas)"
    )
    cancellation_example: str = dspy.OutputField(
        desc="Mensagem REAL da clínica ao tratar um cancelamento (copie do texto das conversas)"
    )
    followup_example: str = dspy.OutputField(
        desc="Mensagem REAL da clínica ao tratar um retorno ou acompanhamento (copie do texto das conversas)"
    )


class ReturningPatientPlaybookModule(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(ReturningPatientPlaybookSignature)

    def forward(self, conversations_sample: str):
        return self.predict(conversations_sample=conversations_sample)


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------

@dataclass
class ShadowDNA:
    # Tone & identity
    tone_classification: str = "Misto"
    personality_traits: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)
    agent_suggested_name: str = ""

    # Greeting / closing examples (few-shot material)
    greeting_example: str = ""
    closing_example: str = ""

    # Handoff rules
    handoff_keywords: list[str] = field(default_factory=list)
    handoff_situations: list[str] = field(default_factory=list)

    # Local entities
    local_procedures: list[str] = field(default_factory=list)
    local_insurances: list[str] = field(default_factory=list)
    local_neighborhoods: list[str] = field(default_factory=list)
    local_payment_conditions: list[str] = field(default_factory=list)

    # Knowledge gaps
    unresolved_queries: list[str] = field(default_factory=list)

    # Quantitative (computed from messages)
    average_response_length_tokens: float = 0.0
    emoji_frequency: dict[str, float] = field(default_factory=dict)
    sentiment_score_distribution: dict[str, float] = field(
        default_factory=lambda: {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
    )
    response_time_metrics: dict[str, float] = field(
        default_factory=lambda: {
            "average_seconds": 0.0,
            "median_seconds": 0.0,
            "sla_adherence_percentage": 0.0,
        }
    )
    common_objections: list[str] = field(default_factory=list)

    # Patient complaints (distinct from commercial objections)
    common_complaints: list[str] = field(default_factory=list)

    # Attendance flow steps (ordered, with example messages)
    attendance_flow_steps: list[dict] = field(default_factory=list)

    # Clinic profile extras
    operating_hours: Optional[dict] = None
    source_signals: dict[str, list] = field(default_factory=dict)

    # Insurance mention counts (pure Python count, filled after LLM extraction)
    insurance_mention_counts: dict[str, int] = field(default_factory=dict)

    # RAG efficiency (filled later by blueprint builder)
    rag_efficiency_score: float = 0.0

    error: Optional[str] = None


# ------------------------------------------------------------------
# Module instance
# ------------------------------------------------------------------

_shadow_module: Optional[ShadowDNAModule] = None


def init_shadow_module():
    global _shadow_module
    _shadow_module = ShadowDNAModule()


# ------------------------------------------------------------------
# Pure-Python helpers
# ------------------------------------------------------------------

def _extract_emojis(text: str) -> list[str]:
    return EMOJI_RE.findall(text)


def _compute_quantitative(conversations: list[Conversation]) -> dict:
    """Compute purely numeric Shadow DNA fields from message data."""
    clinic_msgs: list[Message] = []
    response_times: list[float] = []

    for conv in conversations:
        clinic_msgs.extend(conv.clinic_messages)

        msgs = conv.messages
        for i, msg in enumerate(msgs):
            if msg.sender_type == "patient":
                for j in range(i + 1, len(msgs)):
                    if msgs[j].sender_type == "clinic":
                        delta = (msgs[j].sent_at - msg.sent_at).total_seconds()
                        if 0 < delta < 43200:  # 12h cap
                            response_times.append(delta)
                        break
                    if msgs[j].sender_type == "patient":
                        break

    # Average response length in tokens
    avg_len_tokens = 0.0
    if clinic_msgs:
        avg_chars = statistics.mean(len(m.content) for m in clinic_msgs)
        avg_len_tokens = round(avg_chars / CHARS_PER_TOKEN, 1)

    # Emoji frequency per emoji (ratio of clinic messages containing it)
    all_clinic_text = " ".join(m.content for m in clinic_msgs)
    total_clinic = len(clinic_msgs) or 1
    emoji_counter: Counter = Counter()
    for msg in clinic_msgs:
        for emoji in set(_extract_emojis(msg.content)):
            emoji_counter[emoji] += 1

    emoji_freq = {
        emoji: round(count / total_clinic, 3)
        for emoji, count in emoji_counter.most_common(20)
    }

    # Response time metrics
    rt_metrics = {"average_seconds": 0.0, "median_seconds": 0.0, "sla_adherence_percentage": 0.0}
    if response_times:
        rt_metrics["average_seconds"] = round(statistics.mean(response_times), 1)
        rt_metrics["median_seconds"] = round(statistics.median(response_times), 1)
        # SLA = % of responses within 30 minutes (1800s)
        sla_count = sum(1 for rt in response_times if rt <= 1800)
        rt_metrics["sla_adherence_percentage"] = round(sla_count / len(response_times) * 100, 1)

    return {
        "average_response_length_tokens": avg_len_tokens,
        "emoji_frequency": emoji_freq,
        "response_time_metrics": rt_metrics,
    }


# Keywords that indicate payment-related messages
_PAYMENT_KEYWORDS = re.compile(
    r"parcel|pix|descont|juros|\bx\s+no\b|\bx\s+sem\b|\d+x|\bà\s+vista\b|cartão|crédito|débito|pagament",
    re.IGNORECASE,
)


def _extract_payment_mentions(conversations: list[Conversation], max_msgs: int = 30) -> str:
    """
    Scan ALL conversations for messages mentioning payment methods.
    Returns a deduplicated, truncated string of real payment-related messages.
    """
    seen: set[str] = set()
    results: list[str] = []

    for conv in conversations:
        for msg in conv.clinic_messages:  # clinic messages describe payment terms
            if _PAYMENT_KEYWORDS.search(msg.content):
                normalized = msg.content.strip()
                if normalized not in seen:
                    seen.add(normalized)
                    results.append(f"- {normalized}")
                    if len(results) >= max_msgs:
                        break
        if len(results) >= max_msgs:
            break

    if not results:
        return "(nenhuma menção de pagamento encontrada nas conversas)"
    return "\n".join(results)


def _build_sample(conversations: list[Conversation], max_convs: int = 10) -> str:
    """
    Build a text sample prioritizing conversations with payment mentions.
    Takes up to max_convs conversations, first 10 + last 10 messages each.
    """
    # Sort: payment-mentioning conversations first
    def has_payment(conv: Conversation) -> bool:
        return any(
            _PAYMENT_KEYWORDS.search(m.content)
            for m in conv.clinic_messages
        )

    sorted_convs = sorted(conversations, key=lambda c: (0 if has_payment(c) else 1))
    sample = sorted_convs[:max_convs]

    sample_parts = []
    for conv in sample:
        msgs = conv.messages
        excerpt = msgs[:10] + (msgs[-10:] if len(msgs) > 20 else [])
        lines = [f"[{m.sender_type.upper()}] {m.sender}: {m.content}" for m in excerpt]
        sample_parts.append(
            f"--- Conversa com {conv.phone[:7]}*** ---\n" + "\n".join(lines)
        )

    return "\n\n".join(sample_parts)[:14_000]  # slightly larger budget for 10 convs


def _safe_list(value, default: list) -> list:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        try:
            import ast
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed]
        except Exception:
            pass
        return [v.strip() for v in value.split(",") if v.strip()]
    return default


# ------------------------------------------------------------------
# Main extraction function
# ------------------------------------------------------------------

def extract_shadow_dna(
    conversations: list[Conversation],
    clinic_name: str,
    analyses=None,    # list[SemanticAnalysis] — for sentiment distribution
) -> ShadowDNA:
    """
    Extract the full Shadow DNA profile from all conversations.

    Args:
        conversations:  list of Conversation objects
        clinic_name:    display name of the clinic
        analyses:       optional list of SemanticAnalysis for sentiment stats

    Returns:
        ShadowDNA dataclass
    """
    dna = ShadowDNA()

    # 1. Quantitative (pure Python, always runs)
    quant = _compute_quantitative(conversations)
    dna.average_response_length_tokens = quant["average_response_length_tokens"]
    dna.emoji_frequency = quant["emoji_frequency"]
    dna.response_time_metrics = quant["response_time_metrics"]

    # 2. Sentiment distribution from existing analyses
    if analyses:
        pos = sum(1 for a in analyses if a.sentiment_score > 0.2)
        neg = sum(1 for a in analyses if a.sentiment_score < -0.2)
        neu = len(analyses) - pos - neg
        total = len(analyses) or 1
        dna.sentiment_score_distribution = {
            "positive": round(pos / total, 3),
            "neutral":  round(neu / total, 3),
            "negative": round(neg / total, 3),
        }

    # 3. LLM extraction
    if not _shadow_module:
        dna.error = "ShadowDNAModule not initialized. Call init_shadow_module() first."
        return dna

    sample = _build_sample(conversations)
    payment_mentions = _extract_payment_mentions(conversations)
    logger.info("Payment mentions pre-scanned: %d unique messages", payment_mentions.count("\n-") + 1)

    agg_lm = get_aggregate_lm()
    ctx = dspy.context(lm=agg_lm) if agg_lm else None

    try:
        if ctx:
            with ctx:
                pred = _shadow_module(
                    conversations_sample=sample,
                    clinic_name=clinic_name,
                    payment_mentions_raw=payment_mentions,
                )
        else:
            pred = _shadow_module(
                conversations_sample=sample,
                clinic_name=clinic_name,
                payment_mentions_raw=payment_mentions,
            )

        dna.tone_classification = str(pred.tone_classification).strip() or "Misto"
        dna.greeting_example = str(pred.greeting_example).strip()
        dna.closing_example = str(pred.closing_example).strip()
        dna.agent_suggested_name = str(pred.agent_suggested_name).strip()

        dna.personality_traits = _safe_list(pred.personality_traits, [])
        dna.forbidden_terms = _safe_list(pred.forbidden_terms, [])
        dna.handoff_keywords = _safe_list(pred.handoff_keywords, [])
        dna.handoff_situations = _safe_list(pred.handoff_situations, [])
        dna.local_procedures = _safe_list(pred.local_procedures, [])
        dna.local_insurances = _safe_list(pred.local_insurances, [])
        dna.local_neighborhoods = _safe_list(pred.local_neighborhoods, [])
        dna.local_payment_conditions = _safe_list(pred.local_payment_conditions, [])
        dna.unresolved_queries = _safe_list(pred.unresolved_queries, [])
        dna.common_complaints = _safe_list(pred.common_complaints, [])

        # attendance_flow_steps: expects list of dicts with "step" and "example"
        raw_flow = pred.attendance_flow_steps
        if isinstance(raw_flow, list):
            validated = []
            for step in raw_flow:
                if isinstance(step, dict) and "step" in step and "example" in step:
                    validated.append({
                        "step": str(step["step"]).strip(),
                        "example": str(step["example"]).strip(),
                    })
            dna.attendance_flow_steps = validated[:6]
        elif isinstance(raw_flow, str):
            import ast as _ast
            try:
                parsed = _ast.literal_eval(raw_flow)
                if isinstance(parsed, list):
                    dna.attendance_flow_steps = [
                        {"step": str(s.get("step","")).strip(),
                         "example": str(s.get("example","")).strip()}
                        for s in parsed
                        if isinstance(s, dict) and "step" in s and "example" in s
                    ][:6]
            except Exception:
                dna.attendance_flow_steps = []

        # operating_hours: dict with open/close/days or None
        raw_hours = pred.operating_hours
        if isinstance(raw_hours, dict) and raw_hours:
            dna.operating_hours = raw_hours
        elif isinstance(raw_hours, str):
            import ast as _ast
            try:
                parsed = _ast.literal_eval(raw_hours)
                dna.operating_hours = parsed if isinstance(parsed, dict) and parsed else None
            except Exception:
                dna.operating_hours = None
        else:
            dna.operating_hours = None

        # source_signals: dict channel -> list[str]
        raw_signals = pred.source_signals
        if isinstance(raw_signals, dict):
            dna.source_signals = {
                str(channel): _safe_list(phrases, [])
                for channel, phrases in raw_signals.items()
                if str(channel).strip()
            }
        elif isinstance(raw_signals, str):
            import ast as _ast
            try:
                parsed = _ast.literal_eval(raw_signals)
                if isinstance(parsed, dict):
                    dna.source_signals = {
                        str(channel): _safe_list(phrases, [])
                        for channel, phrases in parsed.items()
                        if str(channel).strip()
                    }
            except Exception:
                dna.source_signals = {}
        else:
            dna.source_signals = {}

    except Exception as e:
        logger.warning("Shadow DNA LLM extraction failed: %s", e)
        dna.error = str(e)

    # Pure-Python insurance mention counts (runs even if LLM extraction failed)
    if dna.local_insurances and conversations:
        all_text_lower = " ".join(
            m.content.lower()
            for conv in conversations
            for m in conv.messages
        )
        raw_counts = {
            ins: all_text_lower.count(ins.lower())
            for ins in dna.local_insurances
            if all_text_lower.count(ins.lower()) > 0
        }
        dna.insurance_mention_counts = dict(
            sorted(raw_counts.items(), key=lambda x: x[1], reverse=True)
        )

    return dna
