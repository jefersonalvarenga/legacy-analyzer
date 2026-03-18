"""
playbook_inference.py
---------------------
Forensic playbook inference — reads conversations without a pre-defined
template and produces a free-form description of how the clinic operates.

Also provides per-service playbook extraction (ServicePlaybookSignature)
that builds a structured playbook for each detected service, based on
conversations where that service was mentioned and the outcome was
"agendado". Admin can supply reference_conversation_ids to restrict
the source conversations.

The LA acts as a forensic investigator: no prior hypothesis, no external
references. It observes the conversations and describes what it finds.

Output schema (clinic_playbook):
  {
    "reasoning": "Free-text in first person of the clinic. E.g.: 'Nossa clinica...'",
    "phases": [
      {
        "name": "free name inferred by LA — e.g. 'Conexao antes da avaliacao'",
        "phase_intent": "first_contact | pre_conversion | conversion |
                         post_conversion | retention | management",
        "description": "what happens in this phase according to the conversations",
        "elements": [
          {
            "element": "canonical vocabulary",
            "initiated_by": "sofia | patient",
            "trigger_signals": ["real patient phrases"],
            "blocked_by": ["already_sent | evaluation_not_done |
                            price_not_asked | appointment_confirmed"],
            "real_example": "string | null"
          }
        ]
      }
    ],
    "observations": "Behaviours that did not fit any phase — exceptions,
                     anomalies, volume warning if < 10 conversations"
  }

Canonical vocabulary for element:
  greeting, identification, connection, active_listening, technical_details,
  before_after, insurances, pricing_payment, objections, scheduling_slots,
  confirmation, closing

Canonical vocabulary for phase_intent:
  first_contact, pre_conversion, conversion, post_conversion, retention,
  management
"""

import logging
from typing import Optional

import dspy

from analyzer.parser import Conversation
from analyzer.dspy_pipeline import get_aggregate_lm

logger = logging.getLogger(__name__)

# Rough token estimate: 1 token ~= 4 chars (pt-BR average)
_CHARS_PER_TOKEN = 4
_SAMPLE_BUDGET = 14_000  # chars


# ------------------------------------------------------------------
# DSPy Signature
# ------------------------------------------------------------------

class ClinicPlaybookSignature(dspy.Signature):
    """
    Voce e um investigador forense de comunicacao. Leia as conversas da clinica
    SEM hipotese previa e produza um relatorio descritivo de como a clinica opera.

    REGRAS OBRIGATORIAS:
    - 'reasoning' DEVE ser escrito em primeira pessoa da clinica
      (ex: 'Nossa clinica opera num modelo consultivo...').
    - NUNCA mencionar concorrentes, benchmarks ou referencias externas em 'reasoning'.
    - 'phases' tem nome LIVRE — voce nomeia com base no que observou, nao em template.
    - Use apenas vocabulario canonico para 'element':
      greeting, identification, connection, active_listening, technical_details,
      before_after, insurances, pricing_payment, objections, scheduling_slots,
      confirmation, closing
    - Use apenas vocabulario canonico para 'phase_intent':
      first_contact, pre_conversion, conversion, post_conversion, retention,
      management
    - Se nao conseguir inferir fases, retorne phases=[] e explique em 'observations'.
    - Responda em portugues.
    """

    conversations_sample: str = dspy.InputField(
        desc=(
            "Amostra de conversas com desfecho=agendado (ou todas se volume insuficiente). "
            "Formato: blocos por conversa, mensagens com prefixo [CLINIC] ou [PATIENT]."
        )
    )
    clinic_name: str = dspy.InputField(desc="Nome da clinica")
    total_conversations_count: int = dspy.InputField(
        desc="Total de conversas analisadas (antes da filtragem por desfecho)"
    )

    reasoning: str = dspy.OutputField(
        desc=(
            "Texto livre em primeira pessoa da clinica descrevendo como ela opera. "
            "Ex: 'Nossa clinica opera num modelo consultivo — a avaliacao nao e triagem, "
            "e o momento de venda...' "
            "NUNCA mencionar concorrentes ou referencias externas."
        )
    )
    phases: list = dspy.OutputField(
        desc=(
            "Lista de fases do playbook inferidas das conversas. "
            "Cada fase e um dict com: "
            "  'name' (str — nome livre, ex: 'Conexao antes da avaliacao'), "
            "  'phase_intent' (str — um de: first_contact, pre_conversion, conversion, "
            "    post_conversion, retention, management), "
            "  'description' (str — o que acontece nessa fase), "
            "  'elements' (lista de dicts com: element, initiated_by, "
            "    trigger_signals, blocked_by, real_example). "
            "Se nao houver fases identificaveis, retorne []."
        )
    )
    observations: str = dspy.OutputField(
        desc=(
            "Comportamentos que nao se encaixaram em nenhuma fase — excecoes, anomalias. "
            "Inclua aviso de volume insuficiente se total_conversations_count < 10."
        )
    )


# ------------------------------------------------------------------
# DSPy Module
# ------------------------------------------------------------------

class ClinicPlaybookModule(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(ClinicPlaybookSignature)

    def forward(
        self,
        conversations_sample: str,
        clinic_name: str,
        total_conversations_count: int,
    ):
        return self.predict(
            conversations_sample=conversations_sample,
            clinic_name=clinic_name,
            total_conversations_count=total_conversations_count,
        )


# ------------------------------------------------------------------
# Module instance (lazy init)
# ------------------------------------------------------------------

_playbook_module: Optional[ClinicPlaybookModule] = None


def init_playbook_module():
    global _playbook_module
    _playbook_module = ClinicPlaybookModule()
    init_service_playbook_module()


# ------------------------------------------------------------------
# Sample builder (agendado-first)
# ------------------------------------------------------------------

def _build_playbook_sample(conversations: list[Conversation], max_convs: int = 10) -> str:
    """
    Build a text sample from conversations.
    Takes up to max_convs, first 10 + last 10 messages each.
    """
    sample = conversations[:max_convs]
    parts = []
    for conv in sample:
        msgs = conv.messages
        excerpt = msgs[:10] + (msgs[-10:] if len(msgs) > 20 else [])
        lines = [
            f"[{m.sender_type.upper()}] {m.sender}: {m.content}"
            for m in excerpt
        ]
        parts.append(
            f"--- Conversa com {conv.phone[:7]}*** ---\n" + "\n".join(lines)
        )
    return "\n\n".join(parts)[:_SAMPLE_BUDGET]


# ------------------------------------------------------------------
# Phase / element validators
# ------------------------------------------------------------------

_VALID_PHASE_INTENTS = frozenset({
    "first_contact", "pre_conversion", "conversion",
    "post_conversion", "retention", "management",
})

_VALID_ELEMENTS = frozenset({
    "greeting", "identification", "connection", "active_listening",
    "technical_details", "before_after", "insurances", "pricing_payment",
    "objections", "scheduling_slots", "confirmation", "closing",
})


def _validate_element(raw: dict) -> dict:
    """Normalise a single element dict, keeping only known fields."""
    element = str(raw.get("element", "")).strip()
    if element not in _VALID_ELEMENTS:
        element = "greeting"  # safe fallback

    initiated_by = str(raw.get("initiated_by", "sofia")).strip()
    if initiated_by not in ("sofia", "patient"):
        initiated_by = "sofia"

    trigger_signals = raw.get("trigger_signals", [])
    if not isinstance(trigger_signals, list):
        trigger_signals = []

    blocked_by = raw.get("blocked_by", [])
    if not isinstance(blocked_by, list):
        blocked_by = []

    real_example = raw.get("real_example")
    if real_example is not None:
        real_example = str(real_example).strip() or None

    return {
        "element": element,
        "initiated_by": initiated_by,
        "trigger_signals": [str(s).strip() for s in trigger_signals if str(s).strip()],
        "blocked_by": [str(b).strip() for b in blocked_by if str(b).strip()],
        "real_example": real_example,
    }


def _validate_phase(raw: dict) -> dict:
    """Normalise a single phase dict."""
    name = str(raw.get("name", "Fase sem nome")).strip()

    phase_intent = str(raw.get("phase_intent", "")).strip()
    if phase_intent not in _VALID_PHASE_INTENTS:
        phase_intent = "first_contact"

    description = str(raw.get("description", "")).strip()

    raw_elements = raw.get("elements", [])
    if not isinstance(raw_elements, list):
        raw_elements = []
    elements = [_validate_element(e) for e in raw_elements if isinstance(e, dict)]

    return {
        "name": name,
        "phase_intent": phase_intent,
        "description": description,
        "elements": elements,
    }


def _validate_phases(raw) -> list:
    """Validate and normalise the phases output from DSPy."""
    if not isinstance(raw, list):
        if isinstance(raw, str):
            import ast
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    raw = parsed
                else:
                    return []
            except Exception:
                return []
        else:
            return []
    return [_validate_phase(p) for p in raw if isinstance(p, dict)]


# ------------------------------------------------------------------
# Main extraction function
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# ServicePlaybook — per-service playbook
# ------------------------------------------------------------------

class ServicePlaybookSignature(dspy.Signature):
    """
    Voce e um especialista em vendas consultivas para clinicas de saude.
    A partir de conversas onde o servico foi mencionado e o paciente agendou,
    extraia o playbook de atendimento especifico para esse servico.

    REGRAS OBRIGATORIAS:
    - Use apenas vocabulario canonico para 'element':
      greeting, identification, connection, active_listening, technical_details,
      before_after, insurances, pricing_payment, objections, scheduling_slots,
      confirmation, closing
    - 'trigger_signals' DEVEM ser frases REAIS dos pacientes copiadas das conversas.
    - 'real_example' DEVE ser mensagem REAL da clinica copiada das conversas, ou null.
    - 'blocked_by' usa vocabulario canonico:
      already_sent, evaluation_not_done, price_not_asked, appointment_confirmed
    - 'default_flow': 'evaluation_first' se o servico exige avaliacao antes do
      agendamento definitivo, 'direct_booking' se pode agendar diretamente.
    - Inclua apenas elementos que realmente apareceram nas conversas.
    - Responda em portugues.
    """

    service_name: str = dspy.InputField(desc="Nome do servico/procedimento a analisar")
    conversations_sample: str = dspy.InputField(
        desc=(
            "Conversas onde esse servico foi mencionado e o desfecho foi agendado. "
            "Formato: blocos por conversa, mensagens com prefixo [CLINIC] ou [PATIENT]."
        )
    )

    requires_evaluation: bool = dspy.OutputField(
        desc=(
            "True se o servico tipicamente exige uma consulta de avaliacao antes do "
            "procedimento principal, False se pode ser agendado diretamente."
        )
    )
    default_flow: str = dspy.OutputField(
        desc=(
            "Fluxo padrao para este servico: "
            "'evaluation_first' se exige avaliacao antes do agendamento definitivo, "
            "'direct_booking' se o agendamento e feito diretamente sem avaliacao previa."
        )
    )
    elements: list = dspy.OutputField(
        desc=(
            "Lista de elementos do playbook para este servico. "
            "Cada item e um dict com: "
            "  'element' (str — um do vocabulario canonico), "
            "  'initiated_by' (str — 'sofia' ou 'patient'), "
            "  'trigger_signals' (lista de frases REAIS do paciente que ativam este elemento), "
            "  'blocked_by' (lista usando vocabulario canonico: already_sent, "
            "    evaluation_not_done, price_not_asked, appointment_confirmed), "
            "  'real_example' (str — mensagem REAL da clinica copiada das conversas, ou null). "
            "Inclua apenas elementos que realmente apareceram nas conversas. "
            "Formato: [{\"element\": \"connection\", \"initiated_by\": \"sofia\", "
            "\"trigger_signals\": [...], \"blocked_by\": [...], \"real_example\": \"...\"}]"
        )
    )


class ServicePlaybookModule(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(ServicePlaybookSignature)

    def forward(self, service_name: str, conversations_sample: str):
        return self.predict(
            service_name=service_name,
            conversations_sample=conversations_sample,
        )


# Module instance (shared, lazy init)
_service_playbook_module: Optional[ServicePlaybookModule] = None


def init_service_playbook_module():
    global _service_playbook_module
    _service_playbook_module = ServicePlaybookModule()


# ------------------------------------------------------------------
# Service playbook helpers
# ------------------------------------------------------------------

_VALID_BLOCKED_BY = frozenset({
    "already_sent", "evaluation_not_done", "price_not_asked", "appointment_confirmed",
})

_VALID_DEFAULT_FLOWS = frozenset({"evaluation_first", "direct_booking"})


def _validate_service_element(raw: dict) -> dict:
    """Normalise a single service playbook element dict."""
    element = str(raw.get("element", "")).strip()
    if element not in _VALID_ELEMENTS:
        element = "connection"  # safe fallback

    initiated_by = str(raw.get("initiated_by", "sofia")).strip()
    if initiated_by not in ("sofia", "patient"):
        initiated_by = "sofia"

    trigger_signals = raw.get("trigger_signals", [])
    if not isinstance(trigger_signals, list):
        trigger_signals = []

    blocked_by_raw = raw.get("blocked_by", [])
    if not isinstance(blocked_by_raw, list):
        blocked_by_raw = []
    # Keep only canonical values
    blocked_by = [str(b).strip() for b in blocked_by_raw if str(b).strip() in _VALID_BLOCKED_BY]

    real_example = raw.get("real_example")
    if real_example is not None:
        real_example = str(real_example).strip() or None

    return {
        "element": element,
        "initiated_by": initiated_by,
        "trigger_signals": [str(s).strip() for s in trigger_signals if str(s).strip()],
        "blocked_by": blocked_by,
        "real_example": real_example,
    }


def _validate_service_elements(raw) -> list:
    """Validate and normalise the elements output from ServicePlaybookSignature."""
    if not isinstance(raw, list):
        if isinstance(raw, str):
            import ast
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    raw = parsed
                else:
                    return []
            except Exception:
                return []
        else:
            return []
    return [_validate_service_element(e) for e in raw if isinstance(e, dict)]


def _service_mentioned_in_conversation(conv: "Conversation", service_name: str) -> bool:
    """Return True if any message in the conversation mentions the service name."""
    service_lower = service_name.lower()
    for msg in conv.messages:
        if service_lower in msg.content.lower():
            return True
    return False


def _build_service_sample(conversations: list, max_chars: int = _SAMPLE_BUDGET) -> str:
    """Build a conversation sample string from the given conversations list."""
    parts = []
    for conv in conversations:
        msgs = conv.messages
        excerpt = msgs[:10] + (msgs[-10:] if len(msgs) > 20 else [])
        lines = [
            f"[{'CLINIC' if m.sender_type == 'clinic' else 'PATIENT'}] {m.sender}: {m.content}"
            for m in excerpt
        ]
        parts.append(
            f"--- Conversa com {conv.phone[:7]}*** ---\n" + "\n".join(lines)
        )
    return "\n\n".join(parts)[:max_chars]


def extract_service_playbooks(
    conversations: list,
    services: list[str],
    reference_ids: "list[str] | None" = None,
    outcome_results=None,  # list[OutcomeResult] — parallel to conversations
) -> list[dict]:
    """
    Extract a per-service playbook for each service in `services`.

    For each service:
      - Filters conversations where the service is mentioned AND outcome == "agendado"
      - If reference_ids is provided, restricts to those conversation phones/ids
      - Skips service if fewer than 2 matching conversations found
      - Calls ServicePlaybookModule to extract elements
      - Returns the assembled playbook dict

    Args:
        conversations:   list of Conversation objects
        services:        list of service/procedure name strings
        reference_ids:   optional list of conversation identifiers (phone numbers or
                         source_filename values) to restrict the source
        outcome_results: optional list of OutcomeResult parallel to conversations

    Returns:
        list of playbook dicts — one per service with sufficient data.
        Services with < 2 matching conversations are omitted.
    """
    if not _service_playbook_module:
        logger.warning(
            "ServicePlaybookModule not initialized — call init_service_playbook_module() first."
        )
        return []

    if not services:
        return []

    # Build a set of agendado conversation indices for fast lookup
    agendado_set: set[int] = set()
    if outcome_results and len(outcome_results) == len(conversations):
        for idx, outcome in enumerate(outcome_results):
            if getattr(outcome, "outcome", None) == "agendado":
                agendado_set.add(idx)
    else:
        # No outcome info — treat all as eligible
        agendado_set = set(range(len(conversations)))

    # Build reference set if supplied
    ref_set: "set[str] | None" = None
    if reference_ids:
        ref_set = {str(r).strip() for r in reference_ids if str(r).strip()}

    result: list[dict] = []

    agg_lm = get_aggregate_lm()
    ctx = dspy.context(lm=agg_lm) if agg_lm else None

    for service in services:
        if not service or not service.strip():
            continue

        # Filter: agendado + service mentioned
        eligible: list = []
        for idx, conv in enumerate(conversations):
            if idx not in agendado_set:
                continue
            # Reference filter
            if ref_set is not None:
                conv_id = str(getattr(conv, "phone", "") or getattr(conv, "source_filename", ""))
                if conv_id not in ref_set:
                    continue
            # Service mention filter
            if not _service_mentioned_in_conversation(conv, service):
                continue
            eligible.append(conv)

        if len(eligible) < 2:
            logger.info(
                "Service '%s': only %d eligible conversations (< 2) — skipping.",
                service, len(eligible),
            )
            continue

        logger.info(
            "Service '%s': %d eligible conversations — inferring playbook.",
            service, len(eligible),
        )

        sample = _build_service_sample(eligible)

        try:
            if ctx:
                with ctx:
                    pred = _service_playbook_module(
                        service_name=service,
                        conversations_sample=sample,
                    )
            else:
                pred = _service_playbook_module(
                    service_name=service,
                    conversations_sample=sample,
                )

            # Normalise requires_evaluation
            req_eval = pred.requires_evaluation
            if isinstance(req_eval, str):
                req_eval = req_eval.strip().lower() in ("true", "yes", "sim", "1")
            else:
                req_eval = bool(req_eval)

            # Normalise default_flow
            raw_flow = str(pred.default_flow).strip().lower()
            default_flow = raw_flow if raw_flow in _VALID_DEFAULT_FLOWS else "direct_booking"

            elements = _validate_service_elements(pred.elements)

            result.append({
                "service_name": service.strip(),
                "requires_evaluation": req_eval,
                "default_flow": default_flow,
                "elements": elements,
            })

        except Exception as e:
            logger.warning("ServicePlaybook LLM extraction failed for '%s': %s", service, e)
            continue

    return result


def extract_clinic_playbook(
    conversations: list[Conversation],
    clinic_name: str,
    outcome_results=None,   # list[OutcomeResult] — parallel to conversations
) -> Optional[dict]:
    """
    Forensic playbook inference from WhatsApp conversations.

    Filters for conversations with outcome == "agendado" as the primary
    source. Falls back to all conversations if fewer than 3 were scheduled.

    Args:
        conversations:   list of Conversation objects
        clinic_name:     display name of the clinic
        outcome_results: optional list of OutcomeResult (parallel to conversations)

    Returns:
        dict conforming to clinic_playbook schema, or None on error.
    """
    if not _playbook_module:
        logger.warning("ClinicPlaybookModule not initialized. Call init_playbook_module() first.")
        return None

    total_count = len(conversations)
    observations_prefix = ""

    # Filter for agendado conversations if outcome_results provided
    agendado_convs: list[Conversation] = []
    if outcome_results and len(outcome_results) == len(conversations):
        for conv, outcome in zip(conversations, outcome_results):
            if getattr(outcome, "outcome", None) == "agendado":
                agendado_convs.append(conv)
    else:
        agendado_convs = []

    if len(agendado_convs) >= 3:
        source_convs = agendado_convs
    else:
        source_convs = conversations
        if len(agendado_convs) < 3:
            observations_prefix = (
                "Volume insuficiente de conversas com desfecho=agendado "
                f"({len(agendado_convs)} encontradas, minimo 3). "
                "Playbook inferido a partir de todas as conversas disponiveis. "
            )

    if total_count < 10:
        volume_warning = (
            f"Atencao: apenas {total_count} conversas analisadas (recomendado >= 10). "
            "Inferencias podem ser menos confiáveis. "
        )
        observations_prefix = volume_warning + observations_prefix

    sample = _build_playbook_sample(source_convs)

    agg_lm = get_aggregate_lm()
    ctx = dspy.context(lm=agg_lm) if agg_lm else None

    try:
        if ctx:
            with ctx:
                pred = _playbook_module(
                    conversations_sample=sample,
                    clinic_name=clinic_name,
                    total_conversations_count=total_count,
                )
        else:
            pred = _playbook_module(
                conversations_sample=sample,
                clinic_name=clinic_name,
                total_conversations_count=total_count,
            )

        reasoning = str(pred.reasoning).strip()
        phases = _validate_phases(pred.phases)
        observations_raw = str(pred.observations).strip()

        observations = (observations_prefix + observations_raw).strip()
        if not observations:
            observations = "Nenhuma observacao adicional."

        return {
            "reasoning": reasoning,
            "phases": phases,
            "observations": observations,
        }

    except Exception as e:
        logger.warning("ClinicPlaybook LLM extraction failed: %s", e)
        return None
