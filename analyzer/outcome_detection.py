"""
outcome_detection.py
--------------------
DSPy module to classify the outcome of each WhatsApp conversation.

Outcomes:
  - agendado       : Patient confirmed or scheduled an appointment
  - ghosting       : Patient stopped responding abruptly
  - objecao_ativa  : Patient expressed a clear objection (price, time, etc.)
  - pendente       : Conversation still in progress, no clear outcome
  - outro          : Uncategorized

Returns outcome label + confidence score (0.0–1.0) + reasoning.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import dspy

logger = logging.getLogger(__name__)

VALID_OUTCOMES = {"agendado", "ghosting", "objecao_ativa", "pendente", "outro"}


# ------------------------------------------------------------------
# DSPy Signature
# ------------------------------------------------------------------

class OutcomeSignature(dspy.Signature):
    """
    Analise a conversa de WhatsApp entre uma clínica e um paciente e classifique
    o desfecho (outcome) da conversa. Leia TODAS as mensagens com atenção,
    especialmente as últimas.

    Critérios de classificação:
    - agendado: paciente confirmou, agendou ou marcou uma consulta/procedimento
    - ghosting: paciente parou de responder sem motivo aparente (última mensagem
      é da clínica sem resposta, ou paciente sumiu após iniciar interesse)
    - objecao_ativa: paciente expressou objeção COMERCIAL clara. Inclui SOMENTE:
        * preço alto ("está caro", "não tenho dinheiro", "quanto custa?", "muito caro")
        * cobertura de convênio ("não aceita meu plano", "é particular?", "não tem o meu convênio")
        * forma de pagamento ("não consigo parcelar", "só aceita à vista?")
        * disponibilidade de serviço ("não faz esse procedimento?", "não atende isso?")
        * localização/distância ("fica muito longe", "não tem perto de mim")
      NÃO classifique como objecao_ativa:
        * queixas de saúde do paciente ("estou com dor", "tive uma crise de labirinto")
        * conflitos de agenda pontuais ("não posso às 17h30", "não consigo sair hoje")
        * respostas vagas sem recusa comercial clara
        * ghosting (paciente simplesmente sumiu)
    - pendente: conversa ainda ativa sem desfecho claro, ou muito recente
    - outro: desfecho que não se encaixa nos anteriores (ex: tirou dúvida, reclamação
      resolvida, cancelamento por motivo de saúde, etc.)

    Responda em português.
    """
    conversation: str = dspy.InputField(desc="Transcrição completa da conversa do WhatsApp")
    clinic_name: str = dspy.InputField(desc="Nome da clínica na conversa")

    outcome: str = dspy.OutputField(
        desc="Um dos: agendado, ghosting, objecao_ativa, pendente, outro"
    )
    confidence_score: float = dspy.OutputField(
        desc="Confiança na classificação de 0.0 (incerto) a 1.0 (certeza absoluta)"
    )
    reasoning: str = dspy.OutputField(
        desc="Uma frase explicando o motivo da classificação (em português)"
    )
    main_objection: str = dspy.OutputField(
        desc=(
            "Se outcome=objecao_ativa, descreva a objeção COMERCIAL específica em 1 frase curta. "
            "Seja específico ao contexto desta conversa — não genérico. "
            "BONS EXEMPLOS: 'Plano Sulamérica não aceito', 'Avaliação cobrada R$180 considerada cara', "
            "'Clínica não realiza o procedimento solicitado', 'Atendimento apenas particular'. "
            "Se outcome NÃO for objecao_ativa, retorne string vazia."
        )
    )


# ------------------------------------------------------------------
# DSPy Module
# ------------------------------------------------------------------

class OutcomeDetector(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(OutcomeSignature)

    def forward(self, conversation: str, clinic_name: str):
        return self.predict(conversation=conversation, clinic_name=clinic_name)


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------

@dataclass
class OutcomeResult:
    outcome: str = "outro"
    confidence_score: float = 0.5
    reasoning: str = ""
    main_objection: str = ""
    error: Optional[str] = None


# ------------------------------------------------------------------
# Module instance (initialized after configure_lm is called)
# ------------------------------------------------------------------

_outcome_module: Optional[OutcomeDetector] = None


def init_outcome_module():
    """Call after dspy.configure() has been set up."""
    global _outcome_module
    _outcome_module = OutcomeDetector()


def _safe_float(value, default: float) -> float:
    try:
        v = float(value)
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return default


def _safe_outcome(value: str) -> str:
    cleaned = str(value).strip().lower().replace(" ", "_")
    return cleaned if cleaned in VALID_OUTCOMES else "outro"


def detect_outcome(messages, clinic_name: str) -> OutcomeResult:
    """
    Classify the outcome of a single conversation.

    Args:
        messages:     list of Message objects (from parser.py)
        clinic_name:  display name of the clinic

    Returns:
        OutcomeResult dataclass
    """
    if not _outcome_module:
        return OutcomeResult(error="OutcomeDetector not initialized. Call init_outcome_module() first.")

    from analyzer.dspy_pipeline import _conversation_to_text, _truncate_conversation
    conv_text = _truncate_conversation(_conversation_to_text(messages))

    try:
        pred = _outcome_module(conversation=conv_text, clinic_name=clinic_name)
        return OutcomeResult(
            outcome=_safe_outcome(pred.outcome),
            confidence_score=_safe_float(pred.confidence_score, 0.5),
            reasoning=str(pred.reasoning).strip(),
            main_objection=str(pred.main_objection).strip(),
        )
    except Exception as e:
        logger.warning("Outcome detection failed: %s", e)
        return OutcomeResult(error=str(e))


# ------------------------------------------------------------------
# Aggregation helper
# ------------------------------------------------------------------

@dataclass
class OutcomeSummary:
    agendado: int = 0
    ghosting: int = 0
    objecao_ativa: int = 0
    pendente: int = 0
    outro: int = 0
    conversion_rate: float = 0.0
    leads_lost: int = 0        # ghosting + objecao_ativa
    common_objections: list[str] = None

    def __post_init__(self):
        if self.common_objections is None:
            self.common_objections = []


def aggregate_outcomes(results: list[OutcomeResult]) -> OutcomeSummary:
    """Roll up per-conversation outcomes into a summary."""
    summary = OutcomeSummary()
    objections = []

    for r in results:
        match r.outcome:
            case "agendado":      summary.agendado += 1
            case "ghosting":      summary.ghosting += 1
            case "objecao_ativa": summary.objecao_ativa += 1
            case "pendente":      summary.pendente += 1
            case _:               summary.outro += 1

        if r.outcome == "objecao_ativa" and r.main_objection:
            objections.append(r.main_objection)

    total = len(results)
    summary.leads_lost = summary.ghosting + summary.objecao_ativa
    summary.conversion_rate = round(summary.agendado / total, 3) if total > 0 else 0.0

    # Deduplicate objections preserving frequency order
    seen = set()
    unique_objections = []
    for obj in objections:
        if obj.lower() not in seen:
            seen.add(obj.lower())
            unique_objections.append(obj)
    summary.common_objections = unique_objections[:10]

    return summary
