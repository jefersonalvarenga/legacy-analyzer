"""
financial_kpis.py
-----------------
Computes financial impact KPIs for the executive report.

Formula:
  opportunity_loss = leads_lost × ticket_medio
  potential_recovery = opportunity_loss × 0.30  (conservative 30% AI uplift)

ticket_medio sources (in priority order):
  1. user_input  — provided via CLI flag --ticket-medio
  2. llm_estimate — estimated by DSPy based on procedures/profile detected

The source is always flagged in the output for transparency.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import dspy

from analyzer.outcome_detection import OutcomeSummary

logger = logging.getLogger(__name__)

AI_RECOVERY_RATE = 0.30   # conservative: AI recovers 30% of lost leads


# ------------------------------------------------------------------
# DSPy Signature — ticket medio estimator
# ------------------------------------------------------------------

class TicketMedioSignature(dspy.Signature):
    """
    Estime o ticket médio (valor médio por procedimento/consulta em BRL) de uma
    clínica com base nos procedimentos detectados nas conversas e no perfil geral
    da clínica. Seja conservador e realista. Responda apenas com o valor numérico
    em BRL (sem símbolo de moeda, sem texto adicional).
    """
    procedures: str = dspy.InputField(
        desc="Lista de procedimentos detectados nas conversas da clínica"
    )
    clinic_profile: str = dspy.InputField(
        desc="Perfil da clínica: nome, bairros mencionados, convênios, tom de atendimento"
    )

    ticket_medio_brl: float = dspy.OutputField(
        desc="Valor estimado do ticket médio em BRL (ex: 350.0)"
    )
    reasoning: str = dspy.OutputField(
        desc="Justificativa de uma frase para a estimativa"
    )


class TicketMedioEstimator(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(TicketMedioSignature)

    def forward(self, procedures: str, clinic_profile: str):
        return self.predict(procedures=procedures, clinic_profile=clinic_profile)


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------

@dataclass
class FinancialKPIs:
    ticket_medio: float = 0.0
    ticket_medio_source: str = "llm_estimate"   # "user_input" | "llm_estimate"
    ticket_medio_reasoning: str = ""

    leads_lost: int = 0
    opportunity_loss_value: float = 0.0
    potential_recovery_value: float = 0.0

    error: Optional[str] = None


# ------------------------------------------------------------------
# Module instance
# ------------------------------------------------------------------

_ticket_estimator: Optional[TicketMedioEstimator] = None


def init_financial_module():
    global _ticket_estimator
    _ticket_estimator = TicketMedioEstimator()


def _safe_float(value, default: float) -> float:
    try:
        return max(0.0, float(str(value).replace(",", ".").strip()))
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------
# Main computation
# ------------------------------------------------------------------

def compute_financial_kpis(
    outcome_summary: OutcomeSummary,
    shadow_dna=None,          # ShadowDNA — used for LLM ticket estimation
    clinic_name: str = "",
    ticket_medio_override: Optional[float] = None,
) -> FinancialKPIs:
    """
    Compute financial KPIs.

    Args:
        outcome_summary:        aggregated outcome results
        shadow_dna:             ShadowDNA (for procedures/profile context)
        clinic_name:            display name of the clinic
        ticket_medio_override:  if provided (user input), skips LLM estimation

    Returns:
        FinancialKPIs dataclass
    """
    kpis = FinancialKPIs()
    kpis.leads_lost = outcome_summary.leads_lost

    # --- Ticket Medio ---
    if ticket_medio_override is not None and ticket_medio_override > 0:
        kpis.ticket_medio = ticket_medio_override
        kpis.ticket_medio_source = "user_input"
        kpis.ticket_medio_reasoning = "Valor informado pelo cliente."
    else:
        # LLM estimation
        if not _ticket_estimator:
            kpis.error = "TicketMedioEstimator not initialized."
            return kpis

        procedures_str = ", ".join(
            (shadow_dna.local_procedures[:10] if shadow_dna else [])
        ) or "procedimentos não identificados"

        profile_parts = [f"Clínica: {clinic_name}"]
        if shadow_dna:
            if shadow_dna.local_neighborhoods:
                profile_parts.append(f"Bairros: {', '.join(shadow_dna.local_neighborhoods[:3])}")
            if shadow_dna.local_insurances:
                profile_parts.append(f"Convênios: {', '.join(shadow_dna.local_insurances[:3])}")
            profile_parts.append(f"Tom: {shadow_dna.tone_classification}")

        clinic_profile = ". ".join(profile_parts)

        try:
            pred = _ticket_estimator(
                procedures=procedures_str,
                clinic_profile=clinic_profile,
            )
            kpis.ticket_medio = _safe_float(pred.ticket_medio_brl, 300.0)
            kpis.ticket_medio_source = "llm_estimate"
            kpis.ticket_medio_reasoning = str(pred.reasoning).strip()
        except Exception as e:
            logger.warning("Ticket medio estimation failed: %s", e)
            kpis.ticket_medio = 300.0   # fallback: R$300 conservative default
            kpis.ticket_medio_source = "llm_estimate"
            kpis.ticket_medio_reasoning = "Estimativa padrão (erro na estimativa por LLM)."
            kpis.error = str(e)

    # --- Opportunity Loss ---
    kpis.opportunity_loss_value = round(kpis.leads_lost * kpis.ticket_medio, 2)
    kpis.potential_recovery_value = round(
        kpis.opportunity_loss_value * AI_RECOVERY_RATE, 2
    )

    logger.info(
        "Financial KPIs: ticket=R$%.0f (%s), leads_lost=%d, loss=R$%.0f, recovery=R$%.0f",
        kpis.ticket_medio,
        kpis.ticket_medio_source,
        kpis.leads_lost,
        kpis.opportunity_loss_value,
        kpis.potential_recovery_value,
    )

    return kpis
