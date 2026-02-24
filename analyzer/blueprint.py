"""
blueprint.py
------------
Assembles the Implementation Blueprint JSON from all analysis results.

Consumes:
  - ShadowDNA
  - OutcomeSummary
  - FinancialKPIs
  - AggregatedMetrics
  - list[SemanticAnalysis]  (for RAG efficiency score)

Produces:
  - dict conforming to implementation_blueprint_schema.json
  - Saved as blueprint_<client_slug>_<timestamp>.json

The blueprint is the "plug & play" config handed to n8n for
zero-touch agent provisioning.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from analyzer.shadow_dna import ShadowDNA
from analyzer.outcome_detection import OutcomeSummary
from analyzer.financial_kpis import FinancialKPIs
from analyzer.metrics import AggregatedMetrics

logger = logging.getLogger(__name__)

ANALYZER_VERSION = "0.2.0"


def _compute_rag_efficiency(
    shadow_dna: ShadowDNA,
    analyses,           # list[SemanticAnalysis]
) -> float:
    """
    RAG Efficiency Score = % of unique patient questions that are
    answerable from the detected knowledge base.

    Proxy formula:
      resolved = total_unique_queries - len(unresolved_queries)
      score = resolved / total_unique_queries * 100

    We estimate total_unique_queries as the union of all topics
    detected across conversations.
    """
    all_topics: set[str] = set()
    if analyses:
        for a in analyses:
            all_topics.update(t.lower() for t in a.topics)

    total_unique = len(all_topics) or 1
    unresolved = len(shadow_dna.unresolved_queries)
    resolved = max(0, total_unique - unresolved)
    score = round((resolved / total_unique) * 100, 1)
    return min(score, 100.0)


def build_blueprint(
    client_slug: str,
    client_name: str,
    shadow_dna: ShadowDNA,
    outcome_summary: OutcomeSummary,
    financial_kpis: FinancialKPIs,
    agg_metrics: AggregatedMetrics,
    analyses=None,          # list[SemanticAnalysis]
    generated_at: Optional[datetime] = None,
) -> dict:
    """
    Assemble and return the full blueprint as a Python dict.
    Conforms to implementation_blueprint_schema.json.
    """
    if generated_at is None:
        generated_at = datetime.utcnow()

    rag_score = _compute_rag_efficiency(shadow_dna, analyses or [])
    shadow_dna.rag_efficiency_score = rag_score

    blueprint = {
        "metadata": {
            "client_slug": client_slug,
            "client_name": client_name,
            "generated_at": generated_at.isoformat() + "Z",
            "analyzer_version": ANALYZER_VERSION,
            "conversation_count": agg_metrics.total_conversations,
            "rag_efficiency_score": rag_score,
        },

        "agent_identity": {
            "name": shadow_dna.agent_suggested_name
                    or f"Assistente Virtual {client_name}",
            "personality_traits": shadow_dna.personality_traits,
            "forbidden_terms": shadow_dna.forbidden_terms,
        },

        "knowledge_base_mapping": {
            "confirmed_procedures": shadow_dna.local_procedures,
            "detected_insurances": shadow_dna.local_insurances,
            "unresolved_queries": shadow_dna.unresolved_queries,
            "payment_methods": shadow_dna.local_payment_conditions,
        },

        "conversational_flow": {
            "greeting_style": {
                "tone": shadow_dna.tone_classification.lower()
                        if shadow_dna.tone_classification in ("Formal", "Informal", "Neutro", "Misto")
                        else "neutro",
                "example": shadow_dna.greeting_example,
            },
            "closing_style": {
                "tone": shadow_dna.tone_classification.lower()
                        if shadow_dna.tone_classification in ("Formal", "Informal", "Neutro", "Misto")
                        else "neutro",
                "example": shadow_dna.closing_example,
            },
            "handoff_trigger": {
                "keywords": shadow_dna.handoff_keywords,
                "situations": shadow_dna.handoff_situations,
            },
        },

        "shadow_dna_profile": {
            "tone_classification": shadow_dna.tone_classification,
            "average_response_length_tokens": shadow_dna.average_response_length_tokens,
            "emoji_frequency": shadow_dna.emoji_frequency,
            "sentiment_score_distribution": shadow_dna.sentiment_score_distribution,
            "response_time_metrics": shadow_dna.response_time_metrics,
            "common_objections": outcome_summary.common_objections,
            "local_entities": {
                "neighborhoods": shadow_dna.local_neighborhoods,
                "procedures": shadow_dna.local_procedures,
                "payment_conditions": shadow_dna.local_payment_conditions,
            },
        },

        "outcome_summary": {
            "agendado": outcome_summary.agendado,
            "ghosting": outcome_summary.ghosting,
            "objecao_ativa": outcome_summary.objecao_ativa,
            "pendente": outcome_summary.pendente,
            "outro": outcome_summary.outro,
            "conversion_rate": outcome_summary.conversion_rate,
        },

        "financial_kpis": {
            "ticket_medio": financial_kpis.ticket_medio,
            "ticket_medio_source": financial_kpis.ticket_medio_source,
            "leads_lost": financial_kpis.leads_lost,
            "opportunity_loss_value": financial_kpis.opportunity_loss_value,
            "potential_recovery_value": financial_kpis.potential_recovery_value,
        },
    }

    return blueprint


def save_blueprint(
    blueprint: dict,
    output_dir: Path,
    client_slug: str,
    timestamp: str,
) -> Path:
    """Save blueprint as a JSON file and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"blueprint_{client_slug}_{timestamp}.json"
    path.write_text(
        json.dumps(blueprint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Blueprint saved: %s", path)
    return path
