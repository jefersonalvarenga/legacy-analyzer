"""
dspy_pipeline.py
----------------
LLM-based semantic analysis using DSPy.

Defines three DSPy modules:
  1. SentimentAnalyzer  → float score (-1.0 to 1.0) + label + reasoning
  2. TopicExtractor     → list of topic strings + most_critical flag
  3. QualityScorer      → quality score (0–10) + flags + improvement tips
  4. ConversationSummarizer → 2-3 sentence Portuguese summary

All modules operate on a conversation-level text blob (not per-message)
to keep costs down while maintaining accuracy.

Usage:
    from analyzer.dspy_pipeline import analyze_conversation
    result = await analyze_conversation(conversation_text, clinic_name)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import dspy

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# DSPy Signatures
# ------------------------------------------------------------------

class SentimentSignature(dspy.Signature):
    """
    Analyse the overall sentiment of a patient in a WhatsApp conversation
    with a healthcare clinic. Focus on the patient's messages only.
    Respond in the same language as the conversation (Portuguese).
    """
    conversation: str = dspy.InputField(desc="Full WhatsApp conversation text")
    clinic_name: str = dspy.InputField(desc="Name of the clinic in the conversation")

    score: float = dspy.OutputField(
        desc="Sentiment score from -1.0 (very negative) to 1.0 (very positive)"
    )
    label: str = dspy.OutputField(
        desc="One of: muito_positivo, positivo, neutro, negativo, muito_negativo"
    )
    reasoning: str = dspy.OutputField(
        desc="One sentence explaining the score (in Portuguese)"
    )


class TopicSignature(dspy.Signature):
    """
    Extract the main topics discussed in this WhatsApp conversation between
    a patient and a healthcare clinic. Topics should be in Portuguese,
    lowercase, as short noun phrases.
    """
    conversation: str = dspy.InputField(desc="Full WhatsApp conversation text")
    clinic_name: str = dspy.InputField(desc="Name of the clinic in the conversation")

    topics: list = dspy.OutputField(
        desc=(
            "List of 1–5 topic strings in Portuguese, e.g.: "
            "[\"confirmação de consulta\", \"reagendamento\", \"dúvida de preço\"]"
        )
    )
    primary_topic: str = dspy.OutputField(
        desc="The single most important topic in this conversation"
    )


class QualitySignature(dspy.Signature):
    """
    Evaluate the quality of service provided by the clinic in this WhatsApp
    conversation. Focus on: clarity, completeness, professionalism, response
    to patient needs, and identification of unanswered questions.
    Score on a scale of 0 (worst) to 10 (best).
    """
    conversation: str = dspy.InputField(desc="Full WhatsApp conversation text")
    clinic_name: str = dspy.InputField(desc="Name of the clinic in the conversation")

    score: float = dspy.OutputField(
        desc="Quality score from 0.0 (terrible) to 10.0 (excellent)"
    )
    flags: list = dspy.OutputField(
        desc=(
            "List of issue flags found, choose from: "
            "sem_resposta, informacao_incorreta, tom_inadequado, "
            "demora_excessiva, pergunta_ignorada, reclamacao_nao_tratada, "
            "paciente_frustrado, reagendamento_problematico"
        )
    )
    tips: list = dspy.OutputField(
        desc=(
            "Lista de 0–3 sugestões de melhoria em português. "
            "Cada sugestão deve ser ESPECÍFICA e ACIONÁVEL com base NESTA conversa — "
            "mencione o contexto real (ex: pergunta ignorada, convênio não confirmado, "
            "valor não informado). "
            "PROIBIDO retornar frases genéricas como: 'Manter tom profissional', "
            "'Melhorar a comunicação', 'Ser mais claro', 'Responder mais rápido' "
            "sem contexto específico desta conversa. "
            "BONS EXEMPLOS: "
            "'Informar o valor da avaliação no primeiro contato para evitar objeção de preço no final.' "
            "'Confirmar cobertura do convênio antes de agendar para reduzir cancelamentos.' "
            "Se a conversa não tiver problemas reais, retorne lista vazia []."
        )
    )


class SummarySignature(dspy.Signature):
    """
    Write a concise 2–3 sentence summary of this WhatsApp conversation between
    a patient and a healthcare clinic. Write in Portuguese. Include: main
    subject, outcome, and any notable issues.
    """
    conversation: str = dspy.InputField(desc="Full WhatsApp conversation text")
    clinic_name: str = dspy.InputField(desc="Name of the clinic in the conversation")

    summary: str = dspy.OutputField(
        desc="2–3 sentence summary in Portuguese"
    )


# ------------------------------------------------------------------
# DSPy Modules
# ------------------------------------------------------------------

class SentimentAnalyzer(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(SentimentSignature)

    def forward(self, conversation: str, clinic_name: str):
        return self.predict(conversation=conversation, clinic_name=clinic_name)


class TopicExtractor(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(TopicSignature)

    def forward(self, conversation: str, clinic_name: str):
        return self.predict(conversation=conversation, clinic_name=clinic_name)


class QualityScorer(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(QualitySignature)

    def forward(self, conversation: str, clinic_name: str):
        return self.predict(conversation=conversation, clinic_name=clinic_name)


class ConversationSummarizer(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(SummarySignature)

    def forward(self, conversation: str, clinic_name: str):
        return self.predict(conversation=conversation, clinic_name=clinic_name)


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------

@dataclass
class SemanticAnalysis:
    sentiment_score: float = 0.0
    sentiment_label: str = "neutro"
    sentiment_reasoning: str = ""

    topics: list[str] = field(default_factory=list)
    primary_topic: str = ""

    quality_score: float = 5.0
    quality_flags: list[str] = field(default_factory=list)
    quality_tips: list[str] = field(default_factory=list)

    summary: str = ""

    # Composite health score (0–100)
    health_score: float = 50.0

    error: Optional[str] = None


def _compute_health_score(sentiment: float, quality: float) -> float:
    """
    Composite health score (0–100):
      50% quality score (normalised to 0–100)
      50% sentiment score (normalised to 0–100)
    """
    quality_norm = (quality / 10.0) * 100
    sentiment_norm = ((sentiment + 1.0) / 2.0) * 100
    return round((quality_norm * 0.6 + sentiment_norm * 0.4), 1)


def _truncate_conversation(text: str, max_chars: int = 8000) -> str:
    """Truncate conversation to avoid exceeding context limits."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[conversa truncada]...\n" + text[-half:]


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_list(value, default: list) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        # DSPy sometimes returns a string representation of a list
        try:
            import ast
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        # Try splitting by comma
        return [v.strip().strip('"\'') for v in value.split(",") if v.strip()]
    return default


# ------------------------------------------------------------------
# Module instances (initialized lazily after LLM is configured)
# ------------------------------------------------------------------
_sentiment_module: Optional[SentimentAnalyzer] = None
_topic_module: Optional[TopicExtractor] = None
_quality_module: Optional[QualityScorer] = None
_summary_module: Optional[ConversationSummarizer] = None


def configure_lm(openai_api_key: str, model: str = "gpt-4o-mini"):
    """
    Call once at startup to configure DSPy's language model.
    Also initializes all other DSPy modules (outcome, shadow DNA, financial).
    """
    global _sentiment_module, _topic_module, _quality_module, _summary_module

    lm = dspy.LM(
        model=f"openai/{model}",
        api_key=openai_api_key,
        temperature=0.1,
        max_tokens=2048,
    )
    dspy.configure(lm=lm)

    _sentiment_module = SentimentAnalyzer()
    _topic_module = TopicExtractor()
    _quality_module = QualityScorer()
    _summary_module = ConversationSummarizer()

    # Initialize new modules (import here to avoid circular imports)
    from analyzer.outcome_detection import init_outcome_module
    from analyzer.shadow_dna import init_shadow_module
    from analyzer.financial_kpis import init_financial_module

    init_outcome_module()
    init_shadow_module()
    init_financial_module()

    logger.info("DSPy configured with model: %s", model)


def _conversation_to_text(messages) -> str:
    """Convert a list of Message objects to a readable text block."""
    lines = []
    for msg in messages:
        ts = msg.sent_at.strftime("%d/%m/%Y %H:%M")
        lines.append(f"[{msg.sender_type.upper()}] {ts} — {msg.sender}: {msg.content}")
    return "\n".join(lines)


def analyze_conversation(messages, clinic_name: str) -> SemanticAnalysis:
    """
    Run the full DSPy pipeline on a single conversation.

    Args:
        messages:     list of Message objects (from parser.py)
        clinic_name:  display name of the clinic

    Returns:
        SemanticAnalysis dataclass
    """
    if not _sentiment_module:
        return SemanticAnalysis(error="DSPy not configured. Call configure_lm() first.")

    result = SemanticAnalysis()
    conv_text = _truncate_conversation(_conversation_to_text(messages))

    # 1. Sentiment
    try:
        sent = _sentiment_module(conversation=conv_text, clinic_name=clinic_name)
        result.sentiment_score = max(-1.0, min(1.0, _safe_float(sent.score, 0.0)))
        result.sentiment_label = str(sent.label).strip()
        result.sentiment_reasoning = str(sent.reasoning).strip()
    except Exception as e:
        logger.warning("Sentiment analysis failed: %s", e)
        result.sentiment_score = 0.0
        result.sentiment_label = "neutro"

    # 2. Topics
    try:
        topics = _topic_module(conversation=conv_text, clinic_name=clinic_name)
        result.topics = _safe_list(topics.topics, [])
        result.primary_topic = str(topics.primary_topic).strip()
    except Exception as e:
        logger.warning("Topic extraction failed: %s", e)
        result.topics = []

    # 3. Quality
    try:
        quality = _quality_module(conversation=conv_text, clinic_name=clinic_name)
        result.quality_score = max(0.0, min(10.0, _safe_float(quality.score, 5.0)))
        result.quality_flags = _safe_list(quality.flags, [])
        result.quality_tips = _safe_list(quality.tips, [])
    except Exception as e:
        logger.warning("Quality scoring failed: %s", e)
        result.quality_score = 5.0

    # 4. Summary
    try:
        summ = _summary_module(conversation=conv_text, clinic_name=clinic_name)
        result.summary = str(summ.summary).strip()
    except Exception as e:
        logger.warning("Summary generation failed: %s", e)
        result.summary = ""

    # 5. Composite health score
    result.health_score = _compute_health_score(
        result.sentiment_score, result.quality_score
    )

    return result
