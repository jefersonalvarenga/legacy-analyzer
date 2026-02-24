"""
training_export.py
------------------
Exports conversation data in formats ready for LLM fine-tuning and RAG.

Supported formats:
  1. openai_jsonl     — {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
  2. anthropic_jsonl  — {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
                        (same structure, Anthropic compatible)
  3. rag_chunks       — {"id": ..., "text": ..., "metadata": {...}}

Selection strategy for training data quality:
  - Only include exchanges where the clinic's response has quality_score >= threshold
  - Skip patient messages that are just one word (e.g. "ok", "sim")
  - Skip system messages
  - For RAG, chunk by individual exchanges (patient msg + clinic reply)
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from analyzer.parser import Conversation, Message
from analyzer.dspy_pipeline import SemanticAnalysis


# Minimum quality score for a conversation to be included in fine-tuning data
DEFAULT_QUALITY_THRESHOLD = 7.0

# Minimum patient message length (chars) to include as a training example
MIN_PATIENT_MSG_CHARS = 10


@dataclass
class ExportStats:
    total_conversations: int = 0
    included_conversations: int = 0
    total_exchanges: int = 0
    exported_records: int = 0
    skipped_low_quality: int = 0
    skipped_short_msg: int = 0


def _clean(text: str) -> str:
    return text.strip().replace("\n", " ").replace("\r", "")


def _build_exchanges(
    messages: list[Message],
) -> list[tuple[Message, Message]]:
    """
    Build (patient_msg, clinic_reply) pairs from the message list.
    Only includes pairs where patient sends something substantive
    and clinic replies.
    """
    exchanges = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.sender_type == "patient" and len(msg.content) >= MIN_PATIENT_MSG_CHARS:
            # Look for the next clinic reply
            for j in range(i + 1, len(messages)):
                next_msg = messages[j]
                if next_msg.sender_type == "clinic":
                    exchanges.append((msg, next_msg))
                    i = j
                    break
                if next_msg.sender_type == "patient":
                    break  # clinic didn't reply before patient sent again
        i += 1
    return exchanges


def export_openai_jsonl(
    conversations: list[Conversation],
    analyses: list[SemanticAnalysis],
    client_name: str,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    system_prompt: Optional[str] = None,
) -> tuple[list[dict], ExportStats]:
    """
    Export fine-tuning data in OpenAI/Anthropic chat JSONL format.
    Each record = one (patient question, clinic answer) exchange.
    """
    if system_prompt is None:
        system_prompt = (
            f"Você é o assistente virtual de atendimento da {client_name}. "
            "Responda de forma clara, empática e profissional às mensagens dos pacientes."
        )

    stats = ExportStats(total_conversations=len(conversations))
    records = []

    for conv, analysis in zip(conversations, analyses):
        stats.total_conversations += 0  # already set above

        if analysis.quality_score < quality_threshold:
            stats.skipped_low_quality += 1
            continue

        stats.included_conversations += 1
        exchanges = _build_exchanges(conv.messages)
        stats.total_exchanges += len(exchanges)

        for patient_msg, clinic_msg in exchanges:
            patient_text = _clean(patient_msg.content)
            clinic_text = _clean(clinic_msg.content)

            if len(patient_text) < MIN_PATIENT_MSG_CHARS:
                stats.skipped_short_msg += 1
                continue

            record = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": patient_text},
                    {"role": "assistant", "content": clinic_text},
                ]
            }
            records.append(record)
            stats.exported_records += 1

    return records, stats


def export_rag_chunks(
    conversations: list[Conversation],
    analyses: list[SemanticAnalysis],
    client_name: str,
    client_slug: str,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> tuple[list[dict], ExportStats]:
    """
    Export RAG chunks: each chunk = one exchange with metadata.
    Suitable for vector store ingestion (e.g. pgvector, Pinecone, etc.)
    """
    stats = ExportStats(total_conversations=len(conversations))
    chunks = []

    for conv, analysis in zip(conversations, analyses):
        if analysis.quality_score < quality_threshold:
            stats.skipped_low_quality += 1
            continue

        stats.included_conversations += 1
        exchanges = _build_exchanges(conv.messages)
        stats.total_exchanges += len(exchanges)

        for patient_msg, clinic_msg in exchanges:
            patient_text = _clean(patient_msg.content)
            clinic_text = _clean(clinic_msg.content)

            if len(patient_text) < MIN_PATIENT_MSG_CHARS:
                stats.skipped_short_msg += 1
                continue

            chunk_text = f"Paciente: {patient_text}\n{client_name}: {clinic_text}"
            chunk = {
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "metadata": {
                    "client_slug": client_slug,
                    "client_name": client_name,
                    "phone": conv.phone,
                    "sent_at": patient_msg.sent_at.isoformat(),
                    "topics": analysis.topics,
                    "primary_topic": analysis.primary_topic,
                    "quality_score": analysis.quality_score,
                    "source": conv.source_filename,
                },
            }
            chunks.append(chunk)
            stats.exported_records += 1

    return chunks, stats


def records_to_jsonl(records: list[dict]) -> str:
    """Convert a list of dicts to a JSONL string (one JSON object per line)."""
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
