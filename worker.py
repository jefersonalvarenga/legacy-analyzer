"""
worker.py
---------
Background worker that polls Supabase for queued analysis jobs
and processes them end-to-end:

  1. Fetch queued job from la_analysis_jobs
  2. Parse Archive.zip → list[Conversation]
  3. Compute metrics per conversation (pure Python)
  4. Run DSPy pipeline per conversation (LLM calls)
  5. Generate embeddings per message + per conversation
  6. Persist everything to Supabase
  7. Build HTML report
  8. Generate training exports (openai_jsonl + rag_chunks)
  9. Mark job as done

Run with:
  python worker.py

Or start alongside the API:
  python worker.py &
  python main.py
"""

import asyncio
import json
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

from config import get_settings
from db import get_db
from analyzer.parser import parse_archive, Conversation
from analyzer.metrics import compute_metrics, aggregate_metrics, ConversationMetrics
from analyzer.dspy_pipeline import analyze_conversation, configure_lm, SemanticAnalysis
from analyzer.knowledge_consolidator import consolidate_knowledge, save_knowledge_to_supabase
from analyzer.embeddings import EmbeddingClient
from analyzer.report_builder import build_report
from analyzer.training_export import (
    export_openai_jsonl,
    export_rag_chunks,
    records_to_jsonl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

# LM references set once at startup and reused across all jobs
_fast_lm = None
_consolidation_lm = None


# ------------------------------------------------------------------
# Job status helpers
# ------------------------------------------------------------------

def _update_job(job_id: str, **kwargs):
    db = get_db()
    db.table("la_analysis_jobs").update({
        **kwargs,
        "updated_at": datetime.utcnow().isoformat(),
    }).eq("id", job_id).execute()


def _set_progress(job_id: str, progress: int, step: str):
    logger.info("[%s] %d%% — %s", job_id[:8], progress, step)
    _update_job(job_id, progress=progress, current_step=step)


def _fail_job(job_id: str, error: str):
    logger.error("[%s] Job failed: %s", job_id[:8], error)
    _update_job(job_id, status="error", error_message=error[:2000])


# ------------------------------------------------------------------
# Main processing pipeline
# ------------------------------------------------------------------

async def process_job(job: dict):
    job_id = job["id"]
    client_id = job["client_id"]
    file_url = job["file_url"]

    db = get_db()

    # Fetch client config
    client_result = (
        db.table("la_clients")
        .select("*")
        .eq("id", client_id)
        .single()
        .execute()
    )
    if not client_result.data:
        _fail_job(job_id, "Client not found")
        return

    client = client_result.data
    clinic_name = client.get("sender_name") or client["name"]
    client_slug = client["slug"]

    _update_job(job_id, status="processing")
    _set_progress(job_id, 2, "Iniciando processamento")

    # ------------------------------------------------------------------
    # 1. Parse zip
    # ------------------------------------------------------------------
    _set_progress(job_id, 5, "Extraindo e lendo conversas do arquivo zip")

    archive_path = Path(file_url)
    if not archive_path.exists():
        _fail_job(job_id, f"Archive file not found: {file_url}")
        return

    conversations: list[Conversation] = []
    total_zips = [0]

    def on_parse_progress(current: int, total: int, filename: str):
        total_zips[0] = total
        pct = 5 + int((current / total) * 20)  # 5% → 25%
        _set_progress(job_id, pct, f"Lendo conversa {current}/{total}: {filename}")

    try:
        conversations = parse_archive(
            archive_path, clinic_name, on_progress=on_parse_progress
        )
    except Exception as e:
        _fail_job(job_id, f"Parse error: {e}")
        return

    total_conversations = len(conversations)
    if total_conversations == 0:
        _fail_job(job_id, "No valid conversations found in the archive")
        return

    _update_job(
        job_id,
        total_conversations=total_conversations,
    )
    _set_progress(job_id, 25, f"{total_conversations} conversas encontradas. Salvando no banco...")

    # ------------------------------------------------------------------
    # 2. Persist conversations + messages to Supabase
    # ------------------------------------------------------------------
    conversation_db_ids: list[str] = []
    for conv in conversations:
        conv_result = db.table("la_conversations").insert({
            "job_id": job_id,
            "client_id": client_id,
            "phone": conv.phone,
            "message_count": conv.message_count,
            "clinic_message_count": len(conv.clinic_messages),
            "patient_message_count": len(conv.patient_messages),
            "date_start": conv.date_start.isoformat() if conv.date_start else None,
            "date_end": conv.date_end.isoformat() if conv.date_end else None,
            "duration_days": (
                (conv.date_end - conv.date_start).days
                if conv.date_start and conv.date_end else 0
            ),
        }).execute()

        if not conv_result.data:
            logger.warning("Failed to insert conversation %s", conv.source_filename)
            conversation_db_ids.append(None)
            continue

        conv_db_id = conv_result.data[0]["id"]
        conversation_db_ids.append(conv_db_id)

        # Batch insert messages (no embeddings yet)
        msg_rows = [
            {
                "conversation_id": conv_db_id,
                "client_id": client_id,
                "sent_at": msg.sent_at.isoformat(),
                "sender": msg.sender,
                "sender_type": msg.sender_type,
                "content": msg.content,
            }
            for msg in conv.messages
        ]
        # Insert in batches of 500
        for i in range(0, len(msg_rows), 500):
            db.table("la_messages").insert(msg_rows[i: i + 500]).execute()

    _set_progress(job_id, 35, "Conversas salvas. Calculando métricas...")

    # ------------------------------------------------------------------
    # 3. Compute metrics (pure Python, fast)
    # ------------------------------------------------------------------
    metrics_list: list[ConversationMetrics] = [
        compute_metrics(conv) for conv in conversations
    ]

    _set_progress(job_id, 40, "Métricas calculadas. Iniciando análise semântica (LLM)...")

    # ------------------------------------------------------------------
    # 4. DSPy semantic analysis (LLM calls — slowest step)
    # ------------------------------------------------------------------
    analyses: list[SemanticAnalysis] = []

    for idx, (conv, conv_db_id) in enumerate(zip(conversations, conversation_db_ids)):
        pct = 40 + int((idx / total_conversations) * 35)  # 40% → 75%
        _set_progress(
            job_id,
            pct,
            f"Análise semântica {idx + 1}/{total_conversations}: {conv.phone[:7]}***",
        )
        _update_job(job_id, processed_conversations=idx + 1)

        analysis = analyze_conversation(conv.messages, clinic_name)
        analyses.append(analysis)

        # Persist analysis
        if conv_db_id:
            db.table("la_chat_analyses").insert({
                "conversation_id": conv_db_id,
                "client_id": client_id,
                "job_id": job_id,
                "avg_response_time_seconds": metrics_list[idx].avg_response_time_seconds,
                "first_response_time_seconds": metrics_list[idx].first_response_time_seconds,
                "max_response_time_seconds": metrics_list[idx].max_response_time_seconds,
                "confirmation_rate": metrics_list[idx].confirmation_rate,
                "reminders_needed": metrics_list[idx].reminders_needed,
                "sentiment_score": analysis.sentiment_score,
                "quality_score": analysis.quality_score,
                "health_score": analysis.health_score,
                "topics": json.dumps(analysis.topics),
                "flags": json.dumps(analysis.quality_flags),
                "summary": analysis.summary,
                "llm_model": settings.llm_model,
            }).execute()

    _set_progress(job_id, 73, "Análise semântica concluída. Consolidando base de conhecimento...")

    # ------------------------------------------------------------------
    # 4b. KnowledgeConsolidator — corpus-wide fact extraction
    # ------------------------------------------------------------------
    knowledge = consolidate_knowledge(
        client_id=client_id,
        clinic_name=clinic_name,
        db=db,
        fast_lm=_fast_lm,
        consolidation_lm=_consolidation_lm,
    )
    save_knowledge_to_supabase(db, job_id, client_id, knowledge)

    _set_progress(job_id, 75, "Base de conhecimento consolidada. Gerando embeddings...")

    # ------------------------------------------------------------------
    # 5. Embeddings
    # ------------------------------------------------------------------
    embed_client = EmbeddingClient(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )

    for idx, (conv, conv_db_id) in enumerate(zip(conversations, conversation_db_ids)):
        if not conv_db_id:
            continue

        pct = 75 + int((idx / total_conversations) * 10)  # 75% → 85%
        if idx % 10 == 0:
            _set_progress(
                job_id, pct,
                f"Embeddings {idx + 1}/{total_conversations}",
            )

        # Conversation-level embedding (one call per conversation)
        conv_embedding = await embed_client.embed_conversation(conv.messages)
        if conv_embedding:
            db.table("la_chat_analyses").update({
                "embedding": conv_embedding,
            }).eq("conversation_id", conv_db_id).execute()

        # Per-message embeddings (batch)
        msg_embeddings = await embed_client.embed_messages(conv.messages)
        msg_db_result = (
            db.table("la_messages")
            .select("id")
            .eq("conversation_id", conv_db_id)
            .order("sent_at")
            .execute()
        )
        if msg_db_result.data:
            for msg_row, emb in zip(msg_db_result.data, msg_embeddings):
                if emb:
                    db.table("la_messages").update({
                        "embedding": emb,
                    }).eq("id", msg_row["id"]).execute()

    _set_progress(job_id, 85, "Embeddings gerados. Construindo relatório...")

    # ------------------------------------------------------------------
    # 6. Build report
    # ------------------------------------------------------------------
    agg = aggregate_metrics(metrics_list)
    html_report = build_report(
        client_name=client["name"],
        client_slug=client_slug,
        agg=agg,
        metrics_list=metrics_list,
        analyses=analyses,
        generated_at=datetime.utcnow(),
    )

    db.table("la_analysis_reports").insert({
        "job_id": job_id,
        "client_id": client_id,
        "html_content": html_report,
        "kpis_summary": {
            "total_conversations": agg.total_conversations,
            "total_messages": agg.total_messages,
            "avg_response_time_seconds": agg.avg_response_time_seconds,
            "avg_confirmation_rate": agg.avg_confirmation_rate,
            "avg_quality_score": agg.avg_quality_score,
            "avg_sentiment_score": agg.avg_sentiment_score,
            "avg_health_score": agg.avg_health_score,
            "cancellation_rate": agg.cancellation_rate,
        },
    }).execute()

    # Also save HTML locally in dev
    if settings.app_env == "development":
        report_dir = Path(settings.reports_output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"report_{client_slug}_{job_id[:8]}.html"
        report_path.write_text(html_report, encoding="utf-8")
        logger.info("Report saved locally: %s", report_path)

    _set_progress(job_id, 92, "Relatório gerado. Exportando dados de treino...")

    # ------------------------------------------------------------------
    # 7. Training exports
    # ------------------------------------------------------------------
    export_dir = Path(settings.reports_output_dir) / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    # OpenAI JSONL
    oa_records, oa_stats = export_openai_jsonl(
        conversations, analyses, client["name"]
    )
    oa_jsonl = records_to_jsonl(oa_records)
    oa_path = export_dir / f"finetune_openai_{client_slug}_{job_id[:8]}.jsonl"
    oa_path.write_text(oa_jsonl, encoding="utf-8")

    db.table("la_training_exports").insert({
        "job_id": job_id,
        "client_id": client_id,
        "format": "openai_jsonl",
        "file_url": str(oa_path),
        "record_count": oa_stats.exported_records,
    }).execute()

    # RAG chunks
    rag_records, rag_stats = export_rag_chunks(
        conversations, analyses, client["name"], client_slug
    )
    rag_jsonl = records_to_jsonl(rag_records)
    rag_path = export_dir / f"rag_chunks_{client_slug}_{job_id[:8]}.jsonl"
    rag_path.write_text(rag_jsonl, encoding="utf-8")

    db.table("la_training_exports").insert({
        "job_id": job_id,
        "client_id": client_id,
        "format": "rag_chunks",
        "file_url": str(rag_path),
        "record_count": rag_stats.exported_records,
    }).execute()

    logger.info(
        "[%s] Exports: %d fine-tune records, %d RAG chunks",
        job_id[:8], oa_stats.exported_records, rag_stats.exported_records,
    )

    # ------------------------------------------------------------------
    # 8. Mark done
    # ------------------------------------------------------------------
    _update_job(
        job_id,
        status="done",
        progress=100,
        current_step="Concluído",
        processed_conversations=total_conversations,
    )
    logger.info("[%s] Job completed successfully", job_id[:8])


# ------------------------------------------------------------------
# Polling loop
# ------------------------------------------------------------------

async def run_worker():
    settings_local = get_settings()

    # Configure DSPy/LLM once — captures fast_lm and consolidation_lm for KnowledgeConsolidator
    global _fast_lm, _consolidation_lm
    _fast_lm, _consolidation_lm = configure_lm(
        openai_api_key=settings_local.llm_api_key,   # GLM_API_KEY se definido, senão OPENAI_API_KEY
        model=settings_local.llm_model,
        base_url=settings_local.openai_base_url,
        anthropic_api_key=settings_local.anthropic_api_key,
        consolidator_model=settings_local.llm_model_consolidator,
    )

    logger.info(
        "Worker started. Polling every %ds...", settings_local.worker_poll_interval
    )

    while True:
        try:
            db = get_db()
            result = (
                db.table("la_analysis_jobs")
                .select("*")
                .eq("status", "queued")
                .order("created_at")
                .limit(1)
                .execute()
            )

            if result.data:
                job = result.data[0]
                logger.info("Picked up job %s", job["id"])
                try:
                    await process_job(job)
                except Exception as e:
                    tb = traceback.format_exc()
                    _fail_job(job["id"], f"{e}\n{tb}")
            else:
                logger.debug("No queued jobs. Sleeping...")

        except Exception as e:
            logger.error("Worker loop error: %s", e)

        await asyncio.sleep(settings_local.worker_poll_interval)


if __name__ == "__main__":
    asyncio.run(run_worker())
