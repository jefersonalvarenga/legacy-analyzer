"""
run_local.py
------------
Convenience script to process a local Archive.zip without running the
full FastAPI + worker stack.

Usage:
    python run_local.py \
        --archive /path/to/Archive.zip \
        --client-slug sgen \
        --output ./legacy-analyzer

This script:
  1. Parses the archive
  2. Computes metrics
  3. Runs DSPy semantic analysis
  4. Generates the HTML report (saved locally)
  5. Saves training exports (JSONL)
  6. Optionally persists to Supabase if SUPABASE_URL is set

Perfect for a quick local demo without Docker or a running server.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from config import get_settings
from analyzer.parser import parse_archive
from analyzer.metrics import compute_metrics, aggregate_metrics
from analyzer.dspy_pipeline import analyze_conversation, configure_lm
from analyzer.outcome_detection import detect_outcome, aggregate_outcomes
from analyzer.shadow_dna import extract_shadow_dna
from analyzer.financial_kpis import compute_financial_kpis
from analyzer.blueprint import build_blueprint, save_blueprint
from analyzer.embeddings import EmbeddingClient
from analyzer.report_builder import build_report
from analyzer.training_export import (
    export_openai_jsonl,
    export_rag_chunks,
    records_to_jsonl,
)

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path, timestamp: str) -> Path:
    """Configure logging to both console and a run log file."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{timestamp}.log"

    fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console handler — INFO only, clean format
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s — %(message)s"))
    root.addHandler(console)

    # File handler — DEBUG level, full detail
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(file_handler)

    # Silence noisy third-party loggers in console (still go to file)
    for noisy in ("httpx", "httpcore", "openai", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return log_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="EasyScale Legacy Analyzer — local runner"
    )
    parser.add_argument(
        "--archive",
        required=True,
        help="Path to the outer Archive.zip containing chat zips",
    )
    parser.add_argument(
        "--client-slug",
        default="sgen",
        help="Client slug (must exist in la_clients if using Supabase)",
    )
    parser.add_argument(
        "--client-name",
        default="Sorriso Da Gente",
        help="Display name of the client/clinic",
    )
    parser.add_argument(
        "--sender-name",
        default=None,
        help="WhatsApp display name used by the clinic (defaults to --client-name)",
    )
    parser.add_argument(
        "--output",
        default="./legacy-analyzer",
        help="Output directory for the report and exports",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding generation (faster, no vector store)",
    )
    parser.add_argument(
        "--no-supabase",
        action="store_true",
        help="Skip persisting to Supabase (local files only)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of conversations to process (0 = all). Useful for testing.",
    )
    parser.add_argument(
        "--ticket-medio",
        type=float,
        default=None,
        help="Average ticket value in BRL (user input). If not provided, estimated by LLM.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    settings = get_settings()

    archive_path = Path(args.archive)
    if not archive_path.exists():
        print(f"ERROR: Archive not found: {archive_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "exports").mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = setup_logging(output_dir, timestamp)
    logger.info("Run started — log: %s", log_path)

    sender_name = args.sender_name or args.client_name

    # ------------------------------------------------------------------
    # 1. Configure LLM
    # ------------------------------------------------------------------
    logger.info("Configuring DSPy with model: %s", settings.llm_model)
    configure_lm(
        openai_api_key=settings.openai_api_key,
        model=settings.llm_model,
    )

    # ------------------------------------------------------------------
    # 2. Parse archive
    # ------------------------------------------------------------------
    logger.info("Parsing archive: %s", archive_path)
    with tqdm(desc="Lendo conversas", unit="zip", leave=False) as pbar:
        def on_progress(cur, total, name):
            pbar.total = total
            pbar.update(1)
            pbar.set_postfix_str(name[:40])
            logger.debug("  [%d/%d] %s", cur, total, name)

        conversations = parse_archive(
            archive_path,
            clinic_sender_name=sender_name,
            on_progress=on_progress,
        )

    logger.info("Parsed %d conversations", len(conversations))

    if not conversations:
        logger.error("No conversations found. Check the archive and sender name.")
        sys.exit(1)

    if args.limit > 0:
        conversations = conversations[: args.limit]
        logger.info("Limited to %d conversations", len(conversations))

    # ------------------------------------------------------------------
    # 3. Compute metrics
    # ------------------------------------------------------------------
    logger.info("Computing metrics...")
    metrics_list = [compute_metrics(conv) for conv in conversations]

    # ------------------------------------------------------------------
    # 4. Semantic analysis
    # ------------------------------------------------------------------
    analyses = []
    errors_semantic = []
    with tqdm(conversations, desc="Análise semântica (LLM)", unit="conv") as pbar:
        for conv in pbar:
            pbar.set_postfix_str(f"{conv.phone[:7]}***")
            analysis = analyze_conversation(conv.messages, args.client_name)
            if analysis.error:
                errors_semantic.append((conv.phone, analysis.error))
                logger.warning("Semantic error [%s]: %s", conv.phone[:7], analysis.error)
            analyses.append(analysis)

    if errors_semantic:
        logger.warning("%d semantic analysis errors (see log for details)", len(errors_semantic))

    # ------------------------------------------------------------------
    # 5. Embeddings (optional)
    # ------------------------------------------------------------------
    conv_embeddings = [None] * len(conversations)
    errors_embed = []
    if not args.no_embeddings:
        embed_client = EmbeddingClient(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )
        with tqdm(conversations, desc="Embeddings", unit="conv") as pbar:
            for idx, conv in enumerate(pbar):
                pbar.set_postfix_str(f"{conv.phone[:7]}***")
                emb = await embed_client.embed_conversation(conv.messages)
                if emb is None:
                    errors_embed.append(conv.phone)
                    logger.warning("Embedding failed for %s", conv.phone[:7])
                conv_embeddings[idx] = emb

        if errors_embed:
            logger.warning("%d embedding failures (conversations still included in report)", len(errors_embed))

    # ------------------------------------------------------------------
    # 6. Outcome detection
    # ------------------------------------------------------------------
    outcome_results = []
    with tqdm(conversations, desc="Detecção de desfechos", unit="conv") as pbar:
        for conv in pbar:
            pbar.set_postfix_str(f"{conv.phone[:7]}***")
            outcome_results.append(detect_outcome(conv.messages, args.client_name))

    outcome_summary = aggregate_outcomes(outcome_results)
    logger.info(
        "Outcomes: agendado=%d, ghosting=%d, objeção=%d, pendente=%d, outro=%d | conversão=%.0f%%",
        outcome_summary.agendado, outcome_summary.ghosting,
        outcome_summary.objecao_ativa, outcome_summary.pendente, outcome_summary.outro,
        outcome_summary.conversion_rate * 100,
    )

    # ------------------------------------------------------------------
    # 6b. Shadow DNA extraction
    # ------------------------------------------------------------------
    logger.info("Extracting Shadow DNA...")
    shadow_dna = extract_shadow_dna(conversations, args.client_name, analyses)

    # ------------------------------------------------------------------
    # 6c. Financial KPIs
    # ------------------------------------------------------------------
    logger.info("Computing financial KPIs...")
    financial_kpis = compute_financial_kpis(
        outcome_summary=outcome_summary,
        shadow_dna=shadow_dna,
        clinic_name=args.client_name,
        ticket_medio_override=args.ticket_medio,
    )

    # ------------------------------------------------------------------
    # 6d. Blueprint JSON
    # ------------------------------------------------------------------
    logger.info("Building implementation blueprint...")
    agg = aggregate_metrics(metrics_list)
    blueprint = build_blueprint(
        client_slug=args.client_slug,
        client_name=args.client_name,
        shadow_dna=shadow_dna,
        outcome_summary=outcome_summary,
        financial_kpis=financial_kpis,
        agg_metrics=agg,
        analyses=analyses,
        generated_at=datetime.now(),
    )
    blueprint_path = save_blueprint(blueprint, output_dir / "exports", args.client_slug, timestamp)

    # ------------------------------------------------------------------
    # 7. Build HTML report
    # ------------------------------------------------------------------
    logger.info("Building report...")
    html_report = build_report(
        client_name=args.client_name,
        client_slug=args.client_slug,
        agg=agg,
        metrics_list=metrics_list,
        analyses=analyses,
        outcome_summary=outcome_summary,
        shadow_dna=shadow_dna,
        financial_kpis=financial_kpis,
        generated_at=datetime.now(),
    )

    report_path = output_dir / f"report_{args.client_slug}_{timestamp}.html"
    report_path.write_text(html_report, encoding="utf-8")
    logger.info("Report saved: %s", report_path)

    # ------------------------------------------------------------------
    # 8. Training exports
    # ------------------------------------------------------------------
    logger.info("Exporting training data...")

    oa_records, oa_stats = export_openai_jsonl(
        conversations, analyses, args.client_name
    )
    oa_path = output_dir / "exports" / f"finetune_openai_{args.client_slug}_{timestamp}.jsonl"
    oa_path.write_text(records_to_jsonl(oa_records), encoding="utf-8")
    logger.info(
        "OpenAI fine-tune export: %d records → %s",
        oa_stats.exported_records, oa_path,
    )

    rag_records, rag_stats = export_rag_chunks(
        conversations, analyses, args.client_name, args.client_slug
    )
    rag_path = output_dir / "exports" / f"rag_chunks_{args.client_slug}_{timestamp}.jsonl"
    rag_path.write_text(records_to_jsonl(rag_records), encoding="utf-8")
    logger.info(
        "RAG chunks export: %d records → %s",
        rag_stats.exported_records, rag_path,
    )

    # ------------------------------------------------------------------
    # 8. Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  EasyScale Legacy Analyzer — {args.client_name}")
    print("=" * 60)
    print(f"  Conversas analisadas:     {agg.total_conversations}")
    print(f"  Total de mensagens:       {agg.total_messages:,}")
    print(f"  Tempo médio de resposta:  {_fmt(agg.avg_response_time_seconds)}")
    print(f"  Qualidade média:          {(agg.avg_quality_score or 0):.1f}/10")
    print(f"  Score de saúde:           {(agg.avg_health_score or 0):.0f}/100")
    print(f"  Tom detectado:            {shadow_dna.tone_classification}")
    print("-" * 60)
    print(f"  Desfechos:")
    print(f"    Agendado:               {outcome_summary.agendado}")
    print(f"    Ghosting:               {outcome_summary.ghosting}")
    print(f"    Objeção ativa:          {outcome_summary.objecao_ativa}")
    print(f"    Pendente:               {outcome_summary.pendente}")
    print(f"    Taxa de conversão:      {outcome_summary.conversion_rate * 100:.0f}%")
    print("-" * 60)
    ticket_flag = "" if financial_kpis.ticket_medio_source == "user_input" else " (estimado)"
    print(f"  Ticket médio:             R$ {financial_kpis.ticket_medio:,.0f}{ticket_flag}")
    print(f"  Oportunidade perdida:     R$ {financial_kpis.opportunity_loss_value:,.0f}")
    print(f"  Recuperável com IA (30%): R$ {financial_kpis.potential_recovery_value:,.0f}")
    print("-" * 60)
    print(f"  Relatório HTML:           {report_path}")
    print(f"  Blueprint JSON:           {blueprint_path}")
    print(f"  Fine-tune (OpenAI):       {oa_path} ({oa_stats.exported_records} registros)")
    print(f"  RAG chunks:               {rag_path} ({rag_stats.exported_records} chunks)")
    print(f"  Log de execução:          {log_path}")
    if errors_semantic:
        print(f"  ⚠ Erros semânticos:       {len(errors_semantic)} conversas (ver log)")
    if not args.no_embeddings and errors_embed:
        print(f"  ⚠ Erros de embedding:     {len(errors_embed)} conversas (ver log)")
    print("=" * 60 + "\n")
    logger.info("Run finished — %d errors semantic, %d errors embedding",
                len(errors_semantic), len(errors_embed) if not args.no_embeddings else 0)

    # ------------------------------------------------------------------
    # 9. Optional Supabase persist
    # ------------------------------------------------------------------
    if not args.no_supabase:
        logger.info("Persisting to Supabase...")
        try:
            from db import get_db
            await _persist_to_supabase(
                args, conversations, metrics_list, analyses,
                conv_embeddings, agg, html_report,
                str(oa_path), oa_stats.exported_records,
                str(rag_path), rag_stats.exported_records,
            )
            logger.info("Supabase persistence complete")
        except Exception as e:
            logger.warning("Supabase persistence failed (continuing): %s", e)


def _fmt(seconds):
    if seconds is None:
        return "—"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}min {s}s" if m > 0 else f"{s}s"


async def _persist_to_supabase(
    args, conversations, metrics_list, analyses,
    conv_embeddings, agg, html_report,
    oa_path, oa_count, rag_path, rag_count,
):
    import json
    from db import get_db
    from datetime import datetime

    db = get_db()

    # Resolve client
    client_result = (
        db.table("la_clients")
        .select("id")
        .eq("slug", args.client_slug)
        .single()
        .execute()
    )
    if not client_result.data:
        logger.warning(
            "Client '%s' not found in Supabase. Run the SQL seed first.", args.client_slug
        )
        return

    client_id = client_result.data["id"]

    # Create job record
    job_result = db.table("la_analysis_jobs").insert({
        "client_id": client_id,
        "status": "done",
        "progress": 100,
        "current_step": "Concluído (local run)",
        "file_url": args.archive,
        "original_filename": Path(args.archive).name,
        "total_conversations": agg.total_conversations,
        "processed_conversations": agg.total_conversations,
    }).execute()

    if not job_result.data:
        logger.warning("Failed to insert job record")
        return

    job_id = job_result.data[0]["id"]

    # Conversations + analyses
    for conv, metrics, analysis, emb in zip(
        conversations, metrics_list, analyses, conv_embeddings
    ):
        conv_result = db.table("la_conversations").insert({
            "job_id": job_id,
            "client_id": client_id,
            "phone": conv.phone,
            "message_count": conv.message_count,
            "clinic_message_count": len(conv.clinic_messages),
            "patient_message_count": len(conv.patient_messages),
            "date_start": conv.date_start.isoformat() if conv.date_start else None,
            "date_end": conv.date_end.isoformat() if conv.date_end else None,
            "duration_days": metrics.duration_days,
        }).execute()

        if not conv_result.data:
            continue

        conv_db_id = conv_result.data[0]["id"]

        # Messages (batch)
        msg_rows = [{
            "conversation_id": conv_db_id,
            "client_id": client_id,
            "sent_at": msg.sent_at.isoformat(),
            "sender": msg.sender,
            "sender_type": msg.sender_type,
            "content": msg.content,
        } for msg in conv.messages]

        for i in range(0, len(msg_rows), 500):
            db.table("la_messages").insert(msg_rows[i: i + 500]).execute()

        # Analysis
        db.table("la_chat_analyses").insert({
            "conversation_id": conv_db_id,
            "client_id": client_id,
            "job_id": job_id,
            "avg_response_time_seconds": metrics.avg_response_time_seconds,
            "first_response_time_seconds": metrics.first_response_time_seconds,
            "max_response_time_seconds": metrics.max_response_time_seconds,
            "confirmation_rate": metrics.confirmation_rate,
            "reminders_needed": metrics.reminders_needed,
            "sentiment_score": analysis.sentiment_score,
            "quality_score": analysis.quality_score,
            "health_score": analysis.health_score,
            "topics": json.dumps(analysis.topics),
            "flags": json.dumps(analysis.quality_flags),
            "summary": analysis.summary,
            "embedding": emb,
            "llm_model": get_settings().llm_model,
        }).execute()

    # Report
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
        },
    }).execute()

    # Exports
    db.table("la_training_exports").insert([
        {
            "job_id": job_id,
            "client_id": client_id,
            "format": "openai_jsonl",
            "file_url": oa_path,
            "record_count": oa_count,
        },
        {
            "job_id": job_id,
            "client_id": client_id,
            "format": "rag_chunks",
            "file_url": rag_path,
            "record_count": rag_count,
        },
    ]).execute()

    logger.info("Persisted to Supabase. Job ID: %s", job_id)


if __name__ == "__main__":
    asyncio.run(main())
