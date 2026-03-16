"""
analysis_runner.py
------------------
Background processing entry point for POST /analyze/{clinic_id}.

Phase 7: stub — updates job status to 'processing', logs intent.
Phase 9: replace stub block with full pipeline call:
    from analyzer.evolution_ingestor import ingest_from_evolution
    conversations = ingest_from_evolution(clinic_id, clinic_name)
    # ... metrics, DSPy, blueprint ...
"""
import logging
from db import get_db

logger = logging.getLogger(__name__)


def run_analysis(job_id: str, clinic_id: str) -> None:
    """
    Background task executed after POST /analyze/{clinic_id} returns.
    Called via FastAPI BackgroundTasks.add_task(run_analysis, job_id, clinic_id).

    STUB: marks job as processing, logs, then marks complete as placeholder.
    Phase 9 replaces the stub block with the real pipeline.
    """
    db = get_db()
    try:
        db.table("la_analysis_jobs").update({
            "status": "processing",
            "progress": 1,
            "current_step": "Iniciando analise...",
        }).eq("id", job_id).execute()

        # --- Phase 9 will replace this stub block ---
        # from analyzer.evolution_ingestor import ingest_from_evolution
        # conversations = ingest_from_evolution(clinic_id, clinic_sender_name)
        # run_full_pipeline(job_id, clinic_id, conversations)
        # --------------------------------------------

        logger.info("[%s] Background stub complete for clinic %s", job_id[:8], clinic_id)

    except Exception as exc:
        logger.error("[%s] Background task failed: %s", job_id[:8], exc)
        try:
            db.table("la_analysis_jobs").update({
                "status": "error",
                "error_message": str(exc)[:2000],
            }).eq("id", job_id).execute()
        except Exception:
            pass
