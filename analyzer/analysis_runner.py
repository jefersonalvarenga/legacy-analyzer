"""
analysis_runner.py
------------------
Pipeline V2 do Legacy Analyzer — 3 fases:
  1. Ingest   — lê todas as Messages da clínica via evolution_ingestor
  2. Extract  — 1 call DSPy (Gemini 2.5 Flash default) → Blueprint completo
  3. Persist  — grava em la_blueprints, popula la_resources/la_services derivados,
                marca job done

Step orchestration:
  - sf_clinics.onboarding_step → 'learning' no início do run
  - sf_clinics.onboarding_step → 'review' ao terminar com sucesso

Tudo o que era pipeline antigo (sentiment/quality/outcomes/KPIs/shadow DNA)
foi removido — git history em main protege.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from db import get_db
from analyzer.evolution_ingestor import ingest_from_evolution
from analyzer.dspy_pipeline import configure_lm
from analyzer.blueprint_v2 import (
    Blueprint,
    extract_blueprint,
    to_storage_dict,
)
from analyzer.chunker import extract_blueprint_chunked
from analyzer.sf_sync import sync_blueprint_to_sf

logger = logging.getLogger(__name__)


_lm_initialized = False


def _ensure_lm_configured() -> tuple[str, str]:
    """Idempotent — configures DSPy once per process. Returns (provider, model)."""
    global _lm_initialized
    import dspy as _dspy
    if _lm_initialized and _dspy.settings.lm is not None:
        # already configured; recover provider/model from settings if possible
        lm = _dspy.settings.lm
        return ("unknown", getattr(lm, "model", "unknown"))
    provider, model = configure_lm()
    _lm_initialized = True
    return provider, model


def _update_job(db, job_id: str, **kwargs) -> None:
    db.table("la_analysis_jobs").update({
        **kwargs,
        "updated_at": datetime.utcnow().isoformat(),
    }).eq("id", job_id).execute()


def _set_progress(db, job_id: str, progress: int, step: str) -> None:
    logger.info("[%s] %d%% — %s", job_id[:8], progress, step)
    _update_job(db, job_id, progress=progress, current_step=step)


def _persist_blueprint(
    db,
    *,
    job_id: str,
    clinic_id: str,
    blueprint: Blueprint,  # noqa: ARG001 — kept for future derived persistence
    storage_dict: dict,
) -> None:
    """Grava blueprint completo em la_blueprints.blueprint_json. Tabelas derivadas
    (la_resources / la_services) ficam pra ponte la_* → sf_* — fora do escopo do LA V2."""
    db.table("la_blueprints").insert({
        "job_id": job_id,
        "clinic_id": clinic_id,
        "blueprint_json": storage_dict,
        "knowledge_base_mapping": {},  # legacy NOT NULL, no longer used
    }).execute()


def run_analysis(
    job_id: str,
    clinic_id: str,
    reference_conversation_ids: "Optional[list[str]]" = None,  # noqa: ARG001 — compat with main.py
) -> None:
    """
    Pipeline V2 — 3 fases. Chamado via FastAPI BackgroundTasks após POST /analyze/{clinic_id}.

    Erros não tratados marcam job status=error.
    """
    db = get_db()
    _update_job(db, job_id, status="processing", progress=5, current_step="Iniciando análise...")

    # Promote clinic step → 'learning' (HerModal screen 7 will keep showing
    # while the LA runs; reload mid-run resumes there).
    try:
        db.table("sf_clinics").update({"onboarding_step": "learning"}).eq("id", clinic_id).execute()
    except Exception as e:
        logger.warning("[%s] failed to set onboarding_step=learning: %s", job_id[:8], e)

    try:
        # Fase 1 — Ingest
        _set_progress(db, job_id, 10, "Buscando dados da clínica...")
        clinic_result = (
            db.table("sf_clinics")
            .select("id, name")
            .eq("id", clinic_id)
            .single()
            .execute()
        )
        clinic_name = clinic_result.data["name"] or clinic_id

        _set_progress(db, job_id, 15, "Inicializando modelo de linguagem...")
        provider, model = _ensure_lm_configured()

        _set_progress(db, job_id, 25, "Importando conversas...")
        conversations = ingest_from_evolution(clinic_id, clinic_name)
        if not conversations:
            _update_job(
                db, job_id,
                status="error",
                error_message="Nenhuma conversa encontrada para essa clínica nos últimos 90 dias.",
                progress=25,
                current_step="Falha: sem conversas",
            )
            return

        message_count = sum(c.message_count for c in conversations)
        logger.info(
            "[%s] Ingested %d conversations / %d messages",
            job_id[:8], len(conversations), message_count,
        )

        # Fase 2 — Extract DNA (1 ou N calls LLM, conforme volume)
        # Estimativa fixa: 30s por chunk. Grava eta_finished_at (absoluto) uma
        # única vez aqui — frontend faz a contagem regressiva derivada do clock.
        from datetime import timedelta
        from analyzer.chunker import DEFAULT_MAX_CONVS_PER_CHUNK
        chunks_total = max(1, (len(conversations) + DEFAULT_MAX_CONVS_PER_CHUNK - 1) // DEFAULT_MAX_CONVS_PER_CHUNK)
        SEC_PER_CHUNK = 30
        eta_finished_at = (datetime.utcnow() + timedelta(seconds=chunks_total * SEC_PER_CHUNK)).isoformat() + "Z"
        _update_job(
            db, job_id,
            progress=50,
            current_step="Analisando conversas com IA...",
            chunks_total=chunks_total,
            chunks_done=0,
            eta_finished_at=eta_finished_at,
        )

        def _on_chunk_progress(done: int, total: int, _eta: int) -> None:
            step = (
                f"Analisando conversas ({done}/{total})"
                if total > 1
                else "Analisando conversas com IA..."
            )
            _update_job(
                db, job_id,
                current_step=step,
                chunks_done=done,
                progress=50 + int(40 * (done / total)) if total else 50,
            )

        blueprint = extract_blueprint_chunked(
            conversations,
            clinic_name,
            on_progress=_on_chunk_progress,
        )

        # Fase 3 — Persist
        _set_progress(db, job_id, 90, "Salvando blueprint...")
        storage_dict = to_storage_dict(
            blueprint,
            clinic_id=clinic_id,
            clinic_name=clinic_name,
            conversation_count=len(conversations),
            message_count=message_count,
            llm_provider=provider,
            llm_model=model,
        )
        _persist_blueprint(
            db,
            job_id=job_id,
            clinic_id=clinic_id,
            blueprint=blueprint,
            storage_dict=storage_dict,
        )

        # Auto-migra blueprint → sf_* (perfil, profissionais, especialidades,
        # serviços, mapeamento, tom da assistente). Aprovação por domínio é
        # feita depois pelo usuário via UI (sf_clinics.onboarding_review).
        _set_progress(db, job_id, 95, "Atualizando dados da clínica...")
        try:
            sync_blueprint_to_sf(db, clinic_id, blueprint)
        except Exception as e:
            # Sync falhou. Mantém blueprint salvo + marca job error.
            logger.error("[%s] sf_sync failed: %s", job_id[:8], e, exc_info=True)
            raise

        # Promote clinic step → 'review' (LA finished; user can now approve
        # the 6 domains in the dashboard).
        try:
            db.table("sf_clinics").update({"onboarding_step": "review"}).eq("id", clinic_id).execute()
        except Exception as e:
            logger.warning("[%s] failed to set onboarding_step=review: %s", job_id[:8], e)

        _update_job(db, job_id, status="done", progress=100, current_step="Concluído")
        logger.info("[%s] Pipeline complete for clinic %s", job_id[:8], clinic_id)

    except Exception as exc:
        logger.error("[%s] Pipeline failed: %s", job_id[:8], exc, exc_info=True)
        try:
            db.table("la_analysis_jobs").update({
                "status": "error",
                "error_message": str(exc)[:2000],
            }).eq("id", job_id).execute()
        except Exception:
            pass
