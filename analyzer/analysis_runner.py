"""
analysis_runner.py
------------------
Background processing entry point for POST /analyze/{clinic_id}.

Phase 9: Full end-to-end pipeline that sequences every component built in
Phases 6-8. Called via FastAPI BackgroundTasks.add_task(run_analysis, job_id, clinic_id).

Pipeline steps:
  Step 1  (5%):  Mark job status="processing"
  Step 2  (10%): Resolve clinic name from sf_clinics
  Step 3  (15%): _ensure_lm_configured() — lazy DSPy init
  Step 4  (20%): ingest_from_evolution() → conversations
  Step 5  (25%): Guard — 0 conversations → fail job fast
  Step 6  (30%): compute_metrics() per conversation → metrics_list
  Step 7  (35-70%): analyze_conversation() per conversation → analyses
  Step 8  (75%): detect_outcome() per conversation → outcome_results
  Step 9  (80%): extract_shadow_dna() → shadow_dna
  Step 10 (85%): aggregate_outcomes() → outcome_summary
  Step 11 (87%): compute_financial_kpis() → financial_kpis
  Step 12 (90%): aggregate_metrics() → agg_metrics
  Step 13 (92%): build_blueprint() → blueprint_dict
  Step 14 (95%): INSERT to la_blueprints with clinic_id
  Step 15 (97%): infer_and_persist_resources() — non-blocking
  Step 16 (100%): Mark job status="done"

Resilience contract:
  - Steps 6-13 are individually wrapped in try/except. Individual analysis
    failures produce degraded (empty) results but do NOT abort the pipeline.
  - Step 15 (resource inference) failure is explicitly non-blocking.
  - Any unhandled exception from steps 1-14 marks the job as "error".
"""
import logging
from datetime import datetime

from db import get_db
from analyzer.evolution_ingestor import ingest_from_evolution
from analyzer.metrics import compute_metrics, aggregate_metrics, AggregatedMetrics
from analyzer.dspy_pipeline import analyze_conversation, configure_lm, SemanticAnalysis
from analyzer.outcome_detection import detect_outcome, aggregate_outcomes, OutcomeSummary
from analyzer.shadow_dna import extract_shadow_dna, extract_returning_patient_playbook, ShadowDNA
from analyzer.financial_kpis import compute_financial_kpis, FinancialKPIs
from analyzer.blueprint import build_blueprint
from analyzer.resources_inference import infer_and_persist_resources
from analyzer.playbook_inference import extract_clinic_playbook

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy DSPy initialization — called once per process before any DSPy step
# ---------------------------------------------------------------------------

_lm_initialized: bool = False


def _ensure_lm_configured() -> None:
    """
    Initialize DSPy language models once per process.

    Skips initialization if:
    - already initialized by this function (_lm_initialized flag), OR
    - dspy.settings.lm is already configured (e.g. by test fixtures or worker startup)

    This prevents overwriting an LM configured by conftest.py or run_worker().
    """
    global _lm_initialized
    if _lm_initialized:
        return
    import dspy as _dspy
    if _dspy.settings.lm is not None:
        # LM already configured externally (test fixture, worker startup, etc.)
        _lm_initialized = True
        return
    from config import get_settings
    s = get_settings()
    configure_lm(
        openai_api_key=s.llm_api_key,
        model=s.llm_model,
        base_url=s.openai_base_url,
        anthropic_api_key=s.anthropic_api_key,
        consolidator_model=s.llm_model_consolidator,
    )
    _lm_initialized = True


# ---------------------------------------------------------------------------
# DB helpers — pass db explicitly so tests can inject a mock
# ---------------------------------------------------------------------------

def _update_job(db, job_id: str, **kwargs) -> None:
    db.table("la_analysis_jobs").update({
        **kwargs,
        "updated_at": datetime.utcnow().isoformat(),
    }).eq("id", job_id).execute()


def _set_progress(db, job_id: str, progress: int, step: str) -> None:
    logger.info("[%s] %d%% — %s", job_id[:8], progress, step)
    _update_job(db, job_id, progress=progress, current_step=step)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_analysis(job_id: str, clinic_id: str) -> None:
    """
    Full end-to-end pipeline executed after POST /analyze/{clinic_id} returns.
    Called via FastAPI BackgroundTasks.add_task(run_analysis, job_id, clinic_id).

    Individual analysis steps (metrics, DSPy, shadow DNA, blueprint assembly)
    are wrapped in try/except to produce degraded-but-valid output on partial
    failures, rather than aborting the entire job.

    A failure in infer_and_persist_resources() does NOT abort the job —
    blueprint is saved and job marked done regardless.

    Any unhandled exception from the core pipeline marks job status='error'.
    """
    db = get_db()

    # Step 1: Mark job as processing
    _update_job(db, job_id, status="processing", progress=5, current_step="Iniciando analise...")

    try:
        # Step 2: Resolve clinic name from sf_clinics
        _set_progress(db, job_id, 10, "Buscando dados da clinica...")
        clinic_result = (
            db.table("sf_clinics")
            .select("id, name")
            .eq("id", clinic_id)
            .single()
            .execute()
        )
        clinic_name = clinic_result.data["name"]

        # Step 3: Lazy DSPy LM initialization (once per process)
        _set_progress(db, job_id, 15, "Inicializando modelos de linguagem...")
        try:
            _ensure_lm_configured()
        except Exception as e:
            logger.warning("[%s] LM init failed (continuing with degraded analysis): %s", job_id[:8], e)

        # Step 4: Ingest conversations from Evolution
        _set_progress(db, job_id, 20, "Importando conversas do Evolution...")
        conversations = ingest_from_evolution(clinic_id, clinic_name)

        # Step 5: Guard — zero conversations → fail fast with human-readable message
        if not conversations:
            _update_job(
                db,
                job_id,
                status="error",
                error_message="Nenhuma conversa encontrada para essa clinica nos ultimos 90 dias.",
                progress=20,
                current_step="Falha: sem conversas",
            )
            return

        # Step 6: Compute per-conversation metrics (resilient)
        _set_progress(db, job_id, 30, "Calculando metricas das conversas...")
        metrics_list = []
        for conv in conversations:
            try:
                metrics_list.append(compute_metrics(conv))
            except Exception as e:
                logger.warning("[%s] compute_metrics failed for conversation: %s", job_id[:8], e)

        # Step 7: DSPy semantic analysis per conversation (35–70%) (resilient)
        _set_progress(db, job_id, 35, "Analisando conversas com IA...")
        analyses = []
        for conv in conversations:
            try:
                analysis = analyze_conversation(conv.messages, clinic_name)
                analyses.append(analysis)
            except Exception as e:
                logger.warning("[%s] DSPy analysis failed for conversation: %s", job_id[:8], e)
                analyses.append(SemanticAnalysis(error=str(e)))
        _set_progress(db, job_id, 70, "Analise semantica concluida...")

        # Step 8: Outcome detection per conversation (resilient)
        _set_progress(db, job_id, 75, "Detectando desfechos das conversas...")
        outcome_results = []
        for conv in conversations:
            try:
                result = detect_outcome(conv.messages, clinic_name)
                outcome_results.append(result)
            except Exception as e:
                logger.warning("[%s] Outcome detection failed for conversation: %s", job_id[:8], e)

        # Step 9: Extract Shadow DNA — MUST precede infer_and_persist_resources (resilient)
        _set_progress(db, job_id, 80, "Extraindo perfil comportamental (Shadow DNA)...")
        try:
            shadow_dna = extract_shadow_dna(conversations, clinic_name, analyses)
        except Exception as e:
            logger.warning("[%s] extract_shadow_dna failed (using empty fallback): %s", job_id[:8], e)
            shadow_dna = ShadowDNA()

        # Step 9b: Infer returning-patient playbook (resilient, non-blocking)
        try:
            returning_patient_playbook = extract_returning_patient_playbook(conversations)
        except Exception as e:
            logger.warning("[%s] extract_returning_patient_playbook failed (skipping): %s", job_id[:8], e)
            returning_patient_playbook = None

        # Step 9c: Infer clinic playbook (forensic — resilient, non-blocking)
        try:
            clinic_playbook = extract_clinic_playbook(
                conversations,
                clinic_name,
                outcome_results=outcome_results,
            )
        except Exception as e:
            logger.warning("[%s] extract_clinic_playbook failed (skipping): %s", job_id[:8], e)
            clinic_playbook = None

        # Step 10: Aggregate outcomes (resilient)
        _set_progress(db, job_id, 85, "Agregando desfechos...")
        try:
            outcome_summary = aggregate_outcomes(outcome_results)
        except Exception as e:
            logger.warning("[%s] aggregate_outcomes failed (using empty fallback): %s", job_id[:8], e)
            outcome_summary = OutcomeSummary()

        # Step 11: Compute financial KPIs (resilient)
        _set_progress(db, job_id, 87, "Calculando KPIs financeiros...")
        try:
            financial_kpis = compute_financial_kpis(
                outcome_summary,
                shadow_dna=shadow_dna,
                clinic_name=clinic_name,
            )
        except Exception as e:
            logger.warning("[%s] compute_financial_kpis failed (using empty fallback): %s", job_id[:8], e)
            financial_kpis = FinancialKPIs()

        # Step 12: Aggregate metrics (resilient)
        _set_progress(db, job_id, 90, "Consolidando metricas...")
        try:
            agg_metrics = aggregate_metrics(metrics_list)
        except Exception as e:
            logger.warning("[%s] aggregate_metrics failed (using empty fallback): %s", job_id[:8], e)
            agg_metrics = AggregatedMetrics()

        # Step 13: Build blueprint dict (resilient)
        _set_progress(db, job_id, 92, "Montando blueprint...")
        try:
            blueprint_dict = build_blueprint(
                client_slug=clinic_id,
                client_name=clinic_name,
                shadow_dna=shadow_dna,
                outcome_summary=outcome_summary,
                financial_kpis=financial_kpis,
                agg_metrics=agg_metrics,
                analyses=analyses,
                generated_at=datetime.utcnow(),
                returning_patient_playbook=returning_patient_playbook,
                clinic_playbook=clinic_playbook,
            )
        except Exception as e:
            logger.warning("[%s] build_blueprint failed (using empty fallback): %s", job_id[:8], e)
            blueprint_dict = {
                "clinic_id": clinic_id,
                "error": str(e),
                "generated_at": datetime.utcnow().isoformat(),
            }

        # Step 14: INSERT blueprint to la_blueprints with clinic_id
        _set_progress(db, job_id, 95, "Salvando blueprint...")
        db.table("la_blueprints").insert({
            "job_id": job_id,
            "clinic_id": clinic_id,
            "blueprint": blueprint_dict,
        }).execute()

        # Step 15: Infer and persist resources — NON-BLOCKING
        _set_progress(db, job_id, 97, "Inferindo profissionais e servicos...")
        try:
            infer_and_persist_resources(
                conversations,
                clinic_name,
                clinic_id,
                job_id,
                shadow_dna,
                db,
            )
        except Exception as e:
            logger.warning(
                "[%s] Resources inference failed (non-blocking): %s",
                job_id[:8],
                e,
            )

        # Step 16: Mark job done
        _update_job(db, job_id, status="done", progress=100, current_step="Concluido")
        logger.info("[%s] Pipeline complete for clinic %s", job_id[:8], clinic_id)

    except Exception as exc:
        logger.error("[%s] Pipeline failed: %s", job_id[:8], exc)
        try:
            db.table("la_analysis_jobs").update({
                "status": "error",
                "error_message": str(exc)[:2000],
            }).eq("id", job_id).execute()
        except Exception:
            pass
