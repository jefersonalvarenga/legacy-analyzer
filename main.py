"""
main.py
-------
FastAPI app — EasyScale Legacy Analyzer V2.

Endpoints:
  GET  /health                  → liveness check
  POST /analyze/{clinic_id}     → trigger DNA extraction for a clinic
  GET  /jobs/{job_id}           → poll job status / current_step

Pipeline V2 lives in analyzer/blueprint_v2.py (1 LLM call) and
analyzer/analysis_runner.py (3 phases). Provider selection via LLM_PROVIDER env.
"""

from __future__ import annotations

import logging
import os

import sentry_sdk
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import get_settings
from db import get_db
from analyzer.analysis_runner import run_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Sentry / GlitchTip — errors only for MVP. No-op when SENTRY_DSN unset.
_sentry_dsn = os.environ.get("SENTRY_DSN")
if _sentry_dsn:
    def _scrub_pii(event, _hint):
        # Backstop: drop the full request body (might contain message
        # content). Primary policy is "never put PII in tags/extra".
        if event.get("request", {}).get("data") is not None:
            event["request"]["data"] = None
        return event

    sentry_sdk.init(
        dsn=_sentry_dsn,
        traces_sample_rate=0,
        send_default_pii=False,
        before_send=_scrub_pii,
    )

settings = get_settings()
app = FastAPI(
    title="EasyScale Legacy Analyzer",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Status normalization: DB enum values → API contract values
STATUS_MAP: dict[str, str] = {
    "pending": "pending",
    "queued": "pending",
    "processing": "running",
    "done": "complete",
    "error": "failed",
}


class AnalyzeRequest(BaseModel):
    reference_conversation_ids: list[str] | None = None


class AnalyzeResponse(BaseModel):
    job_id: str
    clinic_id: str
    status: str
    message: str


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


# ------------------------------------------------------------------
# Analyze
# ------------------------------------------------------------------

@app.post("/analyze/{clinic_id}", status_code=202, response_model=AnalyzeResponse)
async def analyze_clinic(
    clinic_id: str,
    background_tasks: BackgroundTasks,
    body: AnalyzeRequest = AnalyzeRequest(),
):
    """
    Trigger DNA extraction for a clinic. Validates clinic exists in sf_clinics,
    creates a la_analysis_jobs row with status='pending' (worker poller ignores
    'pending'), schedules background_tasks → run_analysis. Returns 202 + job_id.
    """
    db = get_db()

    clinic_result = (
        db.table("sf_clinics")
        .select("id, name")
        .eq("id", clinic_id)
        .single()
        .execute()
    )
    if not clinic_result.data:
        raise HTTPException(status_code=404, detail=f"Clinic '{clinic_id}' not found in sf_clinics")

    job_result = db.table("la_analysis_jobs").insert({
        "clinic_id": clinic_id,
        "status": "pending",
        "progress": 0,
        "current_step": "Na fila de processamento",
    }).execute()
    if not job_result.data:
        raise HTTPException(status_code=500, detail="Failed to create analysis job")

    job_id = job_result.data[0]["id"]
    logger.info("Analysis job %s created for clinic %s", job_id, clinic_id)

    background_tasks.add_task(
        run_analysis,
        job_id,
        clinic_id,
        body.reference_conversation_ids,
    )

    return AnalyzeResponse(
        job_id=job_id,
        clinic_id=clinic_id,
        status="pending",
        message="Análise iniciada. Acompanhe via GET /jobs/{job_id}.",
    )


# ------------------------------------------------------------------
# Jobs
# ------------------------------------------------------------------

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    db = get_db()
    result = (
        db.table("la_analysis_jobs")
        .select("*")
        .eq("id", job_id)
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Job not found")

    job = dict(result.data)
    job["normalized_status"] = STATUS_MAP.get(job.get("status", ""), job.get("status"))
    return job


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
    )
