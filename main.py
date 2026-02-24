"""
main.py
-------
FastAPI application for EasyScale Legacy Analyzer.

Endpoints:
  POST /jobs                     → create a new analysis job (upload zip)
  GET  /jobs/{job_id}            → get job status + progress
  GET  /jobs/{job_id}/report     → get the HTML report for a completed job
  GET  /jobs/{job_id}/export     → download training data export (JSONL)
  GET  /clients                  → list all clients
  POST /clients                  → create a client
  GET  /health                   → health check

The worker (worker.py) polls the DB for queued jobs and processes them.
main.py only handles HTTP — no LLM calls here.
"""

import logging
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from config import get_settings
from db import get_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()
app = FastAPI(
    title="EasyScale Legacy Analyzer",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


# ------------------------------------------------------------------
# Clients
# ------------------------------------------------------------------

class ClientCreate(BaseModel):
    slug: str
    name: str
    sender_name: str | None = None
    config: dict = {}


@app.get("/clients")
def list_clients():
    db = get_db()
    result = db.table("la_clients").select("*").order("created_at").execute()
    return result.data


@app.post("/clients", status_code=201)
def create_client(body: ClientCreate):
    db = get_db()
    result = db.table("la_clients").insert({
        "slug": body.slug,
        "name": body.name,
        "sender_name": body.sender_name or body.name,
        "config": body.config,
    }).execute()
    if not result.data:
        raise HTTPException(status_code=400, detail="Failed to create client")
    return result.data[0]


@app.get("/clients/{slug}")
def get_client(slug: str):
    db = get_db()
    result = (
        db.table("la_clients")
        .select("*")
        .eq("slug", slug)
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Client not found")
    return result.data


# ------------------------------------------------------------------
# Jobs
# ------------------------------------------------------------------

@app.post("/jobs", status_code=201)
async def create_job(
    client_slug: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Upload a .zip file (the outer Archive.zip) and queue an analysis job.
    """
    db = get_db()

    # Resolve client
    client_result = (
        db.table("la_clients")
        .select("id, slug, name, sender_name")
        .eq("slug", client_slug)
        .single()
        .execute()
    )
    if not client_result.data:
        raise HTTPException(status_code=404, detail=f"Client '{client_slug}' not found")

    client = client_result.data

    # Save file to Supabase Storage (or local in dev)
    file_bytes = await file.read()
    filename = file.filename or f"archive_{uuid.uuid4().hex}.zip"

    if settings.app_env == "development":
        # Save locally for dev/testing
        output_dir = Path(settings.reports_output_dir) / "uploads"
        output_dir.mkdir(parents=True, exist_ok=True)
        local_path = output_dir / filename
        local_path.write_bytes(file_bytes)
        file_url = str(local_path)
        logger.info("Saved upload locally: %s", file_url)
    else:
        # Upload to Supabase Storage
        storage_path = f"{client_slug}/{uuid.uuid4().hex}/{filename}"
        db.storage.from_("chat-archives").upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": "application/zip"},
        )
        file_url = storage_path

    # Create the job record
    job_result = db.table("la_analysis_jobs").insert({
        "client_id": client["id"],
        "status": "queued",
        "progress": 0,
        "current_step": "Na fila de processamento",
        "file_url": file_url,
        "original_filename": filename,
    }).execute()

    if not job_result.data:
        raise HTTPException(status_code=500, detail="Failed to create job")

    job = job_result.data[0]
    logger.info("Job %s created for client %s", job["id"], client_slug)

    return {
        "job_id": job["id"],
        "client_slug": client_slug,
        "status": "queued",
        "message": "Job criado com sucesso. O processamento iniciará em breve.",
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    db = get_db()
    result = (
        db.table("la_analysis_jobs")
        .select("*, la_clients(slug, name)")
        .eq("id", job_id)
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Job not found")
    return result.data


@app.get("/jobs/{job_id}/report", response_class=HTMLResponse)
def get_job_report(job_id: str):
    db = get_db()

    # Check job status
    job_result = (
        db.table("la_analysis_jobs")
        .select("status")
        .eq("id", job_id)
        .single()
        .execute()
    )
    if not job_result.data:
        raise HTTPException(status_code=404, detail="Job not found")

    if job_result.data["status"] != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete yet (status: {job_result.data['status']})",
        )

    # Fetch report
    report_result = (
        db.table("la_analysis_reports")
        .select("html_content")
        .eq("job_id", job_id)
        .single()
        .execute()
    )
    if not report_result.data or not report_result.data.get("html_content"):
        raise HTTPException(status_code=404, detail="Report not found")

    return HTMLResponse(content=report_result.data["html_content"])


@app.get("/jobs/{job_id}/export")
def get_training_export(
    job_id: str,
    format: str = "openai_jsonl",
):
    """Download training data export for a completed job."""
    db = get_db()

    if format not in ("openai_jsonl", "anthropic_jsonl", "rag_chunks"):
        raise HTTPException(status_code=400, detail="Invalid format")

    export_result = (
        db.table("la_training_exports")
        .select("file_url, record_count, format")
        .eq("job_id", job_id)
        .eq("format", format)
        .single()
        .execute()
    )
    if not export_result.data:
        raise HTTPException(
            status_code=404,
            detail=f"No export found for job {job_id} in format {format}",
        )

    file_url = export_result.data["file_url"]

    if settings.app_env == "development":
        path = Path(file_url)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Export file not found on disk")
        content = path.read_text(encoding="utf-8")
        return Response(
            content=content,
            media_type="application/jsonl",
            headers={
                "Content-Disposition": f'attachment; filename="{path.name}"'
            },
        )
    else:
        # Stream from Supabase Storage
        raise HTTPException(
            status_code=501,
            detail="Production storage streaming not yet implemented",
        )


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
