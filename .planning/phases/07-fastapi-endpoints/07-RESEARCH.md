# Phase 7: FastAPI Endpoints — Research

**Researched:** 2026-03-16
**Domain:** FastAPI background tasks, HTTP job pattern, clinic_id validation against sf_clinics
**Confidence:** HIGH

---

## Summary

Phase 7 adds two HTTP endpoints to the existing `main.py`: `POST /analyze/{clinic_id}` and `GET /jobs/{job_id}`. The first endpoint validates `clinic_id` against the `sf_clinics` table (fail-fast 404), creates a job record in `la_analysis_jobs`, starts the analysis in background (non-blocking), and returns `job_id` immediately — response must arrive in under 1 second. The second endpoint reads the existing job row and returns status + progress.

The existing `main.py` already has a `GET /jobs/{job_id}` endpoint that returns full job detail. Phase 7 enriches it minimally (the status/progress fields it needs already exist in `la_analysis_jobs`) and adds `POST /analyze/{clinic_id}`. The background execution mechanism is already proven in the codebase: `worker.py` processes jobs via a polling loop. Phase 7 does NOT replace this pattern — instead, it adds a trigger path that creates a job and fires the heavy processing in a `BackgroundTasks` callback (FastAPI built-in), which invokes the same processing logic worker.py uses.

The critical design constraint is backward compatibility: all existing endpoints (`POST /jobs`, `GET /jobs/{job_id}`, `GET /jobs/{job_id}/report`, `GET /jobs/{job_id}/export`, `GET /clients`, `POST /clients`, `GET /health`) must remain unchanged. Phase 7 adds routes only, does not modify existing ones.

**Primary recommendation:** Use FastAPI `BackgroundTasks` (built-in, zero new dependencies) to fire the analysis pipeline asynchronously. Create a dedicated `analyzer/analysis_runner.py` module with the processing function that both the background task and (later) the worker.py can invoke. Validate `clinic_id` in `sf_clinics` as the very first step before creating any job record.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| API-01 | `POST /analyze/{clinic_id}` — valida clinic_id em sf_clinics, cria job, inicia analise em background, retorna job_id imediatamente | FastAPI `BackgroundTasks` fires async after response is sent. `la_analysis_jobs` already has all required columns (id, status, progress). sf_clinics lookup with `.single().execute()` is the fail-fast guard, exactly as done in `evolution_ingestor._resolve_instance_id()`. |
| API-02 | `GET /jobs/{job_id}` — retorna status (pending / running / complete / failed) e progresso | Endpoint already exists in main.py. It returns the full job row including `status` and `progress`. Needs minor alignment: status values must be `pending/running/complete/failed` (requirements use different labels than the existing `queued/processing/done/error` enum values). Either normalize in the response or add a mapping layer. |
| API-03 | `main.py` atualizada para suportar novo fluxo sem quebrar comportamento existente | Additive approach: add new routes only. The existing `POST /jobs` (Archive.zip upload) remains untouched. `GET /jobs/{job_id}` is reused. No existing route signatures change. |
</phase_requirements>

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| fastapi | 0.115.0 (installed) | HTTP framework, routing, `BackgroundTasks` | Already the project's HTTP framework. `BackgroundTasks` is built-in — zero new imports needed. |
| pydantic | 2.10.6 (installed) | Request/response models (`AnalyzeResponse`, `JobStatusResponse`) | Already the project's validation library. |
| supabase-py | 2.13.0 (installed) | `sf_clinics` lookup, `la_analysis_jobs` insert/update | Already in use via `get_db()` singleton. |
| httpx | 0.28.1 (installed) | Powers `fastapi.testclient.TestClient` for endpoint tests | Already installed. TestClient uses httpx under the hood. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| fastapi.testclient.TestClient | (included in fastapi) | Synchronous HTTP test client for endpoint unit tests | Use for all API tests — no live server needed. Mocks db calls. |
| pytest | (installed) | Test runner | Same as Phase 6. |
| unittest.mock | (stdlib) | Mock `get_db()` in API tests | Same pattern as Phase 6 ingestor tests. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `BackgroundTasks` (FastAPI built-in) | `asyncio.create_task()` | Both work. `BackgroundTasks` is the FastAPI idiomatic approach; it integrates with dependency injection and lifespan. Use `BackgroundTasks`. |
| `BackgroundTasks` | Celery / RQ / ARQ | Task queues add Redis/broker dependency. Overkill for Phase 7 — job is already tracked in Supabase, no external broker needed at go live. |
| `BackgroundTasks` | Keep existing worker.py polling | Worker continues to work as-is for the old `POST /jobs` flow. For the new `POST /analyze/{clinic_id}` flow, `BackgroundTasks` is simpler — fires immediately when a request arrives rather than waiting for the poll interval. |

**Installation:**

```bash
# No new runtime dependencies — all needed libraries are already in requirements.txt.
```

---

## Architecture Patterns

### Recommended Project Structure

```
main.py                       # add 2 routes: POST /analyze/{clinic_id}, enrich GET /jobs/{job_id}
analyzer/
└── analysis_runner.py        # NEW — processing logic callable from BackgroundTasks and worker.py
tests/
└── test_api_endpoints.py     # NEW — API endpoint tests using TestClient + mock db
```

The core processing logic (ingest → metrics → DSPy → blueprint) lives in `analyzer/analysis_runner.py`. This module is called by:
1. `main.py` `POST /analyze/{clinic_id}` via `BackgroundTasks.add_task(run_analysis, job_id, clinic_id)`
2. (Phase 9) `worker.py` or `run_local.py` directly

This separation ensures Phase 7 does not duplicate the pipeline and Phase 9 can wire it fully.

### Pattern 1: POST /analyze/{clinic_id} — Fail-Fast + Immediate Return

**What:** Validate clinic_id exists in sf_clinics BEFORE creating any job or touching any table. If not found, return 404 immediately. If found, create job in `la_analysis_jobs` with `status = "pending"`, schedule background processing, and return `job_id` in the response body.

**When to use:** Always. The 404 guard must happen synchronously before `BackgroundTasks` is scheduled — once a task is added to `BackgroundTasks`, it cannot be cancelled.

**Example:**

```python
# Source: FastAPI official docs (BackgroundTasks) + existing main.py patterns (HIGH confidence)
from fastapi import BackgroundTasks, HTTPException
from pydantic import BaseModel

class AnalyzeResponse(BaseModel):
    job_id: str
    clinic_id: str
    status: str
    message: str

@app.post("/analyze/{clinic_id}", status_code=202, response_model=AnalyzeResponse)
async def analyze_clinic(clinic_id: str, background_tasks: BackgroundTasks):
    db = get_db()

    # Step 1: Fail-fast — validate clinic_id in sf_clinics BEFORE creating any job
    clinic_result = (
        db.table("sf_clinics")
        .select("id, name")
        .eq("id", clinic_id)
        .single()
        .execute()
    )
    if not clinic_result.data:
        raise HTTPException(status_code=404, detail=f"Clinic '{clinic_id}' not found in sf_clinics")

    clinic = clinic_result.data

    # Step 2: Create job record (status = "pending")
    job_result = db.table("la_analysis_jobs").insert({
        "clinic_id": clinic_id,           # NOTE: schema uses client_id — see note below
        "status": "pending",
        "progress": 0,
        "current_step": "Na fila de processamento",
    }).execute()

    if not job_result.data:
        raise HTTPException(status_code=500, detail="Failed to create analysis job")

    job_id = job_result.data[0]["id"]

    # Step 3: Schedule background processing — fires AFTER response is sent
    background_tasks.add_task(run_analysis, job_id, clinic_id)

    # Step 4: Return immediately (< 1 second)
    return AnalyzeResponse(
        job_id=job_id,
        clinic_id=clinic_id,
        status="pending",
        message="Analise iniciada. Acompanhe o progresso via GET /jobs/{job_id}.",
    )
```

**CRITICAL NOTE on schema column name:** The existing `la_analysis_jobs` table uses `client_id` (FK to `la_clients.id`), not `clinic_id`. Phase 7 introduces `clinic_id` as the parameter (FK to `sf_clinics.id`). Two options:

1. **Option A (recommended):** Add a `clinic_id` column to `la_analysis_jobs` via a SQL migration. Keep `client_id` nullable for backward compatibility with the existing `POST /jobs` flow. The new `POST /analyze/{clinic_id}` inserts into `clinic_id`; old `POST /jobs` inserts into `client_id`.

2. **Option B:** Resolve `clinic_id` → `la_clients.id` before creating the job. This requires a lookup from `sf_clinics` to `la_clients` using the clinic slug or some other linkage — which may not exist yet. Do NOT use this option unless the linkage is confirmed.

**Option A is the safer path.** A single-column migration is low risk and keeps the data model explicit.

### Pattern 2: GET /jobs/{job_id} — Status + Progress

**What:** The endpoint already exists in `main.py` and returns the full job row. Phase 7 must ensure the response includes `status` and `progress` at minimum. The current implementation returns `result.data` directly (all columns) which already includes both fields.

**Status value alignment:** Requirements use `pending / running / complete / failed`. The existing `la_job_status` enum uses `queued / processing / done / error`. There are two acceptable approaches:

1. **Normalize in SQL migration:** Add new enum values or change the enum. This is risky for backward compat.
2. **Map in response (recommended):** In the `GET /jobs/{job_id}` response, add a `normalized_status` field that translates:
   - `queued` → `pending`
   - `processing` → `running`
   - `done` → `complete`
   - `error` → `failed`

This preserves the DB enum while exposing the contract the frontend expects.

**Example:**

```python
# Source: existing main.py GET /jobs/{job_id} + status mapping (HIGH confidence)
STATUS_MAP = {
    "queued": "pending",
    "processing": "running",
    "done": "complete",
    "error": "failed",
    # New statuses inserted by POST /analyze/{clinic_id}:
    "pending": "pending",
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

    job = result.data
    job["normalized_status"] = STATUS_MAP.get(job.get("status", ""), job.get("status"))
    return job
```

### Pattern 3: BackgroundTasks Execution Function

**What:** The function passed to `background_tasks.add_task()` must be non-blocking from FastAPI's perspective. It runs in the same process/thread as the server (not a separate worker). For Phase 7, this function will be a stub or thin wrapper — the full pipeline implementation is Phase 9.

**Example:**

```python
# Source: FastAPI BackgroundTasks docs (HIGH confidence)
def run_analysis(job_id: str, clinic_id: str) -> None:
    """
    Background processing entry point for POST /analyze/{clinic_id}.

    Phase 7: stub — updates job to 'running', then marks 'complete' as placeholder.
    Phase 9: replaced with full pipeline (ingestor → metrics → DSPy → blueprint).
    """
    db = get_db()
    try:
        db.table("la_analysis_jobs").update({
            "status": "processing",
            "progress": 1,
            "current_step": "Iniciando analise...",
        }).eq("id", job_id).execute()

        # Phase 9 will replace this stub with the real pipeline call:
        # from analyzer.analysis_runner import run_full_pipeline
        # run_full_pipeline(job_id, clinic_id)

        logger.info("[%s] Background task triggered for clinic %s", job_id[:8], clinic_id)

    except Exception as e:
        db.table("la_analysis_jobs").update({
            "status": "error",
            "error_message": str(e)[:2000],
        }).eq("id", job_id).execute()
        logger.error("[%s] Background task failed: %s", job_id[:8], e)
```

### Pattern 4: TestClient Testing Pattern

**What:** `fastapi.testclient.TestClient` provides a synchronous HTTP client that calls the app in-process. Combined with `unittest.mock.patch("main.get_db", ...)`, it enables full endpoint testing without a live Supabase or running server.

**Example:**

```python
# Source: FastAPI official docs + existing test patterns in tests/ (HIGH confidence)
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def _make_db_mock_for_analyze(clinic_exists: bool, job_id: str = "job-uuid-001"):
    db = MagicMock()

    # sf_clinics lookup
    clinic_result = MagicMock()
    clinic_result.data = {"id": "clinic-uuid-001", "name": "Sorriso Da Gente"} if clinic_exists else None
    db.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = clinic_result

    # la_analysis_jobs insert
    job_result = MagicMock()
    job_result.data = [{"id": job_id}]
    db.table.return_value.insert.return_value.execute.return_value = job_result

    return db


def test_analyze_returns_job_id_immediately():
    db_mock = _make_db_mock_for_analyze(clinic_exists=True)
    with patch("main.get_db", return_value=db_mock):
        with patch("main.run_analysis"):  # prevent actual background execution in tests
            response = client.post("/analyze/clinic-uuid-001")
    assert response.status_code == 202
    assert "job_id" in response.json()


def test_analyze_returns_404_for_unknown_clinic():
    db_mock = _make_db_mock_for_analyze(clinic_exists=False)
    with patch("main.get_db", return_value=db_mock):
        response = client.post("/analyze/nonexistent-clinic-id")
    assert response.status_code == 404
```

### Anti-Patterns to Avoid

- **Creating job before validating clinic_id:** If the db insert runs before the sf_clinics check, you get orphan job records in the DB on bad requests. Always validate first.
- **Blocking the HTTP response with heavy work:** The analysis pipeline (DSPy, embeddings) takes minutes. Never `await` it in the endpoint handler — always defer with `BackgroundTasks`.
- **Modifying existing endpoint signatures:** `POST /jobs` and `GET /jobs/{job_id}` are used by existing clients. Additive changes only.
- **Using `asyncio.create_task()` instead of `BackgroundTasks`:** FastAPI `BackgroundTasks` integrates with the request lifecycle and exception handling. `asyncio.create_task()` can silently swallow exceptions.
- **Not wrapping background function in try/except:** Unhandled exceptions in `BackgroundTasks` are silently dropped (FastAPI logs them but does not propagate). Always wrap with try/except and update job status to "error" on failure.
- **Not patching `run_analysis` in tests:** `TestClient` in synchronous mode will execute `BackgroundTasks` synchronously after the response — if not patched, tests will attempt real analysis logic.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Background task execution | Custom threading / subprocess | `fastapi.BackgroundTasks` | Built-in, integrated with request lifecycle, no extra process management |
| HTTP test client | Custom test runner | `fastapi.testclient.TestClient` | Part of FastAPI, uses `httpx` already installed, handles ASGI protocol correctly |
| Job status polling (client-side) | WebSocket / SSE for push | `GET /jobs/{job_id}` polling | Requirements specify polling (GET endpoint), not push. Keep it simple for go live. |
| clinic_id validation | Custom regex or format check | `sf_clinics` lookup via supabase-py | The authoritative check is "does this row exist in sf_clinics" — format validation alone is insufficient |

**Key insight:** FastAPI already provides everything needed: BackgroundTasks for async execution, TestClient for testing, Pydantic for response models, HTTPException for 404. Zero new dependencies required.

---

## Common Pitfalls

### Pitfall 1: la_analysis_jobs schema uses client_id not clinic_id

**What goes wrong:** `la_analysis_jobs.client_id` is a FK to `la_clients.id`, which is the old "client" concept from the Archive.zip flow. The new flow uses `clinic_id` (FK to `sf_clinics.id`). Inserting into `client_id` with a clinic UUID will violate the FK constraint or silently insert a wrong value.

**Why it happens:** The schema predates the Evolution API integration. The `la_clients` table is the v0 concept; `sf_clinics` is the v1.1 concept. They are not the same table.

**How to avoid:** Add a `clinic_id UUID REFERENCES sf_clinics(id)` column (nullable) to `la_analysis_jobs`. Insert into `clinic_id` for new flow; leave `client_id` null. Existing `POST /jobs` continues to insert into `client_id`.

**Warning signs:** Supabase returns FK violation error on `la_analysis_jobs` insert, or job row has null clinic_id when queried later.

### Pitfall 2: BackgroundTasks runs synchronously in TestClient

**What goes wrong:** `fastapi.testclient.TestClient` (synchronous mode) executes `BackgroundTasks` callbacks synchronously after the response is returned. If the background function calls `get_db()` without mocking, it will attempt a real Supabase connection in the test.

**Why it happens:** TestClient's synchronous mode processes everything in one pass; `BackgroundTasks` is not truly deferred.

**How to avoid:** In tests, patch `run_analysis` (or whatever the background function is named) with `unittest.mock.patch`. This prevents the stub/pipeline from executing during endpoint tests.

**Warning signs:** Test passes locally but tries to connect to Supabase; or test hangs/fails with connection error.

### Pitfall 3: Status enum mismatch between requirements and DB

**What goes wrong:** Requirements specify `pending / running / complete / failed`. The existing `la_job_status` enum in Supabase is `queued / processing / done / error`. Inserting `"pending"` into the status column will fail if the enum doesn't include it.

**Why it happens:** The v0 schema was designed before the v1.1 API contract.

**How to avoid:** Two options: (A) alter the enum to add `pending` (or rename `queued`), or (B) use `queued` as the DB value and map to `pending` in API responses. Option B is safer for backward compat — no schema migration needed for the status column itself.

**Warning signs:** Supabase returns "invalid input value for enum la_job_status" on insert.

### Pitfall 4: Forgetting clinic_sender_name when creating jobs

**What goes wrong:** The `ingest_from_evolution()` function (Phase 6) requires `clinic_sender_name` — the WhatsApp display name the clinic uses. When creating a job from `POST /analyze/{clinic_id}`, the background processing will need this. If not stored or derivable at that point, the ingestor call in Phase 9 will fail.

**Why it happens:** The `sf_clinics` table may not have a `sender_name` column equivalent. The `la_clients.sender_name` column exists but it's a different table.

**How to avoid:** When creating the job, also fetch `sf_clinics.name` (or an equivalent display name column) and store it in the job record or derive it at processing time. For Phase 7, record `clinic_name` in the job metadata JSON or a dedicated column.

**Warning signs:** Phase 9 fails with `TypeError` or `ValueError` when calling `ingest_from_evolution(clinic_id, clinic_sender_name)` because `clinic_sender_name` is not available.

### Pitfall 5: GET /jobs/{job_id} already exists — don't duplicate

**What goes wrong:** Creating a new endpoint `GET /jobs/{job_id}` with different behavior instead of extending the existing one causes two conflicting routes.

**Why it happens:** Temptation to create a clean v2 endpoint without touching existing code.

**How to avoid:** Extend the existing `GET /jobs/{job_id}` handler in-place. Add the `normalized_status` mapping. The existing handler already queries all columns including `status` and `progress`. API-02 is satisfied by the existing endpoint with minimal modification.

---

## Code Examples

### FastAPI BackgroundTasks — Minimal Verified Pattern

```python
# Source: FastAPI official docs https://fastapi.tiangolo.com/tutorial/background-tasks/ (HIGH)
from fastapi import BackgroundTasks, FastAPI

app = FastAPI()

def write_notification(email: str, message: str = ""):
    with open("log.txt", mode="w") as email_file:
        content = f"notification for {email}: {message}"
        email_file.write(content)

@app.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_notification, email, message="some notification")
    return {"message": "Notification sent in the background"}
```

### TestClient with BackgroundTasks patch

```python
# Source: FastAPI testing docs + common pattern in projects using BackgroundTasks (HIGH)
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

def test_endpoint_does_not_block(client: TestClient):
    # Patch the background function so it doesn't actually run
    with patch("main.run_analysis") as mock_run:
        response = client.post("/analyze/some-clinic-id")
    assert response.status_code == 202
    mock_run.assert_called_once()  # verify it was scheduled
```

### Supabase sf_clinics lookup (fail-fast guard)

```python
# Source: existing main.py patterns + evolution_ingestor.py (HIGH — same project)
def _validate_clinic_id(db, clinic_id: str) -> dict:
    """
    Verify clinic_id exists in sf_clinics. Raises HTTPException 404 if not found.
    Returns the clinic row dict on success.
    """
    result = (
        db.table("sf_clinics")
        .select("id, name")
        .eq("id", clinic_id)
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(
            status_code=404,
            detail=f"Clinic '{clinic_id}' not found. Validate clinic_id in sf_clinics."
        )
    return result.data
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `POST /jobs` with file upload (Archive.zip) | `POST /analyze/{clinic_id}` (no file — reads from Evolution) | Phase 7 (now) | Frontend no longer uploads a zip; uses clinic_id from onboarding context |
| Worker polls for queued jobs every N seconds | BackgroundTasks fires immediately on POST | Phase 7 (now) | Lower latency from trigger to analysis start; both mechanisms coexist |
| client_id FK to la_clients | clinic_id FK to sf_clinics | Phase 7 (now) | Aligns with Sofia's data model; requires schema migration |

**Deprecated/outdated:**
- `POST /jobs` remains in place for backward compat but is not the v1.1 trigger path. The new trigger is `POST /analyze/{clinic_id}`.

---

## Schema Migration Required

Phase 7 requires one SQL migration against the shared Supabase:

```sql
-- Add clinic_id column to la_analysis_jobs (nullable — backward compat with POST /jobs)
ALTER TABLE la_analysis_jobs
    ADD COLUMN IF NOT EXISTS clinic_id UUID REFERENCES sf_clinics(id) ON DELETE SET NULL;

-- Optional: index for lookups by clinic
CREATE INDEX IF NOT EXISTS idx_la_jobs_clinic_id ON la_analysis_jobs(clinic_id);
```

This migration is additive and safe. The existing `client_id` column and FK remain unchanged. Old jobs from `POST /jobs` have `clinic_id = NULL`; new jobs from `POST /analyze/{clinic_id}` have `client_id = NULL` and `clinic_id = <uuid>`.

---

## Open Questions

1. **Does sf_clinics have a display name / sender_name column?**
   - What we know: `la_clients.sender_name` exists. `sf_clinics.name` exists. It's unclear if `sf_clinics` has a `sender_name` or `whatsapp_name` column specifically.
   - What's unclear: What column to use as `clinic_sender_name` for `ingest_from_evolution()` when called from the background task.
   - Recommendation: For Phase 7 (stub only), use `sf_clinics.name` as the placeholder. Phase 9 will wire the real value. Verify the `sf_clinics` schema before Phase 9.

2. **Should la_analysis_jobs.client_id remain NOT NULL?**
   - What we know: `la_analysis_jobs.client_id UUID NOT NULL REFERENCES la_clients(id)` in the current schema. New jobs from `POST /analyze/{clinic_id}` don't have a `la_clients` record.
   - What's unclear: Whether making `client_id` nullable will break any existing code path (worker.py reads `client_id` to fetch client config).
   - Recommendation: Make `client_id` nullable in the migration. Update worker.py to handle `client_id = NULL` gracefully (skip old-style processing if `clinic_id` is set). This is Phase 9 scope, but the migration must happen in Phase 7.

3. **Background task vs. worker.py coexistence**
   - What we know: worker.py polls `WHERE status = 'queued'`. If the new `POST /analyze/{clinic_id}` sets status = `'queued'` and also fires BackgroundTasks, both the background task AND the worker may attempt to process the same job concurrently.
   - Recommendation: Set initial status = `'pending'` for new jobs (a status the worker does NOT poll for). BackgroundTasks is the sole executor for `clinic_id`-based jobs in Phase 7. Worker continues to poll for `'queued'` (Archive.zip flow). This requires adding `'pending'` to the `la_job_status` enum.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (already in use) |
| Config file | None detected — pytest runs with default settings |
| Quick run command | `pytest tests/test_api_endpoints.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| API-01 | POST /analyze/{clinic_id} returns 202 + job_id within response | unit (TestClient + mock db) | `pytest tests/test_api_endpoints.py::TestAnalyzeEndpoint::test_returns_job_id_immediately -x` | Wave 0 |
| API-01 | POST /analyze/{clinic_id} schedules background task (not blocking) | unit (TestClient + mock bg) | `pytest tests/test_api_endpoints.py::TestAnalyzeEndpoint::test_background_task_scheduled -x` | Wave 0 |
| API-01 | POST /analyze/{clinic_id} with unknown clinic_id returns 404 | unit (TestClient + mock db) | `pytest tests/test_api_endpoints.py::TestAnalyzeEndpoint::test_returns_404_for_unknown_clinic -x` | Wave 0 |
| API-01 | No job record created when clinic_id is invalid (fail-fast) | unit (TestClient + mock db) | `pytest tests/test_api_endpoints.py::TestAnalyzeEndpoint::test_no_job_created_on_404 -x` | Wave 0 |
| API-02 | GET /jobs/{job_id} returns status and progress fields | unit (TestClient + mock db) | `pytest tests/test_api_endpoints.py::TestGetJobEndpoint::test_returns_status_and_progress -x` | Wave 0 |
| API-02 | GET /jobs/{job_id} with unknown job_id returns 404 | unit (TestClient + mock db) | `pytest tests/test_api_endpoints.py::TestGetJobEndpoint::test_returns_404_for_unknown_job -x` | Wave 0 |
| API-03 | GET /health still returns 200 after changes | unit (TestClient) | `pytest tests/test_api_endpoints.py::TestExistingEndpoints::test_health_unchanged -x` | Wave 0 |
| API-03 | POST /jobs still exists and accepts file upload | unit (TestClient + mock db) | `pytest tests/test_api_endpoints.py::TestExistingEndpoints::test_post_jobs_still_works -x` | Wave 0 |

All tests use mocked `get_db()` and patched `run_analysis` — no live Supabase required.

### Sampling Rate

- **Per task commit:** `pytest tests/test_api_endpoints.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_api_endpoints.py` — all tests listed above (new file, does not exist yet)
- [ ] SQL migration: `ALTER TABLE la_analysis_jobs ADD COLUMN clinic_id UUID` — needed before integration tests touch real DB
- [ ] SQL migration: add `'pending'` to `la_job_status` enum — needed to avoid conflict with worker.py polling

*(No framework install needed — pytest + TestClient + httpx all already installed)*

---

## Sources

### Primary (HIGH confidence)

- `main.py` (codebase) — existing endpoint structure, Supabase patterns, response shapes, existing `GET /jobs/{job_id}` implementation
- `supabase/schema.sql` (codebase) — `la_analysis_jobs` column names, `la_job_status` enum values, FK constraints
- `worker.py` (codebase) — `_update_job()`, `_set_progress()` patterns; confirms `status` values `queued/processing/done/error`
- `analyzer/evolution_ingestor.py` (codebase, Phase 6 output) — `_resolve_instance_id()` fail-fast pattern (same pattern for clinic_id validation)
- `tests/test_evolution_ingestor.py` (codebase) — confirms `unittest.mock` pattern for testing with mocked db
- FastAPI official docs (https://fastapi.tiangolo.com/tutorial/background-tasks/) — `BackgroundTasks` usage, TestClient integration

### Secondary (MEDIUM confidence)

- FastAPI testing docs (https://fastapi.tiangolo.com/tutorial/testing/) — `TestClient` and background task mocking patterns
- `db.py` (codebase) — `get_db()` singleton, confirms it can be patched via `unittest.mock`

### Tertiary (LOW confidence — flag for validation)

- `sf_clinics` column names beyond `id`, `name`, `evolution_instance_id`: not verified against live Sofia schema. `sender_name` or equivalent column may or may not exist.
- `la_analysis_jobs.client_id` nullability impact on worker.py: `worker.py` reads `client_id` in `process_job()` to fetch `la_clients` config. Making it nullable requires verifying all code paths that assume `client_id` is always present.

---

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — zero new dependencies; FastAPI, supabase-py, pytest all already installed and in use
- Architecture (BackgroundTasks pattern): HIGH — verified against FastAPI official docs + existing codebase patterns
- Architecture (schema migration): HIGH — schema.sql read directly; migration requirements are clear
- Open questions: MEDIUM — sf_clinics columns and client_id nullability require verification against live DB before coding
- Pitfalls: HIGH — derived from direct code analysis + schema inspection

**Research date:** 2026-03-16
**Valid until:** 2026-04-16
