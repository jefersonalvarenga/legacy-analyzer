---
phase: 07-fastapi-endpoints
plan: 01
subsystem: api
tags: [fastapi, background-tasks, pydantic, supabase, tdd]

requires:
  - phase: 06-evolution-ingestor
    provides: ingest_from_evolution() callable that run_analysis() will invoke in Phase 9

provides:
  - POST /analyze/{clinic_id} — HTTP 202 endpoint that validates clinic, creates job, fires background task
  - GET /jobs/{job_id} — enriched with normalized_status field (pending/running/complete/failed)
  - analyzer/analysis_runner.py — run_analysis() background stub, ready for Phase 9 pipeline wiring
  - supabase/schema.sql migration — pending enum value + clinic_id FK + client_id nullable
affects:
  - phase-09-pipeline
  - worker.py (must NOT poll 'pending' status — only 'queued')
  - frontend (receives job_id + uses GET /jobs/{job_id} for polling)

tech-stack:
  added: []
  patterns:
    - FastAPI BackgroundTasks pattern for non-blocking HTTP responses
    - STATUS_MAP dict for stable API contract across DB enum changes
    - Fail-fast clinic_id validation before any DB mutation
    - MagicMock + TestClient pattern for FastAPI endpoint testing without real DB

key-files:
  created:
    - tests/test_api_endpoints.py
    - analyzer/analysis_runner.py
  modified:
    - main.py
    - supabase/schema.sql

key-decisions:
  - "POST /analyze/{clinic_id} returns 202 immediately — background processing via FastAPI BackgroundTasks"
  - "Status 'pending' used only by Evolution-triggered jobs — worker polls 'queued' only, no conflict"
  - "STATUS_MAP normalizes DB enum to stable API contract (pending/running/complete/failed)"
  - "analysis_runner.py stub logs intent and marks processing — Phase 9 replaces stub with pipeline call"

patterns-established:
  - "Fail-fast validation: check sf_clinics BEFORE creating la_analysis_jobs row (no orphaned jobs)"
  - "BackgroundTasks.add_task fires AFTER response sent — HTTP latency never depends on LLM/DB workload"
  - "Normalized status field pattern: DB enums can evolve freely, API contract stays stable via STATUS_MAP"

requirements-completed: [API-01, API-02, API-03]

duration: 2min
completed: 2026-03-16
---

# Phase 7 Plan 01: FastAPI Endpoints Summary

**POST /analyze/{clinic_id} (202+job_id) with fail-fast clinic validation, BackgroundTasks stub, and normalized_status on GET /jobs/{job_id} — 9 new tests all GREEN, full suite 48/48**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-16T19:39:00Z
- **Completed:** 2026-03-16T19:40:56Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments
- POST /analyze/{clinic_id} returns HTTP 202 with job_id in under 1 second — clinic_id validated against sf_clinics before any job record is created
- analyzer/analysis_runner.py stub ready for Phase 9 pipeline wiring — updates job to 'processing', handles errors gracefully
- GET /jobs/{job_id} enriched with normalized_status field using STATUS_MAP (queued→pending, processing→running, done→complete, error→failed)
- SQL migration appended to schema.sql: pending enum value, client_id nullable, clinic_id FK column with index
- All pre-existing endpoints (GET /health, POST /jobs, GET /jobs/{job_id}/report, etc.) preserved unchanged — backward compat verified

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing tests + SQL migration (TDD RED)** - `51b2160` (test)
2. **Task 2: Implement POST /analyze/{clinic_id} + analysis_runner stub (TDD GREEN)** - `886b810` (feat)
3. **Task 3: Enrich GET /jobs/{job_id} with normalized_status (TDD GREEN)** - `a87176e` (feat)

_Note: TDD tasks have separate test → feat commits per TDD flow_

## Files Created/Modified
- `tests/test_api_endpoints.py` - 9 tests covering API-01/02/03 + backward compat (TestClient + MagicMock)
- `analyzer/analysis_runner.py` - run_analysis() background stub, Phase 9 wires the full pipeline here
- `main.py` - BackgroundTasks import + AnalyzeResponse model + POST /analyze/{clinic_id} + STATUS_MAP + enriched get_job()
- `supabase/schema.sql` - Migration block: pending enum + client_id nullable + clinic_id FK + index

## Decisions Made
- Used FastAPI BackgroundTasks (not Celery/worker polling) for immediate HTTP response with async processing
- 'pending' status reserved for Evolution-triggered jobs; worker continues polling 'queued' only — zero conflict
- STATUS_MAP as module-level dict (not computed per request) — simple, O(1) lookup, easy to extend

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
**Database migration required.** Apply in Supabase SQL Editor after deploying Phase 7:

```sql
-- From supabase/schema.sql (bottom of file)
ALTER TYPE la_job_status ADD VALUE IF NOT EXISTS 'pending';
ALTER TABLE la_analysis_jobs ALTER COLUMN client_id DROP NOT NULL;
ALTER TABLE la_analysis_jobs ADD COLUMN IF NOT EXISTS clinic_id UUID REFERENCES sf_clinics(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_la_jobs_clinic_id ON la_analysis_jobs(clinic_id);
```

## Next Phase Readiness
- POST /analyze/{clinic_id} ready for frontend integration
- analysis_runner.run_analysis() stub ready — Phase 9 replaces stub block with ingest_from_evolution() + pipeline
- GET /jobs/{job_id} polling ready with normalized_status for frontend state machine

---
*Phase: 07-fastapi-endpoints*
*Completed: 2026-03-16*
