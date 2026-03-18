---
task_id: 260318-nsj
title: Migration la_services.requires_evaluation + reference_conversation_ids no endpoint
date: 2026-03-18
status: complete
tags: [migration, la_services, endpoint, phase-8.1]
key_files:
  modified:
    - supabase/schema.sql
    - main.py
commit: 117d491
---

# Quick Task 260318-nsj: Migration la_services.requires_evaluation + reference_conversation_ids no endpoint

**One-liner:** Added `requires_evaluation` column migration to `la_services` and optional `reference_conversation_ids` field to `POST /analyze/{clinic_id}` request body as Phase 8.1 groundwork.

## What Was Done

### 1. SQL Migration — supabase/schema.sql

Added a new migration block at the end of the file (after the Phase 9 block):

```sql
-- MIGRATION: Phase 8.1 — requires_evaluation on la_services
-- 2026-03-18
ALTER TABLE la_services
    ADD COLUMN IF NOT EXISTS requires_evaluation BOOLEAN NOT NULL DEFAULT FALSE;
```

The column indicates whether a service requires an in-person evaluation before the procedure (e.g., implants, orthodontics). LA infers this; admin confirms.

### 2. AnalyzeRequest Pydantic model — main.py

New model added before `AnalyzeResponse`:

```python
class AnalyzeRequest(BaseModel):
    reference_conversation_ids: list[str] | None = None
```

### 3. POST /analyze/{clinic_id} endpoint — main.py

Endpoint signature updated to accept optional request body:

```python
async def analyze_clinic(
    clinic_id: str,
    background_tasks: BackgroundTasks,
    body: AnalyzeRequest = AnalyzeRequest(),
):
```

- `body` defaults to an empty `AnalyzeRequest()` so the endpoint remains backward-compatible — callers that send no body continue to work
- `reference_conversation_ids` is available in `body` for Phase 8.1 wiring; `run_analysis()` does not yet consume it

## Live DB Migration Status

The DDL was NOT applied to the live Supabase database because:

1. The `la_services` base table does not yet exist in production (requires Phase 8 migration from `08-01-PLAN.md` to be applied first)
2. No SUPABASE_ACCESS_TOKEN (Management API personal access token) is available in the environment — the service key cannot execute DDL via the Management API

**To apply when ready:**
1. Run the full `supabase/schema.sql` against the live DB (or apply Phase 8 + 9 + 8.1 migration blocks sequentially in the Supabase SQL Editor)
2. The Phase 8.1 block to run:
```sql
ALTER TABLE la_services
    ADD COLUMN IF NOT EXISTS requires_evaluation BOOLEAN NOT NULL DEFAULT FALSE;
```

## Commit

- `117d491` — feat(8.1-nsj): migration la_services.requires_evaluation + reference_conversation_ids on endpoint

## Self-Check

- [x] `supabase/schema.sql` contains Phase 8.1 migration block
- [x] `main.py` contains `AnalyzeRequest` model
- [x] `POST /analyze/{clinic_id}` accepts optional body without breaking existing callers
- [x] Commit 117d491 exists
- [x] No tests added (per constraints — Jeferson tests manually)
- [x] ROADMAP.md not updated (per constraints)
