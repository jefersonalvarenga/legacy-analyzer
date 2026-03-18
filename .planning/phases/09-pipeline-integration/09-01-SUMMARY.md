---
phase: 09-pipeline-integration
plan: "01"
subsystem: database, testing
tags: [supabase, postgresql, tdd, pytest, unittest-mock, la_blueprints, sf_clinics]

# Dependency graph
requires:
  - phase: 08-resources-and-services-inference
    provides: "infer_and_persist_resources() and la_resources/la_services schema"
  - phase: 07-fastapi-endpoints
    provides: "run_analysis() stub and la_analysis_jobs with clinic_id"
provides:
  - "Phase 9 SQL migration: clinic_id column on la_blueprints with FK to sf_clinics"
  - "Phase 9 SQL migration: client_id nullable on la_blueprints for Evolution-triggered jobs"
  - "idx_la_blueprints_clinic_id index for Sofia polling query"
  - "7 TDD RED stubs in tests/test_analysis_runner.py covering PIPE-01 and PIPE-02"
affects:
  - "09-02 (GREEN plan) — implements run_analysis() to pass these tests"
  - "Sofia integration — la_blueprints.clinic_id enables WHERE clinic_id = '<uuid>' query"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "@patch with create=True to stub imports that don't exist yet in TDD RED phase"

key-files:
  created:
    - tests/test_analysis_runner.py
  modified:
    - supabase/schema.sql

key-decisions:
  - "Use create=True on @patch decorators so RED stubs work before ingest_from_evolution and infer_and_persist_resources are imported in analysis_runner.py"
  - "la_blueprints.client_id made nullable to support Evolution-triggered jobs that have no la_clients record"
  - "clinic_id FK on la_blueprints uses ON DELETE SET NULL to preserve historical blueprints if clinic is deleted"

patterns-established:
  - "TDD RED stubs with create=True: patch names that don't exist yet use create=True so collection passes, tests fail in body"

requirements-completed: [PIPE-01, PIPE-02]

# Metrics
duration: 2min
completed: 2026-03-18
---

# Phase 9 Plan 01: Pipeline Integration TDD RED Summary

**SQL migration adds clinic_id FK + index on la_blueprints and makes client_id nullable; 7 TDD RED stubs in test_analysis_runner.py define the PIPE-01 and PIPE-02 observable contract for the GREEN plan**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-18T01:05:50Z
- **Completed:** 2026-03-18T01:08:10Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Phase 9 SQL migration block appended to supabase/schema.sql: `clinic_id UUID REFERENCES sf_clinics(id)`, index `idx_la_blueprints_clinic_id`, and `client_id DROP NOT NULL` on la_blueprints
- 7 failing TDD stubs created in `tests/test_analysis_runner.py` — all collected by pytest, all FAIL (RED confirmed), zero collection errors, zero skips
- Stubs cover full PIPE-01 contract (ingest call, done/error status transitions, empty conversations guard) and PIPE-02 contract (blueprint clinic_id, resource inference co-execution, resource error isolation)

## Task Commits

Each task was committed atomically:

1. **Task 1: SQL migration** - `f0613e6` (chore)
2. **Task 2: TDD RED stubs** - `eff72ee` (test)

## Files Created/Modified

- `supabase/schema.sql` - Phase 9 migration block appended (clinic_id + index + client_id nullable on la_blueprints)
- `tests/test_analysis_runner.py` - 7 TDD RED stubs for PIPE-01 and PIPE-02 (created)

## Decisions Made

- Used `create=True` on `@patch` decorators so test collection succeeds even though `ingest_from_evolution` and `infer_and_persist_resources` are not yet imported in `analysis_runner.py`. This is the standard approach for TDD RED in this codebase: test body assertions fail, patch setup succeeds.
- `client_id DROP NOT NULL` on `la_blueprints` mirrors the same constraint change already applied to `la_analysis_jobs` in Phase 7 — consistent pattern for Evolution-triggered jobs.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added `create=True` to all @patch decorators**
- **Found during:** Task 2 (TDD RED stubs)
- **Issue:** First pytest run showed `AttributeError: module does not have attribute 'ingest_from_evolution'` because the Phase 7 stub analysis_runner.py does not import `ingest_from_evolution` or `infer_and_persist_resources` yet. Tests errored at collection-level patch setup instead of failing in the test body.
- **Fix:** Added `create=True` to all four `@patch` decorators so unittest.mock creates the attribute on the module namespace at patch time. Tests now fail in the test body with `AssertionError` as expected for RED.
- **Files modified:** tests/test_analysis_runner.py
- **Verification:** `pytest tests/test_analysis_runner.py -q` — 7 collected, 7 FAILED, 0 errors, 0 skips
- **Committed in:** eff72ee (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Required for correct TDD RED behavior. Zero scope creep.

## Issues Encountered

None beyond the auto-fixed patch setup issue above.

## Next Phase Readiness

- Schema migration ready to run in Supabase SQL Editor
- 7 failing tests define the exact contract for Phase 9 GREEN plan (09-02)
- `analyzer/analysis_runner.py` stub still in place — 09-02 replaces its body with the real pipeline

## Self-Check: PASSED

- tests/test_analysis_runner.py: FOUND
- supabase/schema.sql: FOUND
- commit f0613e6: FOUND
- commit eff72ee: FOUND

---
*Phase: 09-pipeline-integration*
*Completed: 2026-03-18*
