---
phase: 08-resources-and-services-inference
plan: 01
subsystem: database
tags: [dspy, supabase, tdd, sql-migration, resources-inference, services-inference]

# Dependency graph
requires:
  - phase: 07-fastapi-endpoints
    provides: "clinic_id FK pattern via sf_clinics, la_analysis_jobs table for job_id FK"
  - phase: 06-evolution-ingestor
    provides: "Conversation/Message types used in test fixtures"
provides:
  - "supabase/schema.sql Phase 8 migration block with la_resources and la_services table definitions"
  - "analyzer/resources_inference.py skeleton with 5 public functions (all raise NotImplementedError)"
  - "tests/test_resources_inference.py with 10 failing RED stubs covering RES-01, RES-02, SVC-01, SVC-02"
affects:
  - phase-08-plan-02  # GREEN phase — implements the skeleton
  - phase-09          # wires infer_and_persist_resources() into analysis_runner.py

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "TDD Wave 0 (RED): SQL migration + module skeleton + failing test stubs before any implementation"
    - "DSPy module skeleton mirrors ShadowDNAModule pattern: ResourcesSignature, ResourcesModule, _resources_module global"
    - "Test fixtures: MagicMock for db chains (.table().delete().eq().execute()), patch for DSPy global"

key-files:
  created:
    - analyzer/resources_inference.py
    - tests/test_resources_inference.py
  modified:
    - supabase/schema.sql

key-decisions:
  - "la_resources and la_services use clinic_id FK to sf_clinics (not la_clients) — v1.1 Evolution flow"
  - "confirmed BOOLEAN DEFAULT FALSE: LA suggests, admin confirms, Website creates in sf_resources"
  - "Delete only confirmed=FALSE rows before insert — admin-confirmed rows survive re-analysis"
  - "schedule_type denormalized: stored on each professional row (mirrors sf_resources schema)"
  - "la_services.mention_count index DESC: optimized for 'top services' query pattern"

patterns-established:
  - "Wave 0 TDD pattern: schema first, skeleton second, stubs third — all committed before implementation begins"
  - "ResourcesSignature follows ShadowDNASignature pattern: same corpus-level input, specialized output fields"

requirements-completed:
  - RES-01
  - RES-02
  - SVC-01
  - SVC-02

# Metrics
duration: 3min
completed: 2026-03-16
---

# Phase 8 Plan 01: Resources and Services Inference (TDD RED) Summary

**SQL migration with la_resources and la_services tables + DSPy module skeleton + 10 failing test stubs establishing the TDD contract for professional and service inference**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-16T20:49:12Z
- **Completed:** 2026-03-16T20:52:16Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Phase 8 SQL migration appended to schema.sql: la_resources (professionals + schedule_type, clinic_id FK to sf_clinics) and la_services (procedures with mention_count, clinic_id FK)
- Module skeleton in analyzer/resources_inference.py: all 5 public functions declared with correct signatures, bodies raise NotImplementedError, imports cleanly
- 10 failing test stubs in tests/test_resources_inference.py covering all 4 requirements (RES-01, RES-02, SVC-01, SVC-02) — all fail RED with NotImplementedError, zero import errors

## Task Commits

Each task was committed atomically:

1. **Task 1: SQL migration — la_resources and la_services tables** - `fc82f18` (chore)
2. **Task 2: Module skeleton + failing test stubs (TDD RED)** - `16c4668` (test)

_Note: TDD Wave 0 plan — two commits: schema migration + skeleton/stubs._

## Files Created/Modified

- `supabase/schema.sql` — Phase 8 migration block appended: la_resources, la_services, indexes, RLS
- `analyzer/resources_inference.py` — Public API skeleton: ResourcesResult, ResourcesSignature, ResourcesModule, 5 NotImplementedError stubs
- `tests/test_resources_inference.py` — 10 failing stubs: TestExtractResources (3), TestCountServiceMentions (3), TestPersistResources (2), TestInferAndPersist (2)

## Decisions Made

- `clinic_id` references `sf_clinics(id)` not `la_clients(id)` — these tables belong to the v1.1 Evolution flow, not the legacy Archive.zip flow
- `confirmed BOOLEAN DEFAULT FALSE` chosen over a status enum — boolean covers v1.1 need; rejected/status is v2+ scope
- Delete-then-insert restricted to `confirmed=FALSE` rows — admin-confirmed suggestions survive re-analysis runs
- `schedule_type` stored on each professional row (denormalized) to mirror the sf_resources schema that Sofia consumes

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. SQL migration must be run manually in Supabase SQL Editor when deploying Phase 8.

## Next Phase Readiness

- Plan 08-01 (RED) complete — all test contracts established
- Plan 08-02 (GREEN) can begin: implement init_resources_module, extract_resources, count_service_mentions, persist_resources, infer_and_persist_resources to turn all 10 tests green
- No blockers — test fixtures, DB mock patterns, and DSPy module shape all confirmed working

---
*Phase: 08-resources-and-services-inference*
*Completed: 2026-03-16*
