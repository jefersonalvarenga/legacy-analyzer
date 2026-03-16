---
phase: 06-evolution-ingestor
plan: 01
subsystem: database
tags: [supabase, evolution-api, whatsapp, tdd, read-only, ingestor]

# Dependency graph
requires:
  - phase: none
    provides: "parser.py Conversation + Message dataclasses (existing)"
provides:
  - "ingest_from_evolution() — read-only Evolution Message adapter producing list[Conversation]"
  - "Two-hop instanceId resolution: sf_clinics → Instance UUID"
  - "Group JID exclusion, days_back pagination guard, fromMe-based sender classification"
affects:
  - "07-api-endpoint"
  - "08-pipeline-integration"
  - "worker.py — future pipeline entry point for online analysis"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-hop Supabase lookup: sf_clinics.evolution_instance_id (name) → Instance.id (UUID)"
    - "Stable mock pattern: pre-created table mocks exposed on db._message_table_mock for post-execution inspection"
    - "days_back pagination guard via .gte('messageTimestamp', cutoff) to cap unbounded queries"

key-files:
  created:
    - "analyzer/evolution_ingestor.py"
    - "tests/test_evolution_ingestor.py"
  modified: []

key-decisions:
  - "fromMe flag is the sole source for sender_type classification — pushName is NEVER used for clinic vs patient"
  - "source_filename set to remoteJid string (conversation identifier, not a real filename) for type compatibility with parser.py"
  - "days_back=90 default limit added to ingest_from_evolution() — prevents unbounded data volume without requiring a separate config"
  - "ValueError raised immediately in _resolve_instance_id() before any Message query — fail fast on unknown clinic_id"
  - "raw_line='' for all Evolution-sourced messages — no raw text line exists in API data"

patterns-established:
  - "Evolution ingestor pattern: _resolve_instance_id() → Message query with instanceId + days_back → _group_messages_by_conversation()"
  - "Test mock pattern: _make_db_mock() with pre-created stable table mocks — use db._message_table_mock for call inspection"

requirements-completed: [ING-01, ING-02, ING-03]

# Metrics
duration: 18min
completed: 2026-03-16
---

# Phase 6 Plan 01: Evolution Ingestor Summary

**Read-only Supabase adapter that resolves sf_clinics → Evolution Instance UUID, queries Message rows, and returns list[Conversation] type-compatible with parse_archive() — replacing Archive.zip for online analysis**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-03-16T19:11:01Z
- **Completed:** 2026-03-16T19:29:00Z
- **Tasks:** 2 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments

- Created `analyzer/evolution_ingestor.py` with `ingest_from_evolution()` exported — zero writes, only `.select()` calls
- Two-hop instanceId resolution: `sf_clinics.evolution_instance_id` (name string) → `Instance.id` (UUID) with fail-fast `ValueError`
- All 11 unit tests passing GREEN; full 39-test suite passes with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing test stubs for evolution_ingestor** - `4d328f3` (test)
2. **Task 2: Implement analyzer/evolution_ingestor.py (GREEN)** - `3fe4afd` (feat)

_Note: TDD — Task 1 is RED (ImportError confirmed), Task 2 is GREEN (all 11 pass)_

## Files Created/Modified

- `analyzer/evolution_ingestor.py` — Read-only Evolution Message adapter; exports `ingest_from_evolution()`
- `tests/test_evolution_ingestor.py` — 11 unit tests covering ING-01, ING-02, ING-03 with stdlib mocks

## Decisions Made

- `fromMe` is the sole sender_type classifier: `True → "clinic"`, `False → "patient"`. pushName is only used for the patient display name when `fromMe=False`
- `source_filename` set to `remoteJid` (e.g. `5511912345678@s.whatsapp.net`) for Conversation identity — matches how parser.py uses the field as a conversation ID
- `days_back=90` default added as a pagination guard — prevents unbounded Message fetches as clinic history grows
- `raw_line=""` for all Evolution messages — there is no raw text line in API data, consistent with the field's `default=""` in the dataclass

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test mock helper for stable post-execution call inspection**
- **Found during:** Task 2 (GREEN — running tests)
- **Issue:** The original `_make_db_mock()` used `table_side_effect` that constructed a new `MagicMock()` on every call to `db.table(table_name)`. After `ingest_from_evolution()` executed, calling `db.table("Message")` again returned a fresh mock with no recorded calls — making `eq.call_args` always `None`
- **Fix:** Pre-created stable `clinic_table`, `instance_table`, `message_table` mocks inside `_make_db_mock()`. `table_side_effect` returns the same instance each time. Exposed as `db._message_table_mock`, `db._clinic_table_mock`, `db._instance_table_mock` for post-execution inspection
- **Files modified:** `tests/test_evolution_ingestor.py`
- **Verification:** All 11 tests pass GREEN
- **Committed in:** `3fe4afd` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Required fix — without it, 2 of the 8 specified tests would permanently fail on call inspection. No scope creep; all 8 planned test functions still exist.

## Issues Encountered

None — implementation followed the plan's specified patterns directly.

## User Setup Required

None - no external service configuration required. The ingestor uses the existing `get_db()` singleton (SUPABASE_URL + SUPABASE_SERVICE_KEY already in .env).

## Next Phase Readiness

- `ingest_from_evolution()` is ready to be called from the API endpoint (Phase 7) and pipeline worker (Phase 8)
- Return type is `list[Conversation]` — drop-in replacement for `parse_archive()` output
- Blocker resolved: Evolution Message table schema confirmed via plan context (no live DB access needed for tests)

## Self-Check: PASSED

- analyzer/evolution_ingestor.py: FOUND
- tests/test_evolution_ingestor.py: FOUND
- 06-01-SUMMARY.md: FOUND
- Commit 4d328f3 (test RED): FOUND
- Commit 3fe4afd (feat GREEN): FOUND

---
*Phase: 06-evolution-ingestor*
*Completed: 2026-03-16*
