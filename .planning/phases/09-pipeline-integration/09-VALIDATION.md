---
phase: 9
slug: pipeline-integration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-17
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pytest.ini or pyproject.toml (existing) |
| **Quick run command** | `pytest tests/test_analysis_runner.py -x -q` |
| **Full suite command** | `pytest tests/ -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_analysis_runner.py -x -q`
- **After every plan wave:** Run `pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 9-01-01 | 01 | 0 | PIPE-02 | migration | `cat supabase/schema.sql | grep clinic_id` | ❌ W0 | ⬜ pending |
| 9-01-02 | 01 | 0 | PIPE-01, PIPE-02 | unit stub | `pytest tests/test_analysis_runner.py -x -q` | ❌ W0 | ⬜ pending |
| 9-02-01 | 02 | 1 | PIPE-01 | unit | `pytest tests/test_analysis_runner.py::test_run_analysis_executes_full_pipeline -x -q` | ❌ W0 | ⬜ pending |
| 9-02-02 | 02 | 1 | PIPE-02 | unit | `pytest tests/test_analysis_runner.py::test_blueprint_saved_with_clinic_id -x -q` | ❌ W0 | ⬜ pending |
| 9-02-03 | 02 | 1 | PIPE-02 | unit | `pytest tests/test_analysis_runner.py::test_resources_persisted_in_same_execution -x -q` | ❌ W0 | ⬜ pending |
| 9-02-04 | 02 | 1 | PIPE-01 | unit | `pytest tests/test_analysis_runner.py::test_dspy_lazy_init -x -q` | ❌ W0 | ⬜ pending |
| 9-02-05 | 02 | 1 | PIPE-01 | unit | `pytest tests/test_analysis_runner.py::test_resources_error_does_not_abort_blueprint -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_analysis_runner.py` — failing stubs for PIPE-01, PIPE-02
- [ ] SQL migration in `supabase/schema.sql` — adds `clinic_id` column to `la_blueprints`, makes `client_id` nullable

*Existing test infrastructure (pytest, conftest, MagicMock) covers all other phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Sofia polls `la_blueprints WHERE clinic_id = UUID` and receives the blueprint | PIPE-02 | Requires live Supabase + Evolution messages | Query `SELECT * FROM la_blueprints WHERE clinic_id = '<test_clinic_id>' ORDER BY created_at DESC LIMIT 1` after triggering job |
| `la_resources` and `la_services` rows visible in Supabase after analysis | PIPE-02 | Requires live data | Check `SELECT * FROM la_resources WHERE clinic_id = '<test_clinic_id>'` after job completes |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
