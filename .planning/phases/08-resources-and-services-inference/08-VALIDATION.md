---
phase: 8
slug: resources-and-services-inference
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-16
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | none — existing pytest setup |
| **Quick run command** | `pytest tests/test_resources_inference.py -x -q` |
| **Full suite command** | `pytest tests/ -x -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_resources_inference.py -x -q`
- **After every plan wave:** Run `pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 8-01-01 | 01 | 0 | RES-01, RES-02, SVC-01, SVC-02 | unit stubs | `pytest tests/test_resources_inference.py -x -q` | ❌ W0 | ⬜ pending |
| 8-01-02 | 01 | 1 | RES-01 | unit (mock DSPy) | `pytest tests/test_resources_inference.py::TestExtractResources -x -q` | ❌ W0 | ⬜ pending |
| 8-01-03 | 01 | 1 | RES-02 | unit (mock DSPy) | `pytest tests/test_resources_inference.py::TestExtractResources::test_schedule_type_by_professional -x -q` | ❌ W0 | ⬜ pending |
| 8-01-04 | 01 | 1 | SVC-01, SVC-02 | unit (pure Python) | `pytest tests/test_resources_inference.py::TestCountServiceMentions -x -q` | ❌ W0 | ⬜ pending |
| 8-01-05 | 01 | 1 | RES-01, SVC-01 | unit (mock db) | `pytest tests/test_resources_inference.py::TestPersistResources -x -q` | ❌ W0 | ⬜ pending |
| 8-01-06 | 01 | 2 | RES-01, RES-02, SVC-01, SVC-02 | unit (mock db+DSPy) | `pytest tests/test_resources_inference.py::TestInferAndPersist -x -q` | ❌ W0 | ⬜ pending |
| 8-01-07 | 01 | 2 | all | full suite | `pytest tests/ -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_resources_inference.py` — stubs for RES-01, RES-02, SVC-01, SVC-02
- [ ] `analyzer/resources_inference.py` — new module (skeleton with raise NotImplementedError)
- [ ] SQL migration in `supabase/schema.sql` — `la_resources` and `la_services` tables

*No new framework install needed — pytest, unittest.mock, dspy, supabase-py already installed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `la_resources` and `la_services` tables exist in Supabase | RES-01, SVC-01 | Requires live Supabase connection | Run migration SQL in Supabase dashboard; confirm tables appear in schema browser |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
