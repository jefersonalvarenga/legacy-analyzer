---
phase: 6
slug: evolution-ingestor
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-16
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pytest.ini` / `pyproject.toml` (existing) |
| **Quick run command** | `pytest tests/test_evolution_ingestor.py -v` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_evolution_ingestor.py -v`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 6-01-01 | 01 | 0 | ING-01, ING-02, ING-03 | unit stub | `pytest tests/test_evolution_ingestor.py -v` | ❌ W0 | ⬜ pending |
| 6-01-02 | 01 | 1 | ING-03 | unit | `pytest tests/test_evolution_ingestor.py::test_resolve_instance_id -v` | ❌ W0 | ⬜ pending |
| 6-01-03 | 01 | 1 | ING-01, ING-02 | unit | `pytest tests/test_evolution_ingestor.py::test_ingest_returns_conversations -v` | ❌ W0 | ⬜ pending |
| 6-01-04 | 01 | 1 | ING-02 | unit | `pytest tests/test_evolution_ingestor.py::test_message_mapping -v` | ❌ W0 | ⬜ pending |
| 6-01-05 | 01 | 2 | ING-01, ING-02, ING-03 | integration | `pytest tests/test_evolution_ingestor.py::test_full_ingest_pipeline -v` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_evolution_ingestor.py` — stubs for ING-01, ING-02, ING-03
- [ ] `tests/conftest.py` — shared fixtures for mock Supabase client

*Existing pytest infrastructure covers framework; only test file stubs needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Read-only guarantee (no INSERT/UPDATE/DELETE) | ING-01 | Requires live DB audit; hard to assert in unit tests that no writes happened | Review `evolution_ingestor.py` — confirm only `.select()` calls, no `.insert()`, `.update()`, `.delete()` present |
| Isolation against cross-clinic leakage | ING-03 | Full isolation test requires two live Evolution instances in shared Supabase | In staging: create two clinic instances, verify messages from clinic B are absent from clinic A's ingest |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
