---
phase: 7
slug: fastapi-endpoints
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-16
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (already installed) |
| **Config file** | none — existing pytest default config |
| **Quick run command** | `pytest tests/test_api_endpoints.py -x -q` |
| **Full suite command** | `pytest tests/ -x -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_api_endpoints.py -x -q`
- **After every plan wave:** Run `pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 7-01-01 | 01 | 0 | API-01, API-02, API-03 | unit stub | `pytest tests/test_api_endpoints.py -x -q` | ❌ W0 | ⬜ pending |
| 7-01-02 | 01 | 1 | API-01 | unit (TestClient) | `pytest tests/test_api_endpoints.py::TestAnalyzeEndpoint -x -q` | ❌ W0 | ⬜ pending |
| 7-01-03 | 01 | 1 | API-01 | unit (TestClient) | `pytest tests/test_api_endpoints.py::TestAnalyzeEndpoint::test_returns_404_for_unknown_clinic -x` | ❌ W0 | ⬜ pending |
| 7-01-04 | 01 | 1 | API-02 | unit (TestClient) | `pytest tests/test_api_endpoints.py::TestGetJobEndpoint -x -q` | ❌ W0 | ⬜ pending |
| 7-01-05 | 01 | 1 | API-03 | unit (TestClient) | `pytest tests/test_api_endpoints.py::TestExistingEndpoints -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_api_endpoints.py` — failing test stubs for all API-01, API-02, API-03 behaviors (new file)
- [ ] SQL migration stub: `ALTER TABLE la_analysis_jobs ADD COLUMN IF NOT EXISTS clinic_id UUID REFERENCES sf_clinics(id)` — in `supabase/schema.sql`
- [ ] SQL migration stub: add `'pending'` to `la_job_status` enum — prevents conflict with worker.py polling on `'queued'`

*(No framework install needed — pytest, TestClient, httpx all already installed)*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Background task fires after response (true async) | API-01 | TestClient runs BackgroundTasks synchronously; true async behavior requires a live server | Start uvicorn locally, POST /analyze/{valid_clinic_id}, verify response arrives in < 1s, then poll GET /jobs/{job_id} to see status transition from pending → running |
| Schema migration applies cleanly to live DB | API-01 | No test DB in this phase | Apply migration SQL in Supabase dashboard, verify `la_analysis_jobs` has `clinic_id` column |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
