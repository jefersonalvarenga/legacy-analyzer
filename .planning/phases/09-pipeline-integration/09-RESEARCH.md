# Phase 9: Pipeline Integration - Research

**Researched:** 2026-03-17
**Domain:** Python pipeline orchestration, FastAPI BackgroundTasks, Supabase upsert, DSPy integration
**Confidence:** HIGH — all findings based directly on existing source code in this repository

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PIPE-01 | Pipeline completo (metricas, DSPy, desfechos, Shadow DNA, blueprint) funciona com mensagens do Evolution | `analysis_runner.py` has a documented stub with explicit comments pointing to Phase 9 implementation; all pipeline modules (`metrics.py`, `dspy_pipeline.py`, `outcome_detection.py`, `shadow_dna.py`, `financial_kpis.py`, `blueprint.py`) are already implemented and unit-tested — Phase 9 only needs to wire them |
| PIPE-02 | Blueprint salvo em `la_blueprints` com `clinic_id` correto para a Sofia consumir | `la_blueprints` table exists in schema.sql but has `client_id` (not `clinic_id`) FK — **requires SQL migration** to add `clinic_id UUID REFERENCES sf_clinics`; Sofia's polling query `WHERE clinic_id = UUID ORDER BY created_at DESC LIMIT 1` must work |
</phase_requirements>

---

## Summary

Phase 9 is a wiring phase, not a building phase. All individual pipeline components (ingestor, metrics, DSPy, outcomes, Shadow DNA, financial KPIs, blueprint assembly, resources/services persistence) are fully implemented across Phases 6-8. The only missing piece is a single orchestrator function in `analysis_runner.py` that sequences these components end-to-end.

The critical architectural challenge is the `la_blueprints` table schema. Currently `la_blueprints` has a `client_id` FK to `la_clients` (the legacy zip-upload clients table). Phase 9 needs blueprints keyed to `sf_clinics.id` so Sofia can query `WHERE clinic_id = UUID`. This requires a SQL migration before any code changes.

The second critical constraint is execution atomicity: PIPE-02's success criterion states `la_resources` and `la_services` are persisted "during the same execution that saves the blueprint." This means `infer_and_persist_resources()` must be called in the same `run_analysis()` execution as `save_blueprint_to_supabase()`, with proper error handling so a DSPy failure on resources does not abort the blueprint write.

**Primary recommendation:** Implement `run_analysis()` in `analysis_runner.py` as a linear synchronous pipeline with granular progress updates. Add `clinic_id` column to `la_blueprints`. Keep the function in `analysis_runner.py` (it is already the registered BackgroundTasks handler — do not move it). No new files are needed.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI BackgroundTasks | (already installed) | Runs `run_analysis()` after HTTP response | Already wired in `main.py`; no changes to caller |
| Supabase Python client | (already installed) | DB reads and writes via `.table().insert()` | Used by all existing phases |
| DSPy | (already installed) | LLM extraction for semantic analysis and resources | `configure_lm()` already initializes all modules including `init_resources_module()` |
| pytest + unittest.mock | (already installed) | TDD pattern matching Phases 6-8 | All phase tests use `MagicMock` + `patch` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Python `logging` | stdlib | Progress tracking to stdout | Every pipeline step logs with `[job_id[:8]]` prefix — Phase 9 must follow same pattern |
| Python `datetime` | stdlib | Timestamps for blueprint metadata | `generated_at` field in blueprint |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Linear synchronous pipeline | Async pipeline with `asyncio` | worker.py uses `async def process_job()` but `run_analysis()` is sync — keep sync for simplicity and testability |
| `analysis_runner.py` as sole orchestrator | New `pipeline_runner.py` | `analysis_runner.py` is already registered in `main.py`'s `background_tasks.add_task(run_analysis, ...)` — creating a new file would require changing `main.py` which was stabilized in Phase 7 |

**Installation:** No new packages required. All dependencies are already installed.

---

## Architecture Patterns

### Recommended Project Structure

No new files. Changes are confined to:
```
analyzer/
├── analysis_runner.py   # REPLACE stub with full pipeline (Phase 9 target)
supabase/
└── schema.sql           # ADD Phase 9 migration block (clinic_id on la_blueprints)
tests/
└── test_pipeline_integration.py  # NEW — covers PIPE-01 and PIPE-02
```

### Pattern 1: Linear Progress Pipeline

**What:** A single synchronous function that sequences all analysis steps, updates `la_analysis_jobs.progress` and `la_analysis_jobs.current_step` after each step, and wraps everything in a try/except that marks the job as `error` on failure.

**When to use:** Always — this is the pattern used by `worker.py:process_job()` which is the reference implementation for the legacy zip flow.

**Reference implementation** (from `worker.py`):
```python
# Source: worker.py:process_job() — the canonical pattern
def _update_job(job_id: str, **kwargs):
    db = get_db()
    db.table("la_analysis_jobs").update({
        **kwargs,
        "updated_at": datetime.utcnow().isoformat(),
    }).eq("id", job_id).execute()

def _set_progress(job_id: str, progress: int, step: str):
    logger.info("[%s] %d%% — %s", job_id[:8], progress, step)
    _update_job(job_id, progress=progress, current_step=step)
```

**Phase 9 pipeline steps** (derived from `worker.py` + Phase 8 requirements):
```
Step 1 (5%):   Mark job status="processing"
Step 2 (10%):  Resolve clinic name from sf_clinics
Step 3 (15%):  ingest_from_evolution(clinic_id, clinic_sender_name) → conversations
Step 4 (20%):  Guard: 0 conversations → fail job with human-readable message
Step 5 (30%):  compute_metrics(conv) for each conversation → metrics_list
Step 6 (35%):  configure_lm() if not already configured → DSPy modules ready
Step 7 (70%):  analyze_conversation() per conversation → analyses (slowest step, 35-70%)
Step 8 (75%):  detect_outcome() per conversation → outcome_results
Step 9 (80%):  extract_shadow_dna(conversations, clinic_name, analyses) → shadow_dna
Step 10 (85%): aggregate_outcomes(outcome_results) → outcome_summary
Step 11 (87%): compute_financial_kpis(outcome_summary, shadow_dna) → financial_kpis
Step 12 (90%): aggregate_metrics(metrics_list) → agg_metrics
Step 13 (92%): build_blueprint(...) → blueprint dict
Step 14 (95%): save blueprint to la_blueprints (INSERT with clinic_id)
Step 15 (97%): infer_and_persist_resources(conversations, clinic_name, clinic_id, job_id, shadow_dna, db)
Step 16 (100%): Mark job status="done"
```

**Critical ordering constraint** (from `resources_inference.py` docstring):
> `infer_and_persist_resources()` MUST be called after `extract_shadow_dna()` — it reads `shadow_dna.local_procedures` for SVC-01.

### Pattern 2: Resilient Step Execution

**What:** Individual pipeline steps are wrapped in try/except that logs warnings but does not abort the entire pipeline. Only truly fatal errors (no conversations, DB write failure for blueprint) should fail the job.

**When to use:** For `infer_and_persist_resources()` and DSPy steps — a single LLM extraction failure should not prevent the blueprint from being saved.

**Example (from existing code style):**
```python
# Source: worker.py DSPy error handling pattern
try:
    analysis = analyze_conversation(conv.messages, clinic_name)
    analyses.append(analysis)
except Exception as e:
    logger.warning("[%s] DSPy failed for conv: %s", job_id[:8], e)
    analyses.append(SemanticAnalysis(error=str(e)))
```

### Pattern 3: blueprint → la_blueprints with clinic_id

**What:** After `build_blueprint()` returns a dict, insert into `la_blueprints` with both `job_id` and `clinic_id`. The `clinic_id` column must be added via migration (see SQL Migration section below).

**Sofia's polling query (must work unchanged):**
```sql
SELECT * FROM la_blueprints WHERE clinic_id = '<uuid>' ORDER BY created_at DESC LIMIT 1
```

**Insert pattern:**
```python
# Source: worker.py:process_job() — analogous pattern for la_analysis_reports
db.table("la_blueprints").insert({
    "job_id": job_id,
    "clinic_id": clinic_id,        # NEW column — requires migration
    "blueprint": blueprint_dict,   # JSONB column
}).execute()
```

### Anti-Patterns to Avoid

- **Re-calling `configure_lm()` on every job invocation:** `configure_lm()` should be called once at startup. In the `run_analysis()` context (FastAPI BackgroundTasks), the caller (`main.py`) does not call `configure_lm()`. The pipeline must call it lazily — check if `dspy.settings.lm` is configured before calling. See the worker's startup pattern: `_fast_lm, _consolidation_lm = configure_lm(...)` called once in `run_worker()`.
- **Calling `infer_and_persist_resources()` before `extract_shadow_dna()`:** The function reads `shadow_dna.local_procedures`. Calling it early produces empty services.
- **Failing the entire job when `infer_and_persist_resources()` raises:** Resources are suggestions, not blocking. Log and continue.
- **Writing to `la_blueprints` with only `client_id`:** The pre-existing column is `client_id` (FK to `la_clients`). Phase 9 adds `clinic_id` (FK to `sf_clinics`). Both can coexist; for Evolution-triggered jobs `client_id` should be NULL.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Conversation ingestion | Custom Evolution query | `evolution_ingestor.ingest_from_evolution()` | Already implements two-hop lookup, group JID filtering, days_back guard |
| Per-conversation metrics | Custom KPI math | `metrics.compute_metrics()` + `aggregate_metrics()` | Already handles response times, confirmation rates, silence periods |
| DSPy analysis | Raw LLM calls | `dspy_pipeline.analyze_conversation()` | Already handles truncation, _safe_float, _safe_list normalization |
| Outcome detection | Custom classifier | `outcome_detection.detect_outcome()` + `aggregate_outcomes()` | Already has deterministic ghosting pre-check to save LLM calls |
| Shadow DNA | New LLM extraction | `shadow_dna.extract_shadow_dna()` | Already handles payment mentions pre-scan, aggregate LM context |
| Financial KPIs | Manual ticket math | `financial_kpis.compute_financial_kpis()` | Already handles LLM ticket estimation fallback |
| Blueprint assembly | Manual dict construction | `blueprint.build_blueprint()` | Already conforms to `implementation_blueprint_schema.json` |
| Resources/services | New DSPy module | `resources_inference.infer_and_persist_resources()` | Already handles delete-before-insert, empty conversations guard |

**Key insight:** Phase 9 is an integration phase. Every building block exists and is tested. The only code to write is the orchestrator in `analysis_runner.py` and a SQL migration.

---

## Common Pitfalls

### Pitfall 1: DSPy modules not initialized when `run_analysis()` fires

**What goes wrong:** `analysis_runner.run_analysis()` is called via FastAPI `BackgroundTasks`. Unlike `worker.py`, which calls `configure_lm()` at startup in `run_worker()`, the FastAPI app startup in `main.py` does NOT call `configure_lm()`. All DSPy modules (`_sentiment_module`, `_outcome_module`, `_shadow_module`, `_resources_module`, `_ticket_estimator`) are `None` until `configure_lm()` is called.

**Why it happens:** Phase 7's `analysis_runner.py` stub was intentionally minimal. The DSPy initialization concern was deferred to Phase 9.

**How to avoid:** Call `configure_lm()` once at the start of `run_analysis()` using `get_settings()` values. Use a module-level flag (`_lm_configured: bool = False`) to avoid re-initializing on every job. Or call it unconditionally — DSPy's `configure()` is idempotent.

**Warning signs:** `analyze_conversation()` returns `SemanticAnalysis(error="DSPy not configured. Call configure_lm() first.")`.

### Pitfall 2: `la_blueprints` missing `clinic_id` column

**What goes wrong:** `INSERT INTO la_blueprints (clinic_id=...) ...` raises a Supabase/PostgreSQL error: column `clinic_id` does not exist.

**Why it happens:** The original `la_blueprints` schema (from Phase 1-5) was designed for `la_clients`-based jobs and has `client_id` only. The Evolution flow (Phase 7+) uses `sf_clinics` UUIDs.

**How to avoid:** Run the Phase 9 SQL migration BEFORE deploying Phase 9 code:
```sql
-- Phase 9 migration
ALTER TABLE la_blueprints
    ADD COLUMN IF NOT EXISTS clinic_id UUID REFERENCES sf_clinics(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_la_blueprints_clinic_id ON la_blueprints(clinic_id);
```

**Warning signs:** `500 Internal Server Error` from Supabase on blueprint insert; Supabase error log shows `column "clinic_id" of relation "la_blueprints" does not exist`.

### Pitfall 3: `clinic_sender_name` not available in `run_analysis()`

**What goes wrong:** `ingest_from_evolution(clinic_id, clinic_sender_name)` requires the clinic's WhatsApp display name. `run_analysis(job_id, clinic_id)` does not receive it. Passing an empty string or None produces conversations where all clinic messages have an empty sender name, degrading DSPy analysis quality.

**Why it happens:** Phase 7's API stub signature is `run_analysis(job_id, clinic_id)` — no clinic name is passed.

**How to avoid:** Inside `run_analysis()`, do a second `sf_clinics` lookup to get the clinic name before calling `ingest_from_evolution()`:
```python
clinic_result = db.table("sf_clinics").select("id, name").eq("id", clinic_id).single().execute()
clinic_name = clinic_result.data["name"]  # use as clinic_sender_name
```

**Warning signs:** Blueprint `agent_identity.name` contains generic fallback "Assistente Virtual " (empty suffix). All Shadow DNA `greeting_example` values reference an empty sender name.

### Pitfall 4: `resources_inference` error aborts blueprint save

**What goes wrong:** `infer_and_persist_resources()` raises an exception (e.g., DSPy module not initialized, Supabase FK violation). If not caught, the exception propagates up and `run_analysis()` marks the job as `error` — but the blueprint was already written. This leaves the job in `error` state while `la_blueprints` has a valid blueprint.

**How to avoid:** Wrap `infer_and_persist_resources()` in a try/except. Log the error but do not re-raise. Mark the job as `done` regardless. Resources are suggestions — a failed resources step should not block Sofia from consuming the blueprint.

### Pitfall 5: Job status stuck at `processing` on unhandled exception

**What goes wrong:** Any unhandled exception in `run_analysis()` leaves `la_analysis_jobs.status = "processing"` permanently. `GET /jobs/{job_id}` returns `running` forever.

**How to avoid:** Wrap the entire pipeline body in try/except:
```python
def run_analysis(job_id: str, clinic_id: str) -> None:
    db = get_db()
    try:
        # ... full pipeline ...
    except Exception as exc:
        logger.error("[%s] Pipeline failed: %s", job_id[:8], exc)
        try:
            db.table("la_analysis_jobs").update({
                "status": "error",
                "error_message": str(exc)[:2000],
            }).eq("id", job_id).execute()
        except Exception:
            pass
```

This pattern is already present in `analysis_runner.py`'s stub — the Phase 9 implementation must preserve it.

---

## Code Examples

Verified patterns from existing source code:

### Clinic name lookup (needed in run_analysis)
```python
# Source: main.py:analyze_clinic() — pattern already used for validation
clinic_result = (
    db.table("sf_clinics")
    .select("id, name")
    .eq("id", clinic_id)
    .single()
    .execute()
)
clinic_name = clinic_result.data["name"]
```

### Blueprint insert to la_blueprints (new clinic_id field)
```python
# Source: worker.py:process_job() — analogous la_analysis_reports insert
db.table("la_blueprints").insert({
    "job_id": job_id,
    "clinic_id": clinic_id,
    "blueprint": blueprint_dict,
}).execute()
```

### Lazy DSPy initialization guard
```python
# Source: worker.py:run_worker() — startup initialization pattern
# For analysis_runner.py, call once per process:
_lm_initialized = False

def _ensure_lm_configured():
    global _lm_initialized
    if _lm_initialized:
        return
    from config import get_settings
    from analyzer.dspy_pipeline import configure_lm
    s = get_settings()
    configure_lm(
        openai_api_key=s.llm_api_key,
        model=s.llm_model,
        base_url=s.openai_base_url,
        anthropic_api_key=s.anthropic_api_key,
        consolidator_model=s.llm_model_consolidator,
    )
    _lm_initialized = True
```

### Test pattern: mock db for pipeline integration (from test_resources_inference.py)
```python
# Source: tests/test_resources_inference.py:_make_db()
from unittest.mock import MagicMock, patch
db = MagicMock()
# MagicMock auto-creates chained attributes: db.table("x").insert(...).execute() works
```

### TDD test pattern for run_analysis
```python
# Pattern: patch evolution ingestor, db, and all DSPy modules to test orchestration
@patch("analyzer.analysis_runner.ingest_from_evolution")
@patch("analyzer.analysis_runner.get_db")
def test_run_analysis_saves_blueprint(mock_db, mock_ingest):
    mock_ingest.return_value = [make_fake_conversation()]
    mock_db.return_value = make_db_mock()
    run_analysis("job-uuid-001", "clinic-uuid-001")
    # assert blueprint insert was called
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Archive.zip ingestion via `parse_archive()` | Evolution API ingestion via `ingest_from_evolution()` | Phase 6 | `run_analysis()` must use `ingest_from_evolution()`, not `parse_archive()` |
| `la_blueprints.client_id` (FK to `la_clients`) | `la_blueprints.clinic_id` (FK to `sf_clinics`) | Phase 9 (NEW) | Migration needed; old `client_id` column becomes nullable/ignored for Evolution jobs |
| `worker.py` as sole pipeline runner (polling-based) | `analysis_runner.run_analysis()` as HTTP-triggered pipeline | Phase 7 stub, Phase 9 implementation | Two parallel paths: `worker.py` handles legacy zip jobs; `analysis_runner.py` handles Evolution jobs |
| `analysis_runner.py` stub (marks processing, does nothing) | `analysis_runner.py` full pipeline | Phase 9 | `POST /analyze/{clinic_id}` goes live |

**Deprecated/outdated:**
- `worker.py` Knowledge Consolidator call (`consolidate_knowledge()`) — KC is suspended per STATE.md decision. Phase 9 pipeline must NOT include KC steps.
- `worker.py` embeddings step — embeddings are expensive and optional. Phase 9 does not include embeddings (not in PIPE-01 or PIPE-02 requirements).
- `worker.py` training exports step — not required for go-live. Phase 9 does not include export generation.

---

## Open Questions

1. **DSPy LM initialization in production: one-time or per-job?**
   - What we know: `worker.py` initializes once at startup in `run_worker()`. `run_analysis()` is called per-request by FastAPI BackgroundTasks — no shared startup hook.
   - What's unclear: Whether `dspy.configure()` is thread-safe when multiple concurrent jobs run.
   - Recommendation: Use a module-level `_lm_initialized` flag. Call `configure_lm()` once on first job. Acceptable for v1.1 with 3 controlled clinics (no concurrency risk).

2. **`la_blueprints.client_id` nullability for Evolution jobs**
   - What we know: Phase 7 migration made `la_analysis_jobs.client_id` nullable. `la_blueprints.client_id` is still NOT NULL in the original schema.
   - What's unclear: Whether current production schema already has `client_id` nullable on `la_blueprints`.
   - Recommendation: Phase 9 migration must make `client_id` nullable on `la_blueprints` (or simply omit it for Evolution inserts if already nullable). Planner should include `ALTER TABLE la_blueprints ALTER COLUMN client_id DROP NOT NULL;` in the migration.

3. **`clinic_name` as `client_slug` in blueprint metadata**
   - What we know: `build_blueprint()` accepts `client_slug` and `client_name`. For Evolution jobs there is no `la_clients` record — no slug.
   - What's unclear: Whether Sofia uses `blueprint.metadata.client_slug` or `clinic_id` for routing.
   - Recommendation: Use `clinic_id` as the `client_slug` in blueprint metadata for Evolution jobs (or omit it / use a derived slug from clinic name). Sofia is documented to use `WHERE clinic_id = UUID` for lookup, not metadata fields.

---

## SQL Migration Required (Phase 9)

```sql
-- ============================================================
-- MIGRATION: Phase 9 — Pipeline Integration
-- ============================================================

-- 1. Add clinic_id FK to la_blueprints (Sofia polling uses this)
ALTER TABLE la_blueprints
    ADD COLUMN IF NOT EXISTS clinic_id UUID REFERENCES sf_clinics(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_la_blueprints_clinic_id ON la_blueprints(clinic_id);

-- 2. Make client_id nullable (Evolution-triggered jobs have no la_clients record)
ALTER TABLE la_blueprints
    ALTER COLUMN client_id DROP NOT NULL;
```

---

## Validation Architecture

`workflow.nyquist_validation` is not set to false — validation section is included.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already installed, used in Phases 6-8) |
| Config file | none — `pytest` run from project root |
| Quick run command | `pytest tests/test_pipeline_integration.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PIPE-01 | `run_analysis()` executes full pipeline (ingest → metrics → DSPy → outcomes → shadow DNA → blueprint) without raising | unit (mocked) | `pytest tests/test_pipeline_integration.py::TestRunAnalysis::test_full_pipeline_completes -x` | ❌ Wave 0 |
| PIPE-01 | `run_analysis()` marks job status=`done` when pipeline succeeds | unit (mocked) | `pytest tests/test_pipeline_integration.py::TestRunAnalysis::test_job_marked_done -x` | ❌ Wave 0 |
| PIPE-01 | `run_analysis()` marks job status=`error` when fatal exception occurs | unit (mocked) | `pytest tests/test_pipeline_integration.py::TestRunAnalysis::test_job_marked_error_on_exception -x` | ❌ Wave 0 |
| PIPE-01 | `run_analysis()` marks job status=`error` when 0 conversations returned | unit (mocked) | `pytest tests/test_pipeline_integration.py::TestRunAnalysis::test_empty_conversations_fails_job -x` | ❌ Wave 0 |
| PIPE-02 | `la_blueprints` insert includes `clinic_id` (not only `job_id`) | unit (mocked) | `pytest tests/test_pipeline_integration.py::TestBlueprintPersistence::test_blueprint_saved_with_clinic_id -x` | ❌ Wave 0 |
| PIPE-02 | `la_resources` and `la_services` are persisted in same execution as blueprint | unit (mocked) | `pytest tests/test_pipeline_integration.py::TestBlueprintPersistence::test_resources_persisted_same_run -x` | ❌ Wave 0 |
| PIPE-02 | Resources failure does not prevent blueprint from being saved | unit (mocked) | `pytest tests/test_pipeline_integration.py::TestBlueprintPersistence::test_resources_failure_does_not_abort -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_pipeline_integration.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_pipeline_integration.py` — covers PIPE-01 and PIPE-02 (7 test cases above)
- [ ] No new conftest.py fixtures needed — existing `_make_db()` pattern from `test_resources_inference.py` is sufficient

*(Note: Sofia contract tests in `tests/test_sofia_contract.py` test the blueprint structure and can be reused as-is via a manual integration test once a real clinic_id is available.)*

---

## Sources

### Primary (HIGH confidence)
- `analyzer/analysis_runner.py` — stub with explicit Phase 9 TODO comments
- `analyzer/resources_inference.py` — documents call order constraint (after shadow_dna)
- `worker.py` — canonical pipeline pattern (reference implementation)
- `main.py` — BackgroundTasks registration and clinic validation pattern
- `supabase/schema.sql` — `la_blueprints` table definition (confirms `client_id` column, missing `clinic_id`)
- `analyzer/blueprint.py` — `build_blueprint()` signature
- `analyzer/evolution_ingestor.py` — `ingest_from_evolution()` signature
- `analyzer/shadow_dna.py` — `extract_shadow_dna()` signature and `init_shadow_module()` requirement
- `analyzer/outcome_detection.py` — `detect_outcome()`, `aggregate_outcomes()` signatures
- `analyzer/financial_kpis.py` — `compute_financial_kpis()` signature
- `analyzer/metrics.py` — `compute_metrics()`, `aggregate_metrics()` signatures
- `tests/test_resources_inference.py` — canonical mock/patch TDD pattern for this codebase
- `.planning/STATE.md` — KC suspended, no embeddings in Phase 9 scope

### Secondary (MEDIUM confidence)
- Memory file (MEMORY.md) — Sofia integration contract (`WHERE clinic_id = UUID ORDER BY created_at DESC LIMIT 1`)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in use, no new dependencies
- Architecture: HIGH — pipeline steps derived directly from `worker.py` reference implementation and Phase 8 call order constraint
- Pitfalls: HIGH — DSPy initialization gap and missing `clinic_id` column confirmed by reading `analysis_runner.py` stub and `schema.sql`
- SQL migration: HIGH — confirmed by reading `la_blueprints` table definition in `schema.sql`

**Research date:** 2026-03-17
**Valid until:** 2026-04-17 (stable codebase — no fast-moving dependencies)
