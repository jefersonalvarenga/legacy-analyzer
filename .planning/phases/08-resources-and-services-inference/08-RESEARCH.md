# Phase 8: Resources and Services Inference — Research

**Researched:** 2026-03-16
**Domain:** DSPy corpus-level extraction, Supabase table design, NER-style inference from Portuguese WhatsApp chat
**Confidence:** HIGH

---

## Summary

Phase 8 adds two new capabilities to the Legacy Analyzer: inferring clinic professionals (`la_resources`) and clinic services/procedures (`la_services`) from the aggregate set of WhatsApp conversations, then persisting them as suggestions to Supabase with `clinic_id` correct. Both tables **do not yet exist** in `supabase/schema.sql` — a SQL migration is the first deliverable of the phase.

The inference work is split between two complementary extraction paths. The first is a **corpus-level DSPy module** that receives a sample of conversations and extracts named professionals (e.g., "Dra. Ana", "Dr. Carlos") plus a `schedule_type` signal (`single` / `by_professional` / `by_room`). The second extracts **services and procedures** across all conversations and counts mention frequency. A critical observation is that the existing `ShadowDNA` module in `analyzer/shadow_dna.py` already extracts `local_procedures` as part of its LLM pass — Phase 8 should reuse that signal rather than invoke a separate LLM call for procedures.

The persistence layer is straightforward: both tables receive one write call per analysis run — an upsert (or delete-then-insert) keyed on `clinic_id`. The admin sees the suggestions, confirms, and the Website creates the confirmed records in `sf_resources`. The LA's role ends at suggestions.

**Primary recommendation:** Create `analyzer/resources_inference.py` with one corpus-level DSPy signature for professionals + schedule_type, and one pure-Python counter for service frequency leveraging the `local_procedures` list already extracted by Shadow DNA. Persist via `get_db()` upsert into two new tables: `la_resources` and `la_services`. Call `infer_and_persist_resources()` from `analysis_runner.py` at the end of the analysis pipeline (Phase 9 wires the call; Phase 8 builds the module).

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| RES-01 | LA infere profissionais mencionados nas conversas (ex: "Dra. Ana", "Dr. Carlos") → salva em `la_resources` | ShadowDNA already extracts `local_procedures` via a corpus-level DSPy pass. A sibling DSPy Signature (`ResourcesSignature`) using the same `conversations_sample` input can extract professional names. NER regex (titles: Dra., Dr., Dentista, etc.) provides a pure-Python fallback to validate LLM output. |
| RES-02 | LA infere `schedule_type` (single / by_professional / by_room) → salva em `la_resources` | `schedule_type` is a classification task across all conversations. Signal: if multiple distinct professionals are mentioned, `by_professional`; if "sala" / "consultório" mentions cluster around appointment scheduling, `by_room`; otherwise `single`. DSPy Signature with the same conversations_sample can classify this as a single-select field. |
| SVC-01 | LA infere procedimentos e servicos oferecidos pela clinica (ex: implante, clareamento, ortodontia) → salva em `la_services` | `ShadowDNA.local_procedures` is already extracted by the existing LLM pass. Phase 8 should consume this list directly. If ShadowDNA is not yet available at the time of calling (Phase 9 context), a lightweight service extractor DSPy signature or regex scan of clinic messages provides the input. |
| SVC-02 | `la_services` inclui frequencia de mencao | Pure-Python string search across all clinic messages: for each service name in the extracted list, `count = sum(1 for conv in conversations for msg in conv.clinic_messages if service_name.lower() in msg.content.lower())`. No LLM needed for counting. |
</phase_requirements>

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| dspy | installed (project default) | Corpus-level extraction: ResourcesSignature extracts professionals + schedule_type from conversation sample | Already the project's LLM framework. Same pattern as ShadowDNAModule — one corpus-level call, not per-conversation. |
| supabase-py | 2.13.0 (installed) | Upsert extracted records into `la_resources` and `la_services` | Already the project's Supabase client. `get_db()` singleton in `db.py`. |
| re (stdlib) | — | Professional name regex validation (Dra., Dr., Dentista, etc.) to post-process LLM output | No new dependency. Provides deterministic validation of LLM-extracted names. |
| collections.Counter (stdlib) | — | Service mention frequency counting across all conversations | No new dependency. Same pattern used in shadow_dna._compute_quantitative(). |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | installed | Unit tests for inference module and DB persistence logic | Same pattern as Phases 6–7. Pure unit tests with mocked DB and mocked DSPy module. |
| unittest.mock | stdlib | Mock `get_db()` and `dspy.Predict` in tests | Same pattern as existing test files. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Reuse `ShadowDNA.local_procedures` for SVC-01 | New DSPy call for services | Shadow DNA already makes the LLM call and has the result. Reusing it saves one LLM call per analysis (~cost and latency). Reuse is the right choice. |
| DSPy for professional extraction | spaCy NER | spaCy NER on pt-BR medical text requires a trained model (not present). DSPy with gpt-4o-mini handles Portuguese professional titles better out-of-the-box. |
| Corpus-level DSPy call for resources | Per-conversation extraction + aggregation | Per-conversation extraction multiplies LLM calls (100+ conversations = 100+ calls). Corpus-level (sample of 10–15 conversations) is 1 call. Much cheaper, same quality for entity discovery. |
| Upsert keyed on (clinic_id, name) | Delete-then-insert per clinic_id | Both work. Upsert is atomic. For simplicity at go live, delete-then-insert is easier to reason about (replace all suggestions on each run). |

**Installation:**

```bash
# No new runtime dependencies. All needed libraries are already in requirements.txt.
```

---

## Architecture Patterns

### Recommended Project Structure

```
analyzer/
├── resources_inference.py   # NEW — DSPy + pure Python extraction of professionals, schedule_type, services
├── shadow_dna.py            # existing — already extracts local_procedures (SVC-01 source)
├── analysis_runner.py       # existing stub — Phase 9 wires call to infer_and_persist_resources()
└── ...

supabase/
└── schema.sql               # existing — add la_resources and la_services migration block

tests/
└── test_resources_inference.py   # NEW — unit tests for RES-01, RES-02, SVC-01, SVC-02
```

### Pattern 1: ResourcesSignature — Corpus-Level DSPy Extraction

**What:** One DSPy Signature that receives a conversation sample (same format as `ShadowDNASignature`) and returns: a list of professional names detected, and a `schedule_type` classification.

**When to use:** Once per analysis run, after Shadow DNA has been extracted (since ShadowDNA already does the corpus pass; a sibling module reuses the same sample).

**Example:**

```python
# Source: existing shadow_dna.py ShadowDNASignature pattern (HIGH confidence)
import dspy

class ResourcesSignature(dspy.Signature):
    """
    Analise o conjunto de conversas de WhatsApp de uma clínica odontológica e identifique:
    1. Os profissionais mencionados (dentistas, doutores, especialistas) com seus títulos.
    2. O tipo de agendamento praticado pela clínica.

    Responda em português. Baseie-se exclusivamente nos textos fornecidos.
    """
    conversations_sample: str = dspy.InputField(
        desc="Amostra de conversas da clínica (até 10 conversas, início e fim de cada uma)"
    )
    clinic_name: str = dspy.InputField(desc="Nome da clínica")

    professionals: list = dspy.OutputField(
        desc=(
            "Lista de nomes de profissionais detectados nas conversas. "
            "Inclua título quando presente (ex: ['Dra. Ana', 'Dr. Carlos', 'Dr. Marcos']). "
            "Retorne lista vazia [] se nenhum profissional for identificado explicitamente. "
            "NÃO invente nomes — inclua apenas os que aparecem no texto."
        )
    )
    schedule_type: str = dspy.OutputField(
        desc=(
            "Tipo de agendamento: "
            "'by_professional' se a clínica agenda por profissional específico (ex: 'com a Dra. Ana'); "
            "'by_room' se agenda por sala ou consultório; "
            "'single' se há apenas um profissional ou se o agendamento não menciona profissionais. "
            "Retorne exatamente um dos três valores: single, by_professional, by_room."
        )
    )
```

### Pattern 2: Professional Name Validation (Pure Python Post-Processing)

**What:** After the LLM returns `professionals`, apply a regex filter to remove hallucinated non-person strings. Entries that contain known professional title prefixes are kept; others are either kept with lower confidence or flagged for review.

**When to use:** Always, after LLM extraction. The LLM may occasionally include clinic names or service names in the `professionals` list.

**Example:**

```python
# Source: codebase analysis + WhatsApp Brazilian clinic naming conventions (HIGH confidence)
import re

_PROFESSIONAL_TITLE_RE = re.compile(
    r"\b(Dr\.?|Dra\.?|Doutor|Doutora|Dentista|Cirurgi[oã]|Especialista|Prof\.?|Professora?)\b",
    re.IGNORECASE,
)

def _validate_professional_name(name: str) -> bool:
    """Return True if name looks like a healthcare professional."""
    return bool(_PROFESSIONAL_TITLE_RE.search(name)) or (
        # Short names without title but not a procedure word
        len(name.split()) <= 3 and not any(
            keyword in name.lower()
            for keyword in ("implante", "clareamento", "ortodontia", "consulta", "exame")
        )
    )

def _filter_professionals(raw_list: list) -> list[str]:
    """Filter and deduplicate the LLM-returned professionals list."""
    seen = set()
    result = []
    for item in raw_list:
        name = str(item).strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        result.append(name)
    return result
```

### Pattern 3: Service Mention Frequency — Pure Python Counter

**What:** Given a list of service names (from `ShadowDNA.local_procedures`), count how many clinic messages across all conversations mention each service. No LLM call needed.

**When to use:** After Shadow DNA extraction is complete and `dna.local_procedures` is populated.

**Example:**

```python
# Source: shadow_dna._compute_quantitative() pattern (HIGH confidence — same codebase)
from collections import Counter
from analyzer.parser import Conversation

def count_service_mentions(
    service_names: list[str],
    conversations: list[Conversation],
) -> list[dict]:
    """
    Count how many clinic messages mention each service name.

    Returns list of dicts: [{"name": "implante", "mention_count": 42}, ...]
    sorted by mention_count descending.
    """
    counts: Counter = Counter()

    all_clinic_content = [
        msg.content.lower()
        for conv in conversations
        for msg in conv.clinic_messages
    ]

    for service in service_names:
        service_lower = service.lower()
        counts[service] = sum(
            1 for content in all_clinic_content if service_lower in content
        )

    # Return sorted by frequency, highest first; include zero-count services
    return [
        {"name": name, "mention_count": count}
        for name, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)
        if name.strip()
    ]
```

### Pattern 4: Database Persistence — la_resources and la_services

**What:** After extraction, persist results into the two new tables using a delete-then-insert strategy per `clinic_id`. This ensures each analysis run replaces the previous suggestions cleanly.

**When to use:** At the end of each analysis run, after `dna` and `resources_result` are available.

**Example:**

```python
# Source: worker.py db.table().insert() patterns (HIGH confidence — same codebase)
def persist_resources(
    db,
    clinic_id: str,
    job_id: str,
    professionals: list[str],
    schedule_type: str,
    services: list[dict],   # [{"name": str, "mention_count": int}]
) -> None:
    """
    Persist resources and services suggestions for a clinic.
    Replaces any previous suggestions (delete-then-insert per clinic_id).
    """
    # Delete previous suggestions for this clinic
    db.table("la_resources").delete().eq("clinic_id", clinic_id).execute()
    db.table("la_services").delete().eq("clinic_id", clinic_id).execute()

    # Insert professionals
    for name in professionals:
        db.table("la_resources").insert({
            "clinic_id": clinic_id,
            "job_id": job_id,
            "resource_type": "professional",
            "name": name,
            "schedule_type": schedule_type,
        }).execute()

    # Insert schedule_type as a clinic-level resource (type="schedule_config") if no professionals
    # Note: schedule_type is attached to each professional row, so if professionals list is empty,
    # insert one row representing the clinic-level schedule type.
    if not professionals:
        db.table("la_resources").insert({
            "clinic_id": clinic_id,
            "job_id": job_id,
            "resource_type": "schedule_config",
            "name": "default",
            "schedule_type": schedule_type,
        }).execute()

    # Insert services
    for svc in services:
        db.table("la_services").insert({
            "clinic_id": clinic_id,
            "job_id": job_id,
            "name": svc["name"],
            "mention_count": svc["mention_count"],
        }).execute()
```

### Pattern 5: Public Entry Point

**What:** A single function callable from `analysis_runner.py` (Phase 9 wires it). Takes conversations + shadow DNA output as inputs. Returns nothing — side effect is DB writes.

**Example:**

```python
# Source: shadow_dna.extract_shadow_dna() and worker.py pipeline patterns (HIGH confidence)
from analyzer.shadow_dna import ShadowDNA
from analyzer.parser import Conversation

def infer_and_persist_resources(
    conversations: list[Conversation],
    clinic_name: str,
    clinic_id: str,
    job_id: str,
    shadow_dna: ShadowDNA,
    db=None,
) -> None:
    """
    Infer resources (professionals, schedule_type) and services from conversations
    and persist as suggestions to la_resources and la_services.

    Args:
        conversations:  list of Conversation objects (full corpus)
        clinic_name:    display name of the clinic
        clinic_id:      UUID from sf_clinics (for la_resources.clinic_id FK)
        job_id:         UUID of the analysis job (for traceability)
        shadow_dna:     ShadowDNA result — local_procedures feeds SVC-01
        db:             Supabase client (defaults to get_db())
    """
    if db is None:
        from db import get_db
        db = get_db()

    # 1. Extract professionals + schedule_type via DSPy
    result = extract_resources(conversations, clinic_name)

    # 2. Build service list from Shadow DNA (already extracted by ShadowDNA LLM call)
    service_names = shadow_dna.local_procedures or []

    # 3. Count service mentions (pure Python)
    services = count_service_mentions(service_names, conversations)

    # 4. Persist
    persist_resources(
        db=db,
        clinic_id=clinic_id,
        job_id=job_id,
        professionals=result.professionals,
        schedule_type=result.schedule_type,
        services=services,
    )
```

### Anti-Patterns to Avoid

- **Running a separate LLM call for services if ShadowDNA is already available:** `ShadowDNA.local_procedures` is extracted in the same LLM pass as `tone_classification`, `greeting_example`, etc. Invoking another LLM call for procedures duplicates cost and may produce contradictory results.
- **Per-conversation LLM calls for professional extraction:** Extract from a sample of 10 conversations at corpus level — same strategy as Shadow DNA. 100 conversations * 1 LLM call each = ~$0.50–$5.00 unnecessary spend.
- **Inserting into `la_resources` with `client_id` instead of `clinic_id`:** The new tables must use `clinic_id UUID REFERENCES sf_clinics(id)`, not `client_id REFERENCES la_clients(id)`. The old `la_clients` concept is v0 (Archive.zip flow). These suggestions belong to the v1.1 Evolution flow.
- **Assuming Shadow DNA is always non-null:** `shadow_dna.local_procedures` can be `[]` if the LLM extraction failed. Always guard with `or []` before counting mentions.
- **Counting service mentions in patient messages:** Only clinic messages confirm what the clinic actually offers. Patient messages may ask about services the clinic does NOT provide. Use `conv.clinic_messages` for frequency counts.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Service/procedure list extraction | Custom regex for dental procedure terms | `ShadowDNA.local_procedures` (already extracted by LLM) | The Shadow DNA LLM pass already identifies local procedures with context. Building a procedure regex would require a curated dental vocabulary and still miss novel terms. |
| Professional title NER | Full NLP pipeline (spaCy, NLTK) | DSPy Signature with targeted prompt + simple regex post-processing | spaCy pt-BR model not in requirements. DSPy already in use. A targeted prompt for "list the professionals mentioned" is sufficient for this use case. |
| Deduplication of professional names | Fuzzy matching library | Lowercase + stripped set comparison | "Dra. Ana" / "dra. ana" / "Dra Ana" are the same — case-insensitive set is sufficient. No need for Levenshtein distance for this scope. |
| Mention counting | Vector similarity search | `str.count()` or `in` operator | Exact substring match is correct here — we are counting brand mentions, not semantic similarity. No embeddings needed for SVC-02. |

**Key insight:** The hardest work (procedure extraction) is already done by the existing Shadow DNA LLM call. Phase 8 primarily adds persistence (two new tables) and a new DSPy call for professionals + schedule_type, reusing the existing corpus-sample infrastructure.

---

## Schema Design

### la_resources (NEW table)

```sql
-- ============================================================
-- MIGRATION: Phase 8 — Resources and Services Inference
-- ============================================================

CREATE TABLE IF NOT EXISTS la_resources (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clinic_id       UUID NOT NULL REFERENCES sf_clinics(id) ON DELETE CASCADE,
    job_id          UUID REFERENCES la_analysis_jobs(id) ON DELETE SET NULL,
    resource_type   TEXT NOT NULL DEFAULT 'professional',
                    -- 'professional': named healthcare professional
                    -- 'schedule_config': clinic-level schedule type record (when no professionals detected)
    name            TEXT NOT NULL,          -- e.g. "Dra. Ana", "Dr. Carlos"
    schedule_type   TEXT NOT NULL DEFAULT 'single',
                    -- 'single' | 'by_professional' | 'by_room'
    confirmed       BOOLEAN NOT NULL DEFAULT FALSE,
                    -- FALSE = LA suggestion; TRUE = admin confirmed (set by Website/admin UI)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_la_resources_clinic_id ON la_resources(clinic_id);
CREATE INDEX IF NOT EXISTS idx_la_resources_job_id ON la_resources(job_id);
ALTER TABLE la_resources ENABLE ROW LEVEL SECURITY;
```

### la_services (NEW table)

```sql
CREATE TABLE IF NOT EXISTS la_services (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clinic_id       UUID NOT NULL REFERENCES sf_clinics(id) ON DELETE CASCADE,
    job_id          UUID REFERENCES la_analysis_jobs(id) ON DELETE SET NULL,
    name            TEXT NOT NULL,          -- e.g. "implante", "clareamento", "ortodontia"
    mention_count   INT NOT NULL DEFAULT 0, -- frequency across clinic messages
    confirmed       BOOLEAN NOT NULL DEFAULT FALSE,
                    -- FALSE = LA suggestion; TRUE = admin confirmed
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_la_services_clinic_id ON la_services(clinic_id);
CREATE INDEX IF NOT EXISTS idx_la_services_job_id ON la_services(job_id);
CREATE INDEX IF NOT EXISTS idx_la_services_mention_count ON la_services(mention_count DESC);
ALTER TABLE la_services ENABLE ROW LEVEL SECURITY;
```

**Design decisions:**
- `clinic_id` references `sf_clinics` (not `la_clients`) — these are v1.1 Evolution flow records.
- `job_id` is nullable (SET NULL on delete) for traceability without blocking cleanup.
- `confirmed` defaults to FALSE — LA suggests, admin confirms, Website creates in `sf_resources`.
- No `UNIQUE` constraint on `(clinic_id, name)` — each analysis run deletes and re-inserts, so duplicates are impossible within a run. The `confirmed=TRUE` rows are managed by a different actor (Website) and are not touched by the LA.
- `schedule_type` on `la_resources` rows: each professional row carries the clinic-level `schedule_type`. This denormalization is intentional — `sf_resources` will eventually have a `schedule_type` per resource.

---

## Common Pitfalls

### Pitfall 1: Confusing la_resources.schedule_type Scope

**What goes wrong:** `schedule_type` is a clinic-level concept (one value per clinic run) but it gets stored on individual professional rows. If a clinic has 3 professionals and `schedule_type = "by_professional"`, all 3 rows will have `schedule_type = "by_professional"`. This is intentional but confusing.

**Why it happens:** `sf_resources` (Sofia's table) likely has `schedule_type` per resource, not per clinic. The LA mirrors this structure.

**How to avoid:** Document clearly in the module docstring that `schedule_type` is inferred once per run at corpus level and repeated on each row. Do not attempt to infer per-professional schedule_type.

**Warning signs:** Tests that expect only one `schedule_type` value per clinic fail because the same value is on multiple rows.

### Pitfall 2: ShadowDNA Not Yet Called When Resources Module Runs

**What goes wrong:** `infer_and_persist_resources()` reads `shadow_dna.local_procedures`. If the call order in `analysis_runner.py` (Phase 9) places resources inference before Shadow DNA extraction, `shadow_dna` will be a default-empty `ShadowDNA()` with `local_procedures = []`. The service list will be empty.

**Why it happens:** Phase 8 creates the module; Phase 9 wires the call order. The dependency on Shadow DNA output is implicit.

**How to avoid:** Document clearly in `resources_inference.py` that `infer_and_persist_resources()` must be called after `extract_shadow_dna()`. In Phase 9, ensure the call order is: Shadow DNA → Resources Inference.

**Warning signs:** `la_services` table is empty after a successful run even for clinics known to offer multiple procedures.

### Pitfall 3: LLM Returns Professional Names as Plain Strings vs. Dicts

**What goes wrong:** DSPy with some models returns the `professionals` list as a list of dicts (e.g., `[{"name": "Dra. Ana"}]`) instead of plain strings. The `_filter_professionals()` function must handle both shapes.

**Why it happens:** The same pattern is documented in `dspy_pipeline.py` for `TopicSignature.topics` — the project already handles this mismatch with `_safe_list()` in multiple places.

**How to avoid:** Use a helper identical to `_safe_list()` from `shadow_dna.py` and additionally extract the string value from dicts if present. Pattern: `item.get("name") or item.get("professional") or str(item)` if `isinstance(item, dict)`.

**Warning signs:** `la_resources` has rows with `name` values like `"{'name': 'Dra. Ana'}"` (stringified dict).

### Pitfall 4: Deleting Confirmed Resources on Re-Analysis

**What goes wrong:** The delete-then-insert strategy removes all rows for `clinic_id` including any where `confirmed = TRUE` that the admin has already verified.

**Why it happens:** A naive `DELETE WHERE clinic_id = X` removes both confirmed and unconfirmed suggestions.

**How to avoid:** Two options: (A) restrict delete to `confirmed = FALSE` only (`DELETE WHERE clinic_id = X AND confirmed = FALSE`), or (B) note that in v1.1 the Website has not yet confirmed anything (go live flow), so this is a non-issue at first deploy. Option A is the safer long-term default. Document this clearly.

**Warning signs:** Admin confirms a resource, re-analysis runs, and the confirmed resource disappears from `la_resources`.

### Pitfall 5: Empty Conversation Sample on First Run

**What goes wrong:** A brand-new clinic that has just completed WhatsApp sync via Evolution API may have very few messages. If the sample passed to `ResourcesSignature` is empty, the LLM returns empty lists and `schedule_type = "single"` by default.

**Why it happens:** `_build_sample()` in `shadow_dna.py` handles this gracefully — it returns an empty string if there are no conversations. The DSPy call may still succeed but produce minimal output.

**How to avoid:** Guard with a minimum conversation count check before invoking the DSPy call. If `len(conversations) == 0`, skip the LLM call and persist empty tables. Log a warning.

**Warning signs:** `la_resources` has one row with `name = "default"` and `schedule_type = "single"` for a clinic with no conversations.

---

## Code Examples

### DSPy Lazy Module Initialization (Mirrors Existing Pattern)

```python
# Source: shadow_dna.py init_shadow_module() pattern (HIGH confidence — same codebase)
_resources_module: Optional[ResourcesModule] = None

def init_resources_module():
    global _resources_module
    _resources_module = ResourcesModule()

# Called from dspy_pipeline.configure_lm() alongside other init_*() calls:
# from analyzer.resources_inference import init_resources_module
# init_resources_module()
```

Note: `init_resources_module()` must be registered in `dspy_pipeline.configure_lm()` alongside the existing `init_shadow_module()`, `init_outcome_module()`, etc. calls. This is a one-line addition to `dspy_pipeline.py`.

### Conversation Sample Reuse

```python
# Source: shadow_dna._build_sample() (HIGH confidence — same codebase)
# Reuse the same _build_sample() helper from shadow_dna.py
from analyzer.shadow_dna import _build_sample

def extract_resources(
    conversations: list,
    clinic_name: str,
) -> "ResourcesResult":
    if not _resources_module:
        return ResourcesResult(error="ResourcesModule not initialized.")
    if not conversations:
        return ResourcesResult(schedule_type="single")

    sample = _build_sample(conversations, max_convs=10)
    # ... call DSPy module ...
```

### Full _safe_list Usage Pattern

```python
# Source: shadow_dna._safe_list() — same codebase pattern (HIGH confidence)
# Already present in shadow_dna.py; import or duplicate in resources_inference.py
def _safe_list(value, default: list) -> list:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        try:
            import ast
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed]
        except Exception:
            pass
        return [v.strip() for v in value.split(",") if v.strip()]
    return default
```

### Supabase Delete-Then-Insert Pattern

```python
# Source: worker.py db.table().insert() and existing schema patterns (HIGH confidence)
# Delete unconfirmed suggestions only (safe for re-runs after admin confirmation)
db.table("la_resources").delete() \
    .eq("clinic_id", clinic_id) \
    .eq("confirmed", False) \
    .execute()

# Batch insert professionals
rows = [
    {
        "clinic_id": clinic_id,
        "job_id": job_id,
        "resource_type": "professional",
        "name": name,
        "schedule_type": schedule_type,
        "confirmed": False,
    }
    for name in professionals
]
if rows:
    db.table("la_resources").insert(rows).execute()
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `local_procedures` extracted but only stored in `la_shadow_dna` | Procedures also persisted in `la_services` with frequency counts | Phase 8 (now) | Admin can now see service list directly without querying `la_shadow_dna.local_procedures` |
| No professional inference | `la_resources` with detected professional names + schedule_type | Phase 8 (now) | Sofia can suggest professional selection at onboarding |
| No `la_resources` or `la_services` tables | Two new tables with `clinic_id` FK to `sf_clinics` | Phase 8 (now) | Requires SQL migration before code runs |

**Deprecated/outdated in this context:**
- Extracting procedures from `la_shadow_dna.local_procedures` directly for Sofia: Phase 8 makes `la_services` the canonical source. `la_shadow_dna.local_procedures` remains as a raw extraction artifact.

---

## Integration with Existing Pipeline

### Where This Fits in the Analysis Flow

The existing pipeline in `worker.py` (Archive.zip flow) and the planned `analysis_runner.py` (Evolution flow) follows this order:

```
1. Ingest conversations
2. Compute metrics (pure Python)
3. DSPy semantic analysis per conversation (sentiment, topics, quality, summary)
4. Outcome detection per conversation
5. Shadow DNA extraction (corpus-level LLM call)
6. Embeddings
7. Build report
8. Training exports
9. Mark job done
```

Phase 8 adds a step **after Shadow DNA extraction (step 5)** and **before embeddings (step 6)**:

```
5b. Resources + Services Inference
    - Call infer_and_persist_resources(conversations, clinic_name, clinic_id, job_id, shadow_dna)
    - Reads: shadow_dna.local_procedures (already extracted)
    - Calls: ResourcesModule (1 DSPy LLM call for professionals + schedule_type)
    - Writes: la_resources, la_services
```

**Phase 8 creates the module. Phase 9 wires the call.** The module is self-contained and testable independently.

### dspy_pipeline.configure_lm() Change

`configure_lm()` in `dspy_pipeline.py` initializes all DSPy modules at startup. Phase 8 adds one line:

```python
# In configure_lm(), after existing init calls:
from analyzer.resources_inference import init_resources_module
init_resources_module()
```

This is a 2-line change to `dspy_pipeline.py` (import + call). It does not modify any existing behavior.

---

## Open Questions

1. **Does `sf_clinics` have a `name` column that equals the WhatsApp clinic display name?**
   - What we know: `sf_clinics.name` exists (used in Phase 7 for the POST /analyze response). `la_clients.sender_name` is the WhatsApp display name in the v0 flow.
   - What's unclear: Whether `sf_clinics.name` matches the WhatsApp display name or is a different canonical name.
   - Recommendation: For Phase 8, accept `clinic_name` as a parameter to `infer_and_persist_resources()`. Phase 9 is responsible for passing the correct value. Phase 8 does not need to resolve this.

2. **Should `la_resources` and `la_services` use `confirmed` boolean or a separate `status` enum?**
   - What we know: STATE.md: "la_resources: LA sugere, admin confirma, Website cria em sf_resources." A boolean `confirmed` is the simplest representation.
   - What's unclear: Whether the Website needs a finer-grained status (e.g., `rejected`).
   - Recommendation: Use `confirmed BOOLEAN DEFAULT FALSE` for go live. A `rejected` state is v2+ scope. Simple boolean covers the v1.1 need.

3. **What happens when multiple analysis runs occur for the same clinic?**
   - What we know: Each run should replace previous suggestions. DELETE WHERE `clinic_id = X AND confirmed = FALSE` handles this safely.
   - What's unclear: Whether `la_services` / `la_resources` should retain history by job.
   - Recommendation: Delete-then-insert per run (unconfirmed only). History is available through `la_analysis_jobs` and `la_shadow_dna`. No need to keep old `la_resources` suggestions.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (already in use — `tests/` directory exists with 5 test files) |
| Config file | None detected — pytest runs with default settings |
| Quick run command | `pytest tests/test_resources_inference.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RES-01 | `extract_resources()` returns non-empty `professionals` list when conversations mention "Dra. Ana" | unit (mock DSPy module) | `pytest tests/test_resources_inference.py::TestExtractResources::test_professionals_extracted -x` | Wave 0 |
| RES-01 | `la_resources` has a row with `name = "Dra. Ana"` and `clinic_id` after persist call | unit (mock db) | `pytest tests/test_resources_inference.py::TestPersistResources::test_professional_inserted -x` | Wave 0 |
| RES-02 | `extract_resources()` returns `schedule_type = "by_professional"` when multiple professionals are mentioned | unit (mock DSPy module) | `pytest tests/test_resources_inference.py::TestExtractResources::test_schedule_type_by_professional -x` | Wave 0 |
| RES-02 | `extract_resources()` returns `schedule_type = "single"` when no professionals are detected | unit (mock DSPy module) | `pytest tests/test_resources_inference.py::TestExtractResources::test_schedule_type_single -x` | Wave 0 |
| SVC-01 | `count_service_mentions()` returns list with "implante" when shadow_dna.local_procedures contains it | unit (no mock — pure Python) | `pytest tests/test_resources_inference.py::TestCountServiceMentions::test_service_in_clinic_messages -x` | Wave 0 |
| SVC-01 | `la_services` has a row with `name = "implante"` after persist call | unit (mock db) | `pytest tests/test_resources_inference.py::TestPersistResources::test_service_inserted -x` | Wave 0 |
| SVC-02 | `count_service_mentions()` returns correct `mention_count` for each service | unit (no mock — pure Python) | `pytest tests/test_resources_inference.py::TestCountServiceMentions::test_mention_count_accuracy -x` | Wave 0 |
| SVC-02 | Services are sorted by `mention_count` descending | unit (no mock — pure Python) | `pytest tests/test_resources_inference.py::TestCountServiceMentions::test_sorted_by_frequency -x` | Wave 0 |
| RES-01+02 | `infer_and_persist_resources()` calls delete for previous unconfirmed records before inserting | unit (mock db) | `pytest tests/test_resources_inference.py::TestInferAndPersist::test_delete_before_insert -x` | Wave 0 |
| RES-01+SVC-01 | Empty conversations list results in no DB writes and no error | unit (mock db) | `pytest tests/test_resources_inference.py::TestInferAndPersist::test_empty_conversations_no_crash -x` | Wave 0 |

All tests use mocked `get_db()` and mocked DSPy `Predict` — no live Supabase or LLM API keys required.

### Sampling Rate

- **Per task commit:** `pytest tests/test_resources_inference.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_resources_inference.py` — all tests listed above (new file, does not exist yet)
- [ ] SQL migration block in `supabase/schema.sql` — `la_resources` and `la_services` table definitions
- [ ] `analyzer/resources_inference.py` — new module (does not exist yet)
- [ ] One-line addition to `dspy_pipeline.configure_lm()` — `init_resources_module()` call

*(No new framework install needed — pytest, unittest.mock, dspy all already installed)*

---

## Sources

### Primary (HIGH confidence)

- `analyzer/shadow_dna.py` (codebase) — `ShadowDNASignature`, `_build_sample()`, `_safe_list()`, `_compute_quantitative()` — all patterns directly reused in Phase 8
- `analyzer/dspy_pipeline.py` (codebase) — `configure_lm()` initialization pattern, lazy module globals, `_safe_list()` for DSPy output handling
- `supabase/schema.sql` (codebase) — existing table structure, FK conventions (`la_` prefix, `clinic_id` pattern from Phase 7 migration, RLS enablement)
- `worker.py` (codebase) — `_update_job()`, db.table().insert() patterns, pipeline step ordering
- `analyzer/analysis_runner.py` (codebase) — Phase 9 integration point (stub with clear comment marking where pipeline call goes)
- `.planning/STATE.md` (decisions) — "la_resources: LA sugere, admin confirma, Website cria em sf_resources"; "la_services: LA infere procedimentos com frequencia de mencao — sugestao para admin"
- `.planning/REQUIREMENTS.md` — RES-01, RES-02, SVC-01, SVC-02 full requirement text

### Secondary (MEDIUM confidence)

- `analyzer/outcome_detection.py` (codebase) — DSPy Signature pattern for classification (schedule_type is a 3-class classification — same pattern as OutcomeSignature's outcome output field)
- `tests/test_evolution_ingestor.py` (codebase) — confirmed `unittest.mock` + `MagicMock` pattern for DB mocking in this project's test style

### Tertiary (LOW confidence — flag for validation)

- `sf_clinics` schema beyond `id`, `name`, `evolution_instance_id`: confirmed `confirmed` boolean column behavior aligns with admin flow described in STATE.md but `sf_clinics` schema not directly read.
- `schedule_type` values (`single` / `by_professional` / `by_room`): inferred from requirement description and common scheduling patterns. Not verified against Sofia's `sf_resources` schema. Verify before Phase 9 wires the contract.

---

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — zero new dependencies; all patterns reuse existing codebase (dspy, supabase-py, stdlib)
- Schema design: HIGH — follows established `la_*` table conventions; FK to `sf_clinics` validated against Phase 7 migration
- Architecture (DSPy module): HIGH — mirrors ShadowDNAModule exactly; corpus-level call is proven in this codebase
- Architecture (service counting): HIGH — pure Python, same pattern as `_compute_quantitative()` in shadow_dna.py
- Integration point: HIGH — analysis_runner.py stub comment explicitly marks where Phase 9 wires the call
- Pitfalls: HIGH — derived from direct code analysis of existing shadow_dna.py and dspy_pipeline.py
- `schedule_type` enum values: MEDIUM — inferred from requirement text; verify against Sofia sf_resources schema before hardcoding

**Research date:** 2026-03-16
**Valid until:** 2026-04-16 (stable stack; 30-day window reasonable)
