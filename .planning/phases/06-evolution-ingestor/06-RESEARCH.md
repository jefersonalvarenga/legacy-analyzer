# Phase 6: Evolution Ingestor — Research

**Researched:** 2026-03-16
**Domain:** Evolution API Supabase Message table → Python Conversation/Message adapter
**Confidence:** MEDIUM-HIGH (schema verified from official Prisma source; sf_clinics linkage inferred from PROJECT.md and MEMORY context)

---

## Summary

Phase 6 creates `analyzer/evolution_ingestor.py` — a read-only adapter that queries the Evolution API's `Message` table in Supabase and produces a `list[Conversation]` identical in type to what `parser.py` currently outputs from Archive.zip files. The existing pipeline (metrics, DSPy, outcome detection, Shadow DNA, blueprint) must accept the output with zero modification.

The core technical challenge is the impedance mismatch between Evolution's schema (a flat `Message` table with JSONB `key` and `message` fields, `instanceId` FK, `messageTimestamp` as Unix int) and the internal `Conversation`/`Message` dataclasses (grouped by phone, `sender_type` derived, `sent_at` as Python datetime). The adapter must handle this mapping faithfully while staying strictly read-only.

The second challenge is the isolation contract: the ingestor must filter exclusively by the `instanceId` associated with the given `clinic_id` via the `sf_clinics` table (column `evolution_instance_id`). It must never accept messages belonging to a different instance, even if they share a phone number.

**Primary recommendation:** Implement a thin synchronous adapter using the existing `supabase-py` client (already in requirements.txt). No new HTTP client, no new libraries. Query `Message WHERE instanceId = <instance_from_sf_clinics>`, group rows by `remoteJid` extracted from `key` JSONB, and reconstruct `Conversation`/`Message` objects. All transformation logic lives in `analyzer/evolution_ingestor.py`. No writes to any table.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ING-01 | LA le mensagens da tabela `Message` do Evolution WHERE instanceId = instancia do onboarding | `Message` table confirmed in Prisma schema with `instanceId` FK and `@@index([instanceId])`. supabase-py `.select().eq("instanceId", ...)` covers this filter. |
| ING-02 | Adapter mapeia formato `Message` → objetos internos `Conversation`/`Message` | `Message.key` (JSONB) contains `remoteJid` and `fromMe`. `message.conversation` or `message.extendedTextMessage.text` holds body. `messageTimestamp` (Unix int) maps to `datetime`. `pushName` maps to sender display name. Full mapping documented in Architecture Patterns. |
| ING-03 | Filtra conversas por clinic_id (via instancia associada ao onboarding da clinica, nunca por outra clinica) | Flow: `clinic_id` → lookup `sf_clinics.evolution_instance_id` (instance name string) → lookup Evolution `Instance.id` WHERE `name = instance_name` → filter `Message WHERE instanceId = instance_uuid`. Isolation guaranteed at query level. |
</phase_requirements>

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| supabase-py | 2.13.0 (already installed) | Query `Message` and `Instance` tables; lookup `sf_clinics` | Already the project's Supabase client. `get_db()` singleton in `db.py` provides it ready. No new library needed. |
| pydantic | 2.10.6 (already installed) | Typed intermediate model for raw Evolution Message rows before mapping | Already the project's validation library. Prevents silent field mismatches on schema drift. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| python-dateutil | 2.9.0 (already installed) | Convert Unix timestamp int to timezone-aware datetime if needed | Use `datetime.fromtimestamp(messageTimestamp)` (stdlib). python-dateutil needed only if timezone-aware conversion required — use stdlib first. |
| pytest | (already installed) | Unit tests for the adapter | Pure unit tests — no live Supabase needed; mock `get_db()`. |
| pytest-asyncio | 0.23.8 (dev only) | Async test support | Only if ingestor is made async. Prefer sync for Phase 6 to match existing `parser.py` API. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| supabase-py (sync `.execute()`) | httpx.AsyncClient direct Supabase REST | Async adds complexity for no gain in Phase 6. supabase-py sync is already used everywhere. Keep sync. |
| supabase-py | psycopg2 direct SQL | Direct SQL is faster for bulk reads but breaks the project's abstraction. supabase-py is sufficient. |
| Pydantic intermediate model | Dict access | Pydantic surfaces column name changes at parse time, not silently downstream. Prefer Pydantic. |

**Installation:**

```bash
# No new runtime dependencies. All needed libraries are already in requirements.txt.
# Dev only if async tests needed:
pip install pytest-asyncio==0.23.8
```

---

## Architecture Patterns

### Recommended Project Structure

```
analyzer/
├── parser.py               # existing — Archive.zip source (unchanged)
├── evolution_ingestor.py   # NEW — Evolution Supabase source
└── ...
```

The ingestor is a new file at `analyzer/evolution_ingestor.py`. It does NOT modify `parser.py`. It produces the same `list[Conversation]` type. The caller (Phase 9 pipeline) will call either `parse_archive()` or `ingest_from_evolution()` — same return type, different source.

### Pattern 1: clinic_id → instanceId Resolution

**What:** The ingestor receives `clinic_id` (UUID from `sf_clinics`). It must look up the Evolution `instanceId` in two hops:
1. Query `sf_clinics WHERE id = clinic_id` → get `evolution_instance_id` (this is the Evolution instance **name**, e.g., `"sgen-prod"`)
2. Query Evolution `Instance WHERE name = evolution_instance_id` → get `Instance.id` (UUID, the FK used in `Message.instanceId`)

**When to use:** Always. The filter `Message WHERE instanceId = instance_uuid` is the isolation guarantee.

**Example:**

```python
# Source: PROJECT.md context + Prisma schema (HIGH confidence)
def _resolve_instance_id(db, clinic_id: str) -> str:
    """Resolve clinic_id → Evolution instanceId (UUID)."""
    clinic_row = (
        db.table("sf_clinics")
        .select("evolution_instance_id")
        .eq("id", clinic_id)
        .single()
        .execute()
    )
    if not clinic_row.data:
        raise ValueError(f"clinic_id {clinic_id!r} not found in sf_clinics")

    instance_name = clinic_row.data["evolution_instance_id"]
    if not instance_name:
        raise ValueError(f"No Evolution instance linked to clinic_id {clinic_id!r}")

    # Instance table is Evolution's own table (not la_* prefixed)
    instance_row = (
        db.table("Instance")
        .select("id")
        .eq("name", instance_name)
        .single()
        .execute()
    )
    if not instance_row.data:
        raise ValueError(f"Evolution instance {instance_name!r} not found in Instance table")

    return instance_row.data["id"]
```

### Pattern 2: Message Row → Internal Message/Conversation

**What:** Fetches all `Message` rows for an `instanceId`, extracts fields from JSONB `key` and `message`, groups by `remoteJid` to form Conversation objects.

**Key schema facts (HIGH confidence — verified from Prisma postgresql-schema.prisma):**

| Column | Type | Content |
|--------|------|---------|
| `id` | String (CUID) | Row PK |
| `key` | JSONB | `{"remoteJid": "55119...@s.whatsapp.net", "fromMe": true/false, "id": "MSG_HEX_ID"}` |
| `pushName` | String? | Sender display name (patient's name as they appear in WhatsApp) |
| `message` | JSONB | `{"conversation": "text"}` or `{"extendedTextMessage": {"text": "text"}}` or media types |
| `messageType` | String | `"conversation"`, `"extendedTextMessage"`, `"imageMessage"`, `"audioMessage"`, etc. |
| `messageTimestamp` | Int | Unix epoch seconds |
| `instanceId` | String | FK to `Instance.id` |
| `participant` | String? | In group chats — sender JID; null for individual chats |
| `source` | Enum | `"android"`, `"ios"`, `"web"` — useful for data quality, not required for parsing |

**remoteJid format:**
- Individual: `"5511912345678@s.whatsapp.net"` — strip `@s.whatsapp.net` to get phone
- Group: `"120363XXXXXXX@g.us"` — group chats, skip unless needed
- Business: may include `@lid` suffix on newer WhatsApp versions

**fromMe semantics:**
- `fromMe: true` → message sent by the clinic's WhatsApp number → `sender_type = "clinic"`
- `fromMe: false` → message received from patient → `sender_type = "patient"`

**Example:**

```python
# Source: Prisma schema (verified) + WhatsApp JID format (HIGH confidence)
from datetime import datetime
from analyzer.parser import Conversation, Message


def _extract_body(message_json: dict, message_type: str) -> str:
    """Extract text content from Evolution Message.message JSONB."""
    if not message_json:
        return ""
    # Most common: plain text
    if "conversation" in message_json:
        return message_json["conversation"]
    # Extended text (links, mentions, etc.)
    ext = message_json.get("extendedTextMessage", {})
    if ext.get("text"):
        return ext["text"]
    # Media messages — return placeholder consistent with parser.py SYSTEM_PATTERNS
    return f"[{message_type}]"


def _build_sender_type(from_me: bool) -> str:
    return "clinic" if from_me else "patient"


def _extract_phone(remote_jid: str) -> str:
    """Strip WhatsApp JID suffix to get phone number."""
    # "5511912345678@s.whatsapp.net" → "5511912345678"
    return remote_jid.split("@")[0]


def _is_group_jid(remote_jid: str) -> bool:
    return remote_jid.endswith("@g.us")
```

### Pattern 3: Grouping Rows into Conversations

**What:** Evolution stores messages in a flat table. Each row is one message. `remoteJid` in `key` JSONB is the conversation identifier (equivalent to a `.zip` filename in the Archive.zip flow).

**Example:**

```python
# Source: codebase analysis + Prisma schema
from collections import defaultdict

def _group_messages_by_conversation(rows: list[dict], clinic_sender_name: str) -> list[Conversation]:
    groups: dict[str, list[Message]] = defaultdict(list)
    phones: dict[str, str] = {}

    for row in rows:
        key = row.get("key") or {}
        remote_jid = key.get("remoteJid", "")

        # Skip group chats (not WhatsApp individual conversations)
        if _is_group_jid(remote_jid):
            continue

        from_me = key.get("fromMe", False)
        push_name = row.get("pushName") or ""
        message_json = row.get("message") or {}
        message_type = row.get("messageType") or "unknown"
        timestamp = row.get("messageTimestamp") or 0

        body = _extract_body(message_json, message_type)
        sender_type = _build_sender_type(from_me)

        # Determine sender display name
        if sender_type == "clinic":
            sender = clinic_sender_name
        else:
            sender = push_name or _extract_phone(remote_jid)

        msg = Message(
            sent_at=datetime.fromtimestamp(timestamp),
            sender=sender,
            sender_type=sender_type,
            content=body,
            raw_line="",  # no raw line in Evolution source
        )
        groups[remote_jid].append(msg)
        phones[remote_jid] = _extract_phone(remote_jid)

    conversations = []
    for remote_jid, messages in groups.items():
        # Sort by timestamp (Evolution rows may not be ordered)
        messages.sort(key=lambda m: m.sent_at)
        conv = Conversation(
            source_filename=remote_jid,  # used as identifier, not a filename
            phone=phones[remote_jid],
            messages=messages,
        )
        conversations.append(conv)

    return conversations
```

### Pattern 4: Public Ingestor Function

**What:** Top-level entry point matching the shape of `parse_archive()`.

**Example:**

```python
# Mirrors parse_archive() signature from analyzer/parser.py
def ingest_from_evolution(
    clinic_id: str,
    clinic_sender_name: str,
    on_progress: Optional[callable] = None,
) -> list[Conversation]:
    """
    Fetch conversations for a clinic from the Evolution API Supabase table.

    Args:
        clinic_id:          UUID of the clinic in sf_clinics
        clinic_sender_name: Display name the clinic uses on WhatsApp (for sender classification)
        on_progress:        Optional callback(current: int, total: int, label: str)

    Returns:
        List of Conversation objects — same type as parse_archive() output
    """
    db = get_db()
    instance_id = _resolve_instance_id(db, clinic_id)

    result = (
        db.table("Message")
        .select("id, key, pushName, message, messageType, messageTimestamp, participant, source")
        .eq("instanceId", instance_id)
        .execute()
    )
    rows = result.data or []
    return _group_messages_by_conversation(rows, clinic_sender_name)
```

### Anti-Patterns to Avoid

- **Writing to Evolution tables:** Never INSERT, UPDATE, or DELETE any row in `Message`, `Chat`, `Contact`, `Instance`, or any other Evolution-owned table. The ingestor is strictly a reader.
- **Filtering by phone number instead of instanceId:** A patient phone may appear in multiple instances (e.g., different clinics). The only correct isolation key is `instanceId`. Never filter by `remoteJid` across instances.
- **Assuming `message.conversation` always exists:** `messageType` can be `"imageMessage"`, `"audioMessage"`, `"stickerMessage"`, etc. Always guard with `.get()` before accessing nested fields.
- **Ignoring message ordering:** Evolution `Message` rows are not guaranteed to be ordered by `messageTimestamp`. Always sort within each conversation group before constructing `Conversation.messages`.
- **Trusting `pushName` for clinic detection:** `pushName` is the remote contact's name as seen by the WhatsApp account. For `fromMe: true` messages, `pushName` may be null or irrelevant. Use `fromMe` field exclusively for clinic/patient classification.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Supabase query client | Custom HTTP wrapper | `supabase-py` `get_db()` already in `db.py` | Zero new dependencies; handles auth, connection, error handling |
| JSON field parsing | Custom JSONB decoder | Standard `dict.get()` with defaults | Prisma/Supabase returns Python dicts; no additional deserialization needed |
| Conversation grouping | Graph-based chat detection | `defaultdict(list)` keyed on `remoteJid` | Each `remoteJid` IS the conversation identifier — no inference needed |
| Timezone handling | Custom tz library | `datetime.fromtimestamp()` (stdlib) | `messageTimestamp` is Unix epoch; stdlib handles it correctly |

**Key insight:** The Evolution Message table already groups messages by conversation via `remoteJid` — there is no need for heuristic conversation detection. The mapping is deterministic.

---

## Common Pitfalls

### Pitfall 1: sf_clinics Column Name for Instance

**What goes wrong:** `sf_clinics.evolution_instance_id` holds the Evolution instance **name** (a string like `"sgen-prod"`), not the UUID `Instance.id`. If you use this value directly as `instanceId` in a `Message` query, you'll get zero rows.

**Why it happens:** Evolution's `Message.instanceId` is a FK to `Instance.id` (CUID string like `clxyz123`), not to `Instance.name`.

**How to avoid:** Always do the two-hop lookup: `sf_clinics.evolution_instance_id` → `Instance WHERE name = X` → `Instance.id`.

**Warning signs:** `Message` query returns 0 rows for a clinic known to have messages.

### Pitfall 2: JSONB key Field Access

**What goes wrong:** The `key` column is JSONB. supabase-py returns it as a Python dict. Accessing `row["key"]["remoteJid"]` will raise `KeyError` if `key` is null or if a message has an unusual structure.

**Why it happens:** Some system/status messages in Evolution may have a different key structure (e.g., no `remoteJid`).

**How to avoid:** Always use `.get()` with a default. Rows where `key.get("remoteJid")` is falsy should be skipped.

**Warning signs:** `KeyError: 'remoteJid'` or `TypeError: 'NoneType' object is not subscriptable` during row processing.

### Pitfall 3: Large Message Sets Without Pagination

**What goes wrong:** A clinic with 10,000+ messages will return a massive result set from supabase-py's default `.execute()` call, potentially timing out or consuming excessive memory.

**Why it happens:** supabase-py does not auto-paginate. The default Supabase PostgREST page size is 1,000 rows.

**How to avoid:** Use `.range(from, to)` to paginate or add a date filter (e.g., last 90 days) as a practical limit. For Phase 6 (single clinic, go live scope), a date filter is sufficient. Full pagination is a v2 concern.

**Warning signs:** Response truncated at 1,000 rows when clinic has more messages; or timeout on very large datasets.

### Pitfall 4: Group Chat Messages Contaminating Individual Conversations

**What goes wrong:** Evolution stores group messages in the same `Message` table. `remoteJid` ends with `@g.us` for groups. If not filtered, group messages will create spurious "conversations" with group JIDs as phone numbers, confusing metrics.

**Why it happens:** The ingestor has no concept of "individual vs group" unless it inspects the JID suffix.

**How to avoid:** Skip rows where `remoteJid.endswith("@g.us")`.

**Warning signs:** Conversations with `phone` values that end in `@g.us` or look like long numeric group IDs.

### Pitfall 5: Missing Sender Name for Clinic Messages

**What goes wrong:** For `fromMe: true` messages, `pushName` is often null or contains the name of the patient (the recipient), not the clinic. Using `pushName` to determine the clinic sender name will misclassify clinic messages.

**Why it happens:** `pushName` records the WhatsApp contact name of the **remote party**, not the sender.

**How to avoid:** For `fromMe: true`, always use the `clinic_sender_name` parameter passed to the ingestor (same pattern as `parser.py`'s `clinic_sender_name` argument). Never derive the clinic name from `pushName`.

---

## Code Examples

### Verified Message Key Structure

```python
# Source: EvolutionAPI/evolution-api GitHub issues + Prisma schema (MEDIUM-HIGH confidence)
# Example row["key"] for an individual inbound message:
key = {
    "remoteJid": "5511912345678@s.whatsapp.net",
    "fromMe": False,
    "id": "3EB073ADB8DD86D47345ABD99D2213513E739793"
}

# Example row["key"] for an outbound message (clinic sent):
key = {
    "remoteJid": "5511912345678@s.whatsapp.net",
    "fromMe": True,
    "id": "BAE5A4AB2B57F4C1"
}
```

### Verified message Body Extraction

```python
# Source: EvolutionAPI webhook payload documentation (MEDIUM confidence)
# Plain text message
message = {"conversation": "Olá, bom dia! Quero agendar uma consulta."}

# Extended text (with formatting, links)
message = {
    "extendedTextMessage": {
        "text": "Clique aqui: https://...",
        "canonicalUrl": "https://...",
    }
}

# Audio (media — no text body)
message = {
    "audioMessage": {
        "url": "https://...",
        "seconds": 12
    }
}
# → body should be "[audioMessage]" to match parser.py SYSTEM_PATTERNS convention
```

### Supabase Query Pattern (Read-Only)

```python
# Source: supabase-py docs + existing codebase patterns in run_local.py (HIGH confidence)
result = (
    db.table("Message")
    .select("id, key, pushName, message, messageType, messageTimestamp")
    .eq("instanceId", instance_uuid)
    .order("messageTimestamp", desc=False)
    .execute()
)
rows = result.data or []
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| parse_archive(Archive.zip) | ingest_from_evolution(clinic_id) | Phase 6 (now) | No more file upload; live data from Supabase |
| Source identified via .zip filename | Source identified via remoteJid JID | Phase 6 (now) | Phone extraction changes from filename regex to JID parsing |
| sender_type from clinic name string match | sender_type from fromMe boolean | Phase 6 (now) | More reliable; no string matching needed |

**Deprecated/outdated in this phase:**
- Archive.zip as message source: remains unchanged for `run_local.py` compatibility (v0 use case). Phase 6 adds an alternative, not a replacement.

---

## Open Questions

1. **sf_clinics column name for Evolution instance**
   - What we know: PROJECT.md references `evolution_instance_id` as the column linking a clinic to its Evolution instance. The N8N webhook uses `evolution_instance_id` for the `UPDATE sf_clinics` flow.
   - What's unclear: The exact column name (`evolution_instance_id`? `instance_name`? `instance_id`?) in the live `sf_clinics` table has not been verified against the Sofia database schema. It may differ.
   - Recommendation: Before coding `_resolve_instance_id()`, verify the actual column name by running `SELECT column_name FROM information_schema.columns WHERE table_name = 'sf_clinics'` against the live Supabase or reading the Sofia schema migration file.

2. **Evolution Instance table name casing**
   - What we know: Prisma schema defines `model Instance` which maps to a PostgreSQL table named `"Instance"` (capital I, quoted) or `instance` depending on Prisma's `@map` or default behavior.
   - What's unclear: supabase-py `.table("Instance")` may need to match exact PostgreSQL table name. Prisma with PostgreSQL typically uses the model name as table name with default quoting.
   - Recommendation: Confirm table name via `SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename ILIKE 'instance'` before implementation.

3. **Message pagination threshold**
   - What we know: Supabase PostgREST default page size is 1,000 rows. A clinic with active WhatsApp usage for 6+ months may have 5,000-50,000+ messages.
   - What's unclear: Whether the first go-live clinic (sgen) has a message volume requiring pagination.
   - Recommendation: Add a configurable `days_back` parameter (default: 90) to limit the query to recent messages. This caps volume for Phase 6 without blocking Phase 9 from expanding the range.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (already in use — `tests/` directory exists) |
| Config file | None detected — pytest runs with default settings |
| Quick run command | `pytest tests/test_evolution_ingestor.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ING-01 | ingest_from_evolution calls Message table with correct instanceId filter | unit (mock db) | `pytest tests/test_evolution_ingestor.py::test_queries_by_instance_id -x` | Wave 0 |
| ING-02 | Message row with fromMe=true maps to sender_type="clinic"; fromMe=false to "patient" | unit | `pytest tests/test_evolution_ingestor.py::test_sender_type_mapping -x` | Wave 0 |
| ING-02 | messageTimestamp (Unix int) maps to correct Python datetime | unit | `pytest tests/test_evolution_ingestor.py::test_timestamp_conversion -x` | Wave 0 |
| ING-02 | message.conversation body extracted correctly | unit | `pytest tests/test_evolution_ingestor.py::test_body_extraction -x` | Wave 0 |
| ING-02 | Output Conversation/Message objects are same type as parser.py output | unit | `pytest tests/test_evolution_ingestor.py::test_output_type_compatibility -x` | Wave 0 |
| ING-03 | Group JIDs (@g.us) are excluded from output | unit | `pytest tests/test_evolution_ingestor.py::test_group_jid_excluded -x` | Wave 0 |
| ING-03 | Messages with instanceId not matching clinic are never returned | unit | `pytest tests/test_evolution_ingestor.py::test_isolation_by_instance_id -x` | Wave 0 |
| ING-01+03 | _resolve_instance_id raises ValueError for unknown clinic_id | unit | `pytest tests/test_evolution_ingestor.py::test_resolve_invalid_clinic_id -x` | Wave 0 |

All tests use mocked `supabase-py` client — no live Supabase required.

### Sampling Rate

- **Per task commit:** `pytest tests/test_evolution_ingestor.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_evolution_ingestor.py` — all tests listed above (new file, does not exist yet)
- [ ] No framework install needed — pytest already used in `tests/`
- [ ] No conftest additions needed for Phase 6 unit tests (all mocked, no LLM or Supabase required)

---

## Sources

### Primary (HIGH confidence)

- `EvolutionAPI/evolution-api` GitHub Prisma schema (`prisma/postgresql-schema.prisma`) — Message model fields, types, constraints, instanceId FK
- `analyzer/parser.py` (codebase) — `Conversation` and `Message` dataclass definitions, exact field names and types
- `supabase/schema.sql` (codebase) — existing `la_*` tables, confirms `get_db()` pattern
- `db.py` (codebase) — `get_db()` singleton, supabase-py client usage
- `run_local.py` (codebase) — existing supabase-py query patterns (`.table().select().eq().execute()`)
- `.planning/PROJECT.md` — `evolution_instance_id` in `sf_clinics`, N8N flow, read-only constraint

### Secondary (MEDIUM confidence)

- EvolutionAPI GitHub issues (#1916, #2279, #2080) — `remoteJid` format, `fromMe` semantics, `pushName` behavior verified across multiple issues
- Evolution API webhook payload documentation (via WebSearch) — `message.conversation` and `message.extendedTextMessage.text` body structure
- `.planning/research/STACK.md` (codebase, 2026-03-13) — prior stack research confirming supabase-py as the correct client, no new HTTP dependencies needed

### Tertiary (LOW confidence — flag for validation)

- `sf_clinics.evolution_instance_id` exact column name: inferred from PROJECT.md and N8N flow description. Not verified against live Sofia schema migration. **Must confirm before coding `_resolve_instance_id()`.**
- Evolution `Instance` table PostgreSQL name casing: inferred from Prisma model name. Must confirm actual table name in shared Supabase.

---

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — supabase-py already installed and in use; no new dependencies confirmed
- Architecture (Message schema): HIGH — Prisma schema fetched directly from EvolutionAPI GitHub repository
- Architecture (sf_clinics linkage): MEDIUM — column name inferred from project docs, not verified against live schema
- Pitfalls: HIGH — derived from code + confirmed schema structure
- Test plan: HIGH — mirrors existing test patterns in `tests/`

**Research date:** 2026-03-16
**Valid until:** 2026-04-16 (Evolution API Prisma schema is stable; 30-day window reasonable)
