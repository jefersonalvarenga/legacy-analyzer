# Stack Research

**Domain:** Evolution API + N8N + WhatsApp sync integration (Python/Supabase pipeline)
**Researched:** 2026-03-13
**Confidence:** MEDIUM — Evolution API specifics from training knowledge + codebase inference; N8N from training knowledge. WebSearch/WebFetch unavailable for live verification.

---

## Context: What Already Exists (Do Not Re-add)

The existing stack is fully operational. Research scope is ONLY the delta needed for:

1. Python client that calls Evolution API REST to fetch conversations and messages
2. N8N workflow that receives Evolution webhook and writes a sync flag to Supabase
3. Polling trigger in FastAPI that reads the flag and dispatches analysis jobs
4. Schema additions to Supabase (flag table, no new DB)

---

## Recommended Stack — New Additions Only

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| httpx | 0.28.1 (already installed) | HTTP client for Evolution API REST calls | Already in requirements.txt. Async-native, supports `AsyncClient` with connection pooling. Replaces the synchronous `requests` used in notifier.py for the new async code path. |
| Evolution API | v2.x (self-hosted or cloud) | WhatsApp Business automation — REST API for listing/reading conversations and messages | The project's chosen WhatsApp infrastructure. Auth via `apikey` header. Endpoints: `GET /chat/findChats/{instance}`, `GET /chat/findMessages/{instance}`. No SDK needed — plain REST. |
| N8N | 1.x (self-hosted at n8n.easyscale.co) | Workflow orchestrator — receives Evolution webhook, writes sync flag to Supabase | Already referenced in codebase (`NOTIFY_WEBHOOK_URL`). The pattern (N8N webhook → Supabase action) is proven in `notifier.py`. No new deployment needed. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pydantic | 2.10.6 (already installed) | Typed data models for Evolution API response payloads | Model `EvolutionMessage`, `EvolutionChat` as Pydantic models so the adapter layer mirrors the `Message`/`Conversation` dataclasses in `parser.py`. Always. |
| python-dotenv | 1.0.1 (already installed) | Load `EVOLUTION_API_URL`, `EVOLUTION_API_KEY`, `EVOLUTION_INSTANCE` from `.env` | Already wired via `pydantic-settings`. Add new fields to `config.py` — no new library. |
| supabase | 2.13.0 (already installed) | Read/write `la_sync_queue` flag table | Already used everywhere. The N8N workflow calls Supabase REST directly (no Python needed on that path). |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| pytest + pytest-asyncio | Test the async Evolution API client | `pytest-asyncio` is the standard async test runner. Pin to `0.23.x` for compatibility with pytest 8.x. Add to dev dependencies only. |
| respx | Mock `httpx` calls in tests | `respx` is the canonical mock library for `httpx.AsyncClient`. Use instead of `responses` (which targets `requests`). Add to dev dependencies only. |

---

## Installation

```bash
# No new runtime dependencies — all required libraries are already in requirements.txt.
# httpx, pydantic, pydantic-settings, supabase, python-dotenv are all present.

# Dev dependencies only:
pip install pytest-asyncio==0.23.8 respx==0.21.1
```

---

## New Config Fields (config.py additions)

```python
# Evolution API
evolution_api_url: str = Field(..., env="EVOLUTION_API_URL")
# Example: https://evolution.easyscale.co
evolution_api_key: str = Field(..., env="EVOLUTION_API_KEY")
evolution_instance: str = Field(..., env="EVOLUTION_INSTANCE")
# Example: sgen-prod
```

---

## New Schema (Supabase migration)

A single control table is all that's needed. No new databases, no new services.

```sql
-- la_sync_queue: one row per clinic, updated by N8N, polled by FastAPI worker
CREATE TABLE IF NOT EXISTS la_sync_queue (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id       UUID NOT NULL REFERENCES la_clients(id) ON DELETE CASCADE,
    status          TEXT NOT NULL DEFAULT 'idle'
                    CHECK (status IN ('idle', 'pending', 'processing', 'done', 'error')),
    triggered_by    TEXT,           -- 'n8n_webhook' | 'manual'
    evolution_instance TEXT,        -- which Evolution instance to pull from
    last_synced_at  TIMESTAMPTZ,    -- last successful sync
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_la_sync_queue_client ON la_sync_queue(client_id);
```

N8N writes `status = 'pending'` via Supabase REST. The FastAPI worker polls `WHERE status = 'pending'` and transitions to `'processing'` → `'done'`.

---

## Integration Architecture

```
Evolution API webhook → N8N
  N8N workflow:
    1. Receive POST /webhook/la-sync-trigger
    2. Upsert la_sync_queue SET status='pending' WHERE client_id=X

  FastAPI worker (worker.py):
    1. Poll la_sync_queue WHERE status='pending' (every worker_poll_interval seconds)
    2. Set status='processing'
    3. Call EvolutionClient.fetch_conversations(instance, client_id)
       → GET /chat/findChats/{instance}    (list conversations)
       → GET /chat/findMessages/{instance} (messages per chat)
    4. Normalize Evolution messages → Conversation/Message dataclasses (same as parser.py output)
    5. Feed into existing pipeline (same code path as Archive.zip flow)
    6. Set status='done'
```

The adapter (`analyzer/evolution_client.py`) must output the same `list[Conversation]` type that `parser.py` outputs. The downstream pipeline (`worker.py`, `dspy_pipeline.py`, `report_builder.py`) never needs to know the source.

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| httpx.AsyncClient | requests (already in notifier.py) | Use requests only if the Evolution client is synchronous and called from a non-async context. Prefer httpx everywhere new. |
| Custom thin client (analyzer/evolution_client.py) | python-evolution-api SDK (if one exists) | A community SDK would be acceptable, but Evolution API v2 REST is simple enough that a thin httpx wrapper is more maintainable and has zero extra dependencies. |
| la_sync_queue table | Supabase Realtime subscriptions | Realtime is appropriate if the frontend needs push notification. For worker-side polling (which already exists), a table flag is simpler, testable, and doesn't require a Realtime connection in the worker. |
| respx (httpx mock) | responses (requests mock) | responses only works with the requests library. Since we're using httpx, respx is the correct choice. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| celery / redis / rq | Task queue overhead — the existing worker.py polling loop is sufficient for current scale | Continue with worker.py poll loop + la_sync_queue flag |
| websockets / socket.io | Overkill for polling trigger — frontend polling against FastAPI GET /sync-status is simpler | FastAPI GET endpoint + polling interval |
| A new database (Redis, MongoDB) | Project constraint: Python + Supabase only | la_sync_queue table in existing Supabase |
| Evolution API webhooks directly to FastAPI | Exposes FastAPI to the internet, requires auth validation, complicates deployment | Route through N8N (already deployed at n8n.easyscale.co), which acts as the trusted intermediary |
| requests library for new code | Synchronous, blocks the event loop in FastAPI async context | httpx.AsyncClient (already installed) |

---

## Stack Patterns by Variant

**If the clinic does NOT have Evolution API (legacy clients):**
- Keep `run_local.py` with `parse_archive()` unchanged
- The la_sync_queue table is irrelevant — jobs are created via `POST /jobs` with file upload
- No code changes needed for backward compatibility

**If Evolution API rate limits are hit:**
- Add `asyncio.sleep(0.5)` between `findMessages` calls per conversation
- Evolution API v2 default rate limits are not publicly documented; assume 10 req/s per instance as a safe starting point

**If N8N is unavailable:**
- Add a `POST /sync-trigger` FastAPI endpoint that manually sets `la_sync_queue.status = 'pending'`
- This provides a direct fallback without N8N in the loop

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| httpx 0.28.1 | pydantic 2.10.6 | No conflicts. httpx uses its own model layer. |
| httpx 0.28.1 | fastapi 0.115.6 | FastAPI uses httpx internally for TestClient. No conflicts. |
| pytest-asyncio 0.23.8 | pytest 8.x | 0.23.x series is compatible with pytest 8. Do NOT use 0.21.x (asyncio_mode default changed). |
| respx 0.21.1 | httpx 0.28.x | respx 0.21.x supports httpx 0.28.x. Do NOT use respx < 0.20 (API changed). |
| supabase 2.13.0 | Evolution API (REST) | No direct interaction. supabase-py writes the flag; httpx reads Evolution. No conflict. |

---

## Sources

- Codebase analysis (`notifier.py`, `config.py`, `worker.py`, `parser.py`, `requirements.txt`) — HIGH confidence for existing patterns
- `supabase/schema.sql` — HIGH confidence for database structure and extension patterns
- Evolution API v2 REST interface (`apikey` header auth, `/chat/findChats`, `/chat/findMessages`) — MEDIUM confidence (training knowledge, not live-verified; verify against https://doc.evolution-api.com/v2/ before implementation)
- N8N webhook + Supabase node pattern — MEDIUM confidence (training knowledge; pattern already proven in this codebase via `NOTIFY_WEBHOOK_URL`)
- httpx, pytest-asyncio, respx version compatibility — MEDIUM confidence (training knowledge as of Aug 2025; verify pinned versions against PyPI before install)

---

*Stack research for: Evolution API + N8N + WhatsApp sync integration into Python/Supabase pipeline*
*Researched: 2026-03-13*
