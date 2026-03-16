---
phase: 06-evolution-ingestor
verified: 2026-03-16T20:00:00Z
status: passed
score: 7/7 must-haves verified
re_verification: false
---

# Phase 6: Evolution Ingestor Verification Report

**Phase Goal:** Create the Evolution Ingestor — a read-only adapter that queries Evolution API's Message table in Supabase and produces list[Conversation] identical in type to parse_archive() output, so the existing pipeline works without modification.
**Verified:** 2026-03-16T20:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | ingest_from_evolution(clinic_id, clinic_sender_name) returns list[Conversation] when the clinic has messages in the Evolution Message table | VERIFIED | Function defined at line 215 of evolution_ingestor.py with return type list[Conversation]; test_output_type_compatibility asserts isinstance(conv, Conversation); 11 tests pass |
| 2 | Each returned Conversation contains Message objects with sent_at (datetime), sender (str), sender_type ('clinic'\|'patient'), and content (str) — identical fields to parser.py output | VERIFIED | Message objects created in _group_messages_by_conversation() (lines 187-193) using the Conversation and Message dataclasses imported directly from analyzer.parser; test_output_type_compatibility asserts isinstance(msg, Message) |
| 3 | Messages where fromMe=true map to sender_type='clinic'; fromMe=false map to sender_type='patient' | VERIFIED | _build_sender_type() at line 134 returns "clinic" if from_me else "patient"; test_sender_type_clinic and test_sender_type_patient both pass |
| 4 | messageTimestamp (Unix int) is converted to a Python datetime object | VERIFIED | Line 179: sent_at = datetime.fromtimestamp(int(raw_ts)); test_timestamp_conversion asserts sent_at == datetime.fromtimestamp(1700000000) and passes |
| 5 | Rows with remoteJid ending in @g.us are excluded from output | VERIFIED | _is_group_jid() at line 129 checks endswith("@g.us"); skipped in _group_messages_by_conversation() lines 162-163; test_group_jid_excluded passes |
| 6 | The function raises ValueError for an unknown clinic_id | VERIFIED | _resolve_instance_id() raises ValueError("Clinic '...' not found in sf_clinics table") when clinic_data is None (line 60-63); test_resolve_invalid_clinic_id with match="not found" passes |
| 7 | No INSERT, UPDATE, or DELETE call exists in evolution_ingestor.py — only .select().eq().execute() calls | VERIFIED | grep -n '\.insert(\|\.update(\|\.delete(\|\.upsert(' returns no matches; only .select() chains present in lines 52-57 and 70-75 and 257-263 |

**Score:** 7/7 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `analyzer/evolution_ingestor.py` | Public ingest_from_evolution() function + private helpers | VERIFIED | File exists, 284 lines, substantive implementation; exports ingest_from_evolution, _resolve_instance_id, _extract_body, _extract_phone, _is_group_jid, _build_sender_type, _group_messages_by_conversation |
| `tests/test_evolution_ingestor.py` | Unit tests for all ING-0x requirements | VERIFIED | File exists, 285 lines; 11 tests collected covering all 8 planned behavioral scenarios; all pass |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| ingest_from_evolution(clinic_id) | sf_clinics.evolution_instance_id | _resolve_instance_id() first hop | WIRED | Line 52-58: db.table("sf_clinics").select("evolution_instance_id").eq("id", clinic_id).single().execute() — exact pattern from PLAN frontmatter |
| sf_clinics.evolution_instance_id (instance name string) | Instance.id (UUID) | _resolve_instance_id() second hop | WIRED | Lines 71-77: db.table("Instance").select("id").eq("name", instance_name).single().execute() — exact pattern from PLAN frontmatter |
| Message rows | Conversation objects | _group_messages_by_conversation() | WIRED | Lines 152-206: defaultdict(list) keyed on remote_jid (from key["remoteJid"]); called at line 276; returns list[Conversation] |

All three key links wired with exact query patterns matching the PLAN contract.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| ING-01 | 06-01-PLAN.md | LA le mensagens da tabela Message do Evolution WHERE instanceId = instancia do onboarding | SATISFIED | _resolve_instance_id() resolves clinic → instance UUID; Message query at line 257-263 filters by .eq("instanceId", instance_uuid); test_queries_by_instance_id verifies eq("instanceId", INSTANCE_UUID) is called |
| ING-02 | 06-01-PLAN.md | Adapter mapeia formato Message → objetos internos Conversation/Message | SATISFIED | _group_messages_by_conversation() maps all fields: sent_at from messageTimestamp, sender from pushName/clinic_sender_name, sender_type from fromMe, content from _extract_body(); types imported from analyzer.parser |
| ING-03 | 06-01-PLAN.md | Filtra conversas por clinic_id (via instancia associada ao onboarding) | SATISFIED | Two-hop resolution ensures only rows belonging to the clinic's Evolution instance are fetched; group JID exclusion at lines 162-163; ValueError for invalid clinic_id at lines 60-63; test_isolation_by_instance_id and test_resolve_invalid_clinic_id pass |

No orphaned requirements: REQUIREMENTS.md traceability table maps ING-01, ING-02, ING-03 exclusively to Phase 6. No additional IDs mapped to this phase.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| analyzer/evolution_ingestor.py | 21 | Comment mentions "insert, update, delete, upsert" as prohibited | Info | Comment-only — no live write calls present. grep for actual method calls (.insert(), .update(), .delete(), .upsert()) returns zero matches. No impact. |

No TODO, FIXME, placeholder, or stub anti-patterns found. No empty implementations. No return null or return {} in public API.

---

### Human Verification Required

None. All observable truths are programmatically verifiable via unit tests and static analysis. The ingestor is a pure data-mapping adapter with no UI, no real-time behavior, and no external service integration beyond the mocked Supabase client.

---

### Test Coverage Note

The PLAN specified 8 test function names. The implementation delivers 11 tests — TestSenderTypeMapping split into test_sender_type_clinic + test_sender_type_patient, and TestBodyExtraction split into test_body_conversation + test_body_extended_text_message + test_body_media_fallback. All 8 planned behavioral scenarios are covered; the expansion improves granularity without scope creep.

---

### Gaps Summary

None. All 7 must-have truths verified, all 3 artifacts pass levels 1-3 (exists, substantive, wired), all 3 key links verified against actual code, all 3 requirement IDs satisfied with evidence.

---

## Verification Commands Run

```
pytest tests/test_evolution_ingestor.py -v         → 11 passed
pytest tests/ -x -q                                → 39 passed, no regressions
grep -n '.insert(' analyzer/evolution_ingestor.py  → no matches
grep -n '.update(' analyzer/evolution_ingestor.py  → no matches
grep -n '.delete(' analyzer/evolution_ingestor.py  → no matches
grep -n '.upsert(' analyzer/evolution_ingestor.py  → no matches
git log --oneline 4d328f3 3fe4afd                  → both commits found
```

---

_Verified: 2026-03-16T20:00:00Z_
_Verifier: Claude (gsd-verifier)_
