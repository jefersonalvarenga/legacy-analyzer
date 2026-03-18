---
task_id: 260318-nsl-clinic-profile
date: 2026-03-18
subsystem: shadow_dna, blueprint
tags: [clinic_profile, operating_hours, source_signals, neighborhood, shadow_dna, blueprint]
key_files:
  modified:
    - analyzer/shadow_dna.py
    - analyzer/blueprint.py
decisions:
  - operating_hours defaults to None (not empty dict) when LLM returns {} or unparseable value
  - source_signals uses _safe_list() per channel to normalize LLM list output consistently
  - neighborhood maps local_neighborhoods[0] (first entry) to avoid inventing a value
commit: 3128bc0
duration: ~10 min
---

# Quick Task 260318-nsl: clinic_profile inference — operating_hours + neighborhood + source_signals

**One-liner:** DSPy OutputFields and ShadowDNA dataclass fields for operating_hours and source_signals, wired into blueprint.clinic_profile block alongside neighborhood mapped from local_neighborhoods.

## What Was Done

### analyzer/shadow_dna.py

**ShadowDNASignature — 2 new OutputFields added:**
- `operating_hours: dict` — LLM infers `{"open": "HH:MM", "close": "HH:MM", "days": [...]}` from conversation text. Returns `{}` when not inferable.
- `source_signals: dict` — LLM extracts channel-keyed dict of real quoted phrases (instagram, google, referral, etc.). Returns `{}` when no evidence.

**ShadowDNA dataclass — 2 new fields:**
- `operating_hours: Optional[dict] = None` — None when LLM returns empty or unparseable.
- `source_signals: dict[str, list] = field(default_factory=dict)` — empty dict when no signals found.

**extract_shadow_dna() — assignment block after attendance_flow_steps:**
- `operating_hours`: accepts dict directly or parses string via `ast.literal_eval`; sets `None` on empty/failure.
- `source_signals`: accepts dict directly or parses string; normalizes each channel's phrases via `_safe_list()`; sets `{}` on failure.

**local_neighborhoods** was not modified — it was already an OutputField and dataclass field. Only mapped in blueprint.

### analyzer/blueprint.py

New `clinic_profile` top-level key added to the blueprint dict returned by `build_blueprint()`:
```python
"clinic_profile": {
    "operating_hours": shadow_dna.operating_hours,
    "neighborhood": shadow_dna.local_neighborhoods[0] if shadow_dna.local_neighborhoods else None,
    "source_signals": shadow_dna.source_signals,
    "tone": shadow_dna.tone_classification,
    "agent_name": shadow_dna.agent_suggested_name,
}
```

## Deviations from Plan

None — plan executed exactly as written.

## Decisions Made

1. `operating_hours` defaults to `None` (not `{}`) when LLM returns empty or string parse fails — cleaner for consumers to distinguish "not found" from "empty object".
2. `source_signals` per-channel normalization reuses existing `_safe_list()` helper — consistent with all other list fields in the module.
3. `neighborhood` uses `local_neighborhoods[0]` — takes the most prominent neighborhood the LLM extracted. No fallback invented per task rules.

## Self-Check

- [x] `analyzer/shadow_dna.py` modified with 2 new OutputFields + 2 new dataclass fields + assignment wiring
- [x] `analyzer/blueprint.py` modified with `clinic_profile` block
- [x] Commit `3128bc0` exists and includes both files (2 files changed, 207 insertions)
- [x] No pytest run (per constraints)
- [x] ROADMAP.md not updated (per constraints)
