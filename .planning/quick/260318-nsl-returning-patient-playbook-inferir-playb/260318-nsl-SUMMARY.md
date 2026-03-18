---
quick_task_id: 260318-nsl-returning-patient-playbook
date: "2026-03-18"
subsystem: shadow_dna / blueprint
tags: [dspy, playbook, returning-patient, shadow-dna]
key_files:
  modified:
    - analyzer/shadow_dna.py
    - analyzer/blueprint.py
    - analyzer/analysis_runner.py
decisions:
  - returning_patient_playbook computed in shadow_dna.py, not a new file — keeps all DSPy playbook logic co-located
  - returns None (not empty dict) when no recurrence signals found — avoids hallucination
  - wired as resilient step 9b in analysis_runner — failure never blocks pipeline
metrics:
  completed_date: "2026-03-18"
---

# Quick Task 260318-nsl: returning_patient_playbook — inferir playbook de paciente recorrente

**One-liner:** DSPy inference of reschedule/cancellation/followup playbooks from recurrence-signal-filtered conversations, wired into the main pipeline as a resilient non-blocking step.

## What Was Done

### 1. `analyzer/shadow_dna.py`

Added:

- `ReturningPatientPlaybookSignature` — DSPy Signature with one input (`conversations_sample`) and six outputs: `reschedule_elements`, `cancellation_elements`, `followup_elements`, `reschedule_example`, `cancellation_example`, `followup_example`. Uses canonical element vocabulary in the docstring.
- `ReturningPatientPlaybookModule` — thin `dspy.Module` wrapping the signature.
- `_RECURRENCE_SIGNALS` — compiled regex covering all specified recurrence patterns (remarcar, cancelar, retorno, "já fui aí", etc.)
- `_has_recurrence_signal(conv)` — returns True if any message in the conversation matches.
- `_build_recurrence_sample(conversations, max_convs=8)` — builds text sample from recurrent conversations, capped at 12,000 chars.
- `_validate_playbook_elements(raw)` — parses DSPy list output (str or list) into validated dicts with canonical keys.
- `extract_returning_patient_playbook(conversations)` — filters conversations, returns None if none found, otherwise calls DSPy and returns dict with `reschedule`, `cancellation`, `followup` sections.
- `_returning_patient_module` instance initialized inside `init_shadow_module()` (no new init function needed).

### 2. `analyzer/blueprint.py`

- Added `returning_patient_playbook: Optional[dict] = None` parameter to `build_blueprint()`.
- Added `"returning_patient_playbook": returning_patient_playbook or None` to the blueprint dict.

### 3. `analyzer/analysis_runner.py`

- Added `extract_returning_patient_playbook` to import from `analyzer.shadow_dna`.
- Added resilient step 9b between shadow_dna extraction and outcome aggregation.
- Passed `returning_patient_playbook` to `build_blueprint()` call.

## Schema Produced

```json
{
  "reschedule": {
    "elements": [{"element": "greeting", "initiated_by": "sofia", "trigger_signals": [], "blocked_by": [], "real_example": "..."}],
    "real_example": "mensagem real da clínica ao remarcar"
  },
  "cancellation": {
    "elements": [...],
    "real_example": "..."
  },
  "followup": {
    "elements": [...],
    "real_example": "..."
  }
}
```

## Deviations from Plan

None — plan executed exactly as written. Implementation placed in `shadow_dna.py` (not a new file) as the task explicitly allowed either option and co-location is cleaner.

## Commits

- `d7cc215`: feat(260318-nsl): add returning_patient_playbook inference

## Self-Check

- `analyzer/shadow_dna.py` modified: FOUND
- `analyzer/blueprint.py` modified: FOUND
- `analyzer/analysis_runner.py` modified: FOUND
- Commit `d7cc215` exists: FOUND

## Self-Check: PASSED
