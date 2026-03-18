---
quick_task: 260318-nwq
title: service_playbooks por servico — ClinicServicePlaybookSignature + extrator de elements
completed_at: "2026-03-18"
commit: 770e203
files_modified:
  - analyzer/playbook_inference.py
  - analyzer/blueprint.py
  - analyzer/analysis_runner.py
  - main.py
---

# Quick Task 260318-nwq: service_playbooks por servico

**One-liner:** Per-service DSPy playbook extraction (ServicePlaybookSignature) with agendado-filtered conversations, reference_ids support, and blueprint integration.

## What Was Done

### 1. `analyzer/playbook_inference.py`

Added three new components:

**`ServicePlaybookSignature`** — DSPy Signature with:
- Inputs: `service_name: str`, `conversations_sample: str`
- Outputs: `requires_evaluation: bool`, `default_flow: str`, `elements: list`
- System prompt enforces canonical vocabulary for `element` and `blocked_by`, mandates real examples

**`ServicePlaybookModule(dspy.Module)`** — wraps `dspy.Predict(ServicePlaybookSignature)`

**`extract_service_playbooks(conversations, services, reference_ids, outcome_results)`**:
- Loops over each service in `services`
- Filters conversations: `outcome == "agendado"` + service name appears in any message
- If `reference_ids` supplied, further restricts by matching `conv.phone` or `conv.source_filename`
- Skips service if < 2 eligible conversations (returns nothing for that service)
- Calls `ServicePlaybookModule`, normalises `requires_evaluation` (bool), `default_flow` (canonical), `elements` (validated)
- Returns `list[dict]` with one entry per service with sufficient data

**`_validate_service_elements()`** — validates element list from LLM output, filters `blocked_by` to canonical values only

**`init_playbook_module()`** — extended to also call `init_service_playbook_module()` so a single init call covers both modules

### 2. `analyzer/blueprint.py`

`build_blueprint()` now accepts `service_playbooks: Optional[list] = None` and includes it in the returned dict as `"service_playbooks": service_playbooks or []`.

### 3. `analyzer/analysis_runner.py`

- Added `extract_service_playbooks` to imports
- `run_analysis()` signature extended with `reference_conversation_ids: list[str] | None = None`
- New **Step 9d** (between Step 9c clinic_playbook and Step 10 aggregate_outcomes):
  - Calls `extract_service_playbooks(conversations, shadow_dna.local_procedures, reference_conversation_ids, outcome_results)`
  - Wrapped in try/except — failure is non-blocking (`service_playbooks = []`)
- `build_blueprint()` call now passes `service_playbooks=service_playbooks`

### 4. `main.py`

- `background_tasks.add_task(run_analysis, ...)` now passes `body.reference_conversation_ids` as third argument
- Updated docstring to reflect the param is now consumed (not "stored for future use")

## Decisions Made

- Service skipped if < 2 eligible conversations — no invention, consistent with task spec
- `default_flow` defaults to `"direct_booking"` if LLM returns an unrecognised value
- `blocked_by` filtered strictly to canonical values (`already_sent`, `evaluation_not_done`, `price_not_asked`, `appointment_confirmed`) — non-canonical values silently dropped
- `init_playbook_module()` initialises both `ClinicPlaybookModule` and `ServicePlaybookModule` to keep init surface minimal
- Step 9d placed before Step 10 (aggregate_outcomes) so service playbooks use the same `outcome_results` list already computed in Step 8

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check

- `analyzer/playbook_inference.py` — FOUND (modified)
- `analyzer/blueprint.py` — FOUND (modified)
- `analyzer/analysis_runner.py` — FOUND (modified)
- `main.py` — FOUND (modified)
- Commit 770e203 — FOUND

## Self-Check: PASSED
