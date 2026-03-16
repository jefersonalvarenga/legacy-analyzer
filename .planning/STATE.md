---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: — Evolution API Go Live
status: completed
stopped_at: Completed 08-02-PLAN.md — Resources and Services Inference (TDD GREEN)
last_updated: "2026-03-16T20:58:14.058Z"
last_activity: 2026-03-16 — 06-01 Evolution Ingestor implemented
progress:
  total_phases: 4
  completed_phases: 3
  total_plans: 4
  completed_plans: 4
  percent: 11
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-16)

**Core value:** Transformar conversas de WhatsApp em conhecimento estruturado (blueprint + resources + services) que a Sofia usa para atender pacientes automaticamente.
**Current focus:** Milestone v1.1 — Evolution API Go Live

## Current Position

Phase: 6 of 9 (Evolution Ingestor)
Plan: 1 of 1 — COMPLETE
Status: Phase 6 complete — ready for Phase 7 (API Endpoint)
Last activity: 2026-03-16 — 06-01 Evolution Ingestor implemented

Progress: [█░░░░░░░░░] 11%

## Performance Metrics

**Velocity:**
- Total plans completed: 1 (v1.1)
- Average duration: 18 min
- Total execution time: 18 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 06 | 1 | 18 min | 18 min |

**Recent Trend:**
- Last 5 plans: 06-01 (18 min)
- Trend: —

*Updated after each plan completion*
| Phase 07-fastapi-endpoints P01 | 2 | 3 tasks | 4 files |
| Phase 08-resources-and-services-inference P01 | 3min | 2 tasks | 3 files |
| Phase 08-resources-and-services-inference P02 | 12 | 2 tasks | 2 files |

## Accumulated Context

### Decisions

- Frontend chama POST /analyze/{clinic_id} — nao instance_name
- LA valida clinic_id em sf_clinics (fail fast) antes de criar job
- LA le tabela Message do Evolution (Supabase compartilhado) — read-only, nunca escreve
- la_resources: LA sugere, admin confirma, Website cria em sf_resources
- la_services: LA infere procedimentos com frequencia de mencao — sugestao para admin
- KC suspenso integralmente — nao evoluir consolidate_knowledge_offline()
- Deploy: EasyPanel VPS
- v1.1 analisa 1 instancia por clinica; multi-instancia e v2+
- fromMe flag e o unico classificador para sender_type (clinic vs patient) — pushName nunca usado para classificar
- source_filename = remoteJid string no Conversation (compatibilidade de tipo com parser.py)
- days_back=90 como limite padrao para queries de Message — evita volume ilimitado
- raw_line="" para todas as mensagens do Evolution — nao ha linha de texto bruto na API
- [Phase 07-fastapi-endpoints]: POST /analyze/{clinic_id} retorna 202 imediatamente via FastAPI BackgroundTasks
- [Phase 07-fastapi-endpoints]: STATUS_MAP normaliza enums do DB para contrato estavel de API (pending/running/complete/failed)
- [Phase 07-fastapi-endpoints]: Status 'pending' reservado para jobs Evolution — worker continua polling 'queued' sem conflito
- [Phase 08-resources-and-services-inference]: la_resources/la_services use clinic_id FK to sf_clinics (not la_clients) — v1.1 Evolution flow tables
- [Phase 08-resources-and-services-inference]: Delete only confirmed=FALSE rows before insert — admin-confirmed resources survive re-analysis
- [Phase 08-resources-and-services-inference]: schedule_type denormalized on each la_resources row to mirror sf_resources schema
- [Phase 08-resources-and-services-inference]: DSPy module called via .forward() directly for testability with MagicMock patches
- [Phase 08-resources-and-services-inference]: Services inserted before resources in persist_resources() to match mock call_args_list assertion ordering

### Blockers/Concerns

- [Phase 9]: Relacao blueprint → AgentProfile nao definida — fora do escopo v1.1, resolver apos go live

### Pending Todos

None yet.

## Session Continuity

Last session: 2026-03-16T20:58:14.055Z
Stopped at: Completed 08-02-PLAN.md — Resources and Services Inference (TDD GREEN)
Resume file: None
