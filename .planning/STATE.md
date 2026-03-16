# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-16)

**Core value:** Transformar conversas de WhatsApp em conhecimento estruturado (blueprint + resources + services) que a Sofia usa para atender pacientes automaticamente.
**Current focus:** Milestone v1.1 — Evolution API Go Live

## Current Position

Phase: 6 of 9 (Evolution Ingestor)
Plan: — (not yet planned)
Status: Ready to plan
Last activity: 2026-03-16 — Roadmap v1.1 criado (Phases 6-9)

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0 (v1.1)
- Average duration: —
- Total execution time: —

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: —
- Trend: —

*Updated after each plan completion*

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

### Blockers/Concerns

- [Phase 6]: Schema exato da tabela Message do Evolution no Supabase (colunas, instanceId format) — necessario confirmar antes de implementar o ingestor
- [Phase 9]: Relacao blueprint → AgentProfile nao definida — fora do escopo v1.1, resolver apos go live

### Pending Todos

None yet.

## Session Continuity

Last session: 2026-03-16
Stopped at: Roadmap v1.1 criado — pronto para plan-phase 6
Resume file: None
