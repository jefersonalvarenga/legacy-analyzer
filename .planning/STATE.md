# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-16)

**Core value:** Transformar conversas de WhatsApp em conhecimento estruturado (blueprint + resources) que a Sofia usa para atender pacientes automaticamente.
**Current focus:** Milestone v1.1 — Evolution API Go Live

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-03-16 — Milestone v1.1 iniciado (v1.0 fechado sem execução)

## Accumulated Context

### Decisions

- Frontend chama `POST /analyze/{clinic_id}` — não instance_name
- LA valida clinic_id em sf_clinics (fail fast)
- LA lê tabela `Message` do Evolution (Supabase compartilhado) — read-only
- la_resources: LA sugere, admin confirma, Website cria em sf_resources
- KC suspenso integralmente — não evoluir
- Deploy: EasyPanel VPS
- v1.1 analisa 1 instância por clínica; multi-instância é v2+

### Blockers/Concerns

- Schema exato da tabela `Message` do Evolution no Supabase (colunas, instanceId format) — necessário antes de Phase 6
- Relação blueprint → AgentProfile não definida — fora do escopo v1.1, resolver após go live
