# EasyScale Legacy Analyzer

## What This Is

Ferramenta de análise de conversas de WhatsApp de clínicas, que lê mensagens diretamente do Supabase do Evolution API, infere os 5 eixos de comportamento clínico, extrai resources (profissionais, procedimentos), gera blueprint estruturado para a Sofia e expõe API HTTP para trigger pelo frontend no onboarding.

## Core Value

Transformar conversas de WhatsApp em conhecimento estruturado (blueprint + resources) que a Sofia usa para atender automaticamente pacientes com a mesma linguagem e processo da clínica.

## Requirements

### Validated

- ✓ Parser de Archive.zip → Conversation objects — v0
- ✓ Análise semântica DSPy (sentimento, tópicos, qualidade, resumo) — v0
- ✓ Detecção de desfechos (agendado, ghosting, objeção, pendente) — v0
- ✓ Shadow DNA extraction (tom, padrões linguísticos da clínica) — v0
- ✓ Financial KPIs — v0
- ✓ Blueprint JSON salvo em la_blueprints com clinic_id — v0
- ✓ Persistência completa no Supabase — v0

### Active

- [ ] Ler mensagens da tabela `Message` do Evolution (Supabase) em vez de Archive.zip
- [ ] Endpoint `POST /analyze/{clinic_id}` — trigger pelo frontend no onboarding
- [ ] Endpoint `GET /jobs/{job_id}` — status e progresso da análise
- [ ] Validar `clinic_id` em `sf_clinics` antes de iniciar análise
- [ ] Pipeline completo funcionando com mensagens do Evolution → blueprint salvo
- [ ] Inferir e salvar resources em `la_resources` (profissionais, procedimentos detectados nas conversas)

### Out of Scope

- KC (Knowledge Consolidator) — suspenso integralmente, foco no go live online
- Consolidação multi-instância por Unit — v2+
- Archive.zip como fallback — só se performance do Evolution inviabilizar
- Relação blueprint → AgentProfile — a definir após go live

## Context

- **Evolution API:** persiste mensagens na tabela `Message` no Supabase (mesmo banco da Sofia)
- **N8N:** recebe webhook MESSAGES_SET do Evolution → `UPDATE sf_clinics SET onboarding_status='sync_complete', onboarding_step=3 WHERE evolution_instance_id=instance`
- **Frontend:** faz subscribe em `sf_clinics`, detecta `sync_complete` → chama `POST /analyze/{clinic_id}`
- **Sofia:** consome `la_blueprints` via polling `WHERE clinic_id = UUID ORDER BY created_at DESC LIMIT 1`
- **sf_resources:** tabela operacional da Sofia — o LA sugere via `la_resources`, admin confirma, Website cria em `sf_resources`
- **v1.0 (fechado):** planejado mas não executado — substituído por este milestone com escopo mais preciso

## Constraints

- **Stack:** Python + Supabase — sem novos bancos ou linguagens
- **Contrato Sofia:** `la_blueprints` com `blueprint_json` + `clinic_id` — não quebrar
- **KC suspenso:** não evoluir `consolidate_knowledge_offline()` até retomada explícita
- **Deploy:** EasyPanel VPS

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Frontend chama com clinic_id (não instance_name) | Frontend conhece clinic_id pelo contexto de autenticação | ✓ Good |
| sf_instance_clinic_map para multi-instância v2+ | v1.0 foca em 1 instância por clínica no onboarding | — Pending |
| la_resources como sugestão do LA | LA infere, admin confirma, Website cria em sf_resources | — Pending |
| KC suspenso integralmente | Acelerar go live online | — Pending |
| Deploy EasyPanel VPS | Mesmo ambiente do N8N | — Pending |
| Blueprint por organização com overwrite por unidade | Suporte a multi-unit no futuro | — Pending |

## Current Milestone: v1.1 — Evolution API Go Live

**Goal:** Integrar o LA com o Evolution API (leitura via Supabase), expor API HTTP para trigger do frontend, inferir resources detectados nas conversas, e entregar blueprint completo para a Sofia no onboarding.

**Target features:**
- Evolution Ingestor: ler `Message` WHERE instanceId = instance do onboarding
- API: `POST /analyze/{clinic_id}` + `GET /jobs/{job_id}`
- la_resources: inferir profissionais e procedimentos das conversas
- Pipeline end-to-end: Evolution → blueprint → la_blueprints

---
*Last updated: 2026-03-16 — Milestone v1.1 started (v1.0 fechado sem execução)*
