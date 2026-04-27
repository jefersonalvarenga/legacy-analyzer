# Roadmap: EasyScale Legacy Analyzer

## Milestones

- ✅ **v0 — Pipeline Local** - Phases 1-5 (shipped 2026-03-13)
- ❌ **v1.0 — Evolution API Integration** - Phases 6-8 (closed without execution 2026-03-16)
- 🚧 **v1.1 — Evolution API Go Live** - Phases 6-9 + 8.1 (in progress)

## Phases

<details>
<summary>✅ v0 — Pipeline Local (Phases 1–5) - SHIPPED 2026-03-13</summary>

Phases 1–5 were executed informally before GSD adoption. Shipped capabilities:
- Parser WhatsApp Archive.zip → Conversation objects
- Analise semantica DSPy (sentiment, topics, quality, summary)
- Deteccao de desfechos (agendado, ghosting, objecao, pendente)
- Shadow DNA extraction + Financial KPIs
- Blueprint JSON salvo em la_blueprints (Supabase) com clinic_id
- Knowledge Consolidator offline mode + testes de contrato Sofia

</details>

<details>
<summary>❌ v1.0 — Evolution API Integration (Phases 6-8) - CLOSED 2026-03-16 (not executed)</summary>

Planned but not executed. Superseded by v1.1 with more precise scope:
- Phase 6: Evolution Ingestor
- Phase 7: FastAPI Endpoints
- Phase 8: Pipeline Integration

Reason closed: Planejado com lacunas antes do alinhamento arquitetural completo.
Substituido por v1.1 com escopo mais preciso incluindo la_resources, la_services,
e KC suspenso integralmente.

</details>

### 🚧 v1.1 — Evolution API Go Live (In Progress)

**Milestone Goal:** Integrar o LA com o Evolution API (leitura via Supabase), expor API HTTP para trigger do frontend, inferir resources e services detectados nas conversas, e entregar blueprint completo para a Sofia no onboarding.

- [x] **Phase 6: Evolution Ingestor** - Adapter que le conversas da tabela Message do Evolution no Supabase e produz objetos Conversation/Message identicos ao parser existente (completed 2026-03-16)
- [x] **Phase 7: FastAPI Endpoints** - Endpoints POST /analyze/{clinic_id} e GET /jobs/{job_id} com validacao de clinic_id em sf_clinics e execucao em background (completed 2026-03-17)
- [x] **Phase 8: Resources and Services Inference** - Inferencia de profissionais (la_resources) e procedimentos/servicos (la_services) a partir das conversas, com schedule_type e frequencia de mencao (completed 2026-03-16)
- [x] **Phase 9: Pipeline Integration** - Pipeline completo end-to-end (Evolution → metricas, DSPy, Shadow DNA, blueprint) funcionando com clinic_id correto e blueprint acessivel pela Sofia (completed 2026-03-18)
- [ ] **Phase 8.1: Enriched Inference & Playbook** - Inferencia enriquecida: clinic_playbook forense, service_playbooks por servico, returning_patient_playbook, operating_hours, source_signals, requires_evaluation, reference_conversation_ids

## Phase Details

### Phase 6: Evolution Ingestor
**Goal**: LA consegue ler conversas diretamente da tabela Message do Evolution no Supabase e entregar objetos Conversation/Message prontos para o pipeline existente
**Depends on**: Nothing (first phase of this milestone)
**Requirements**: ING-01, ING-02, ING-03
**Success Criteria** (what must be TRUE):
  1. Dado um clinic_id valido com instancia Evolution associada, o ingestor retorna uma lista de objetos Conversation preenchidos com as mensagens dessa clinica
  2. Cada Conversation contem objetos Message com sender, timestamp e body no mesmo formato que o parser do Archive.zip produziria — pipeline existente aceita sem modificacao
  3. Conversas de outras clinicas nao aparecem no resultado (filtragem por instanceId associado ao onboarding da clinica, nunca por outra clinica)
  4. O ingestor e read-only: nenhum INSERT, UPDATE ou DELETE e feito na tabela Message ou em qualquer outra tabela do Evolution
**Plans**: 1 plan
Plans:
- [x] 06-01-PLAN.md — Evolution Ingestor: test stubs (TDD RED) + full implementation (GREEN)

### Phase 7: FastAPI Endpoints ✅ COMPLETED 2026-03-17
**Goal**: Frontend pode disparar analise passando clinic_id e acompanhar o progresso via API REST, com validacao fail-fast se clinic_id nao existir
**Depends on**: Phase 6
**Requirements**: API-01, API-02, API-03
**Plans**: 1 plan
Plans:
- [x] 07-01-PLAN.md — FastAPI Endpoints: test stubs (TDD RED) + POST /analyze + GET /jobs enrichment (GREEN)

### Phase 8: Resources and Services Inference ✅ COMPLETED 2026-03-16
**Goal**: LA infere automaticamente os profissionais, schedule_type e procedimentos/servicos da clinica a partir das conversas e persiste como sugestoes em la_resources e la_services
**Depends on**: Phase 6
**Requirements**: RES-01, RES-02, SVC-01, SVC-02
**Success Criteria** (what must be TRUE):
  1. Ao final da analise, la_resources contem os profissionais detectados nas conversas (ex: "Dra. Ana", "Dr. Carlos") com clinic_id correto
  2. la_resources contem schedule_type inferido (single / by_professional / by_room) baseado nos padroes das conversas
  3. la_services contem os procedimentos e servicos mencionados pela clinica (ex: implante, clareamento, ortodontia) com clinic_id correto
  4. Cada registro em la_services inclui frequencia de mencao, permitindo ao admin identificar os servicos mais relevantes para a clinica
**Plans**: 2 plans
Plans:
- [x] 08-01-PLAN.md — SQL migration (la_resources + la_services) + test stubs RED + module skeleton
- [x] 08-02-PLAN.md — Full implementation GREEN + dspy_pipeline registration

### Phase 9: Pipeline Integration ✅ COMPLETED 2026-03-18
**Goal**: Analise completa end-to-end funciona com mensagens vindas do Evolution: do ingestor ao blueprint salvo em la_blueprints com clinic_id correto para a Sofia consumir
**Depends on**: Phase 7, Phase 8
**Requirements**: PIPE-01, PIPE-02
**Plans**: 2 plans
Plans:
- [x] 09-01-PLAN.md — SQL migration (la_blueprints clinic_id) + TDD RED stubs (PIPE-01, PIPE-02)
- [x] 09-02-PLAN.md — Full run_analysis() pipeline implementation GREEN + human verify checkpoint

### Phase 8.1: Enriched Inference & Playbook
**Goal**: Blueprint entrega inferencia enriquecida completa — clinic_playbook forense com reasoning, service_playbooks por servico, returning_patient_playbook, dados de perfil da clinica (operating_hours, source_signals) e suporte a selecao de conversas de referencia
**Depends on**: Phase 8, Phase 9
**Requirements**: PLAY-01, PLAY-02, PLAY-03, PROF-01
**Success Criteria** (what must be TRUE):
  1. `la_blueprints.blueprint_json` contem `clinic_playbook` com `reasoning` (texto livre forense), `phases[]` (nome livre + phase_intent enum + elements[]) e `observations`
  2. `la_blueprints.blueprint_json` contem `service_playbooks[]` — um por servico detectado, extraido das conversas com outcome=agendado, com elements[] usando vocabulario canonico
  3. `la_blueprints.blueprint_json` contem `returning_patient_playbook` com intencoes reschedule, cancellation, followup
  4. `la_blueprints.blueprint_json` contem `clinic_profile` com operating_hours, neighborhood, source_signals
  5. `la_services` tem campo `requires_evaluation boolean`
  6. `POST /analyze/{clinic_id}` aceita `reference_conversation_ids[]` opcional
**Plans**: a definir (fragmentar em chunks pequenos)
Plans:
- [ ] 08.1-01-PLAN.md — Migration la_services.requires_evaluation + reference_conversation_ids no endpoint
- [ ] 08.1-02-PLAN.md — clinic_profile: operating_hours + neighborhood + source_signals (ShadowDNASignature)
- [ ] 08.1-03-PLAN.md — service_playbooks[]: ClinicServicePlaybookSignature + extrator de elements por servico
- [ ] 08.1-04-PLAN.md — returning_patient_playbook: filtro de recorrencia + extrator de intencoes
- [ ] 08.1-05-PLAN.md — clinic_playbook forense: ClinicPlaybookSignature + reasoning + phases livres

## Progress

**Execution Order:**
6 → 7 → 8 → 9 → 8.1 (sequencial; 8.1 depende de 8 e 9)

| Phase | Plans | Status | Completed |
|-------|-------|--------|-----------|
| 1–5. Pipeline Local | — | ✅ Complete | 2026-03-13 |
| 6. Evolution Ingestor | 1/1 | ✅ Complete | 2026-03-16 |
| 7. FastAPI Endpoints | 1/1 | ✅ Complete | 2026-03-17 |
| 8. Resources & Services | 2/2 | ✅ Complete | 2026-03-16 |
| 9. Pipeline Integration | 2/2 | ✅ Complete | 2026-03-18 |
| 8.1. Enriched Inference & Playbook | 0/5 | 🔲 Not started | — |
