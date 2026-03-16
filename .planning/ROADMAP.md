# Roadmap: EasyScale Legacy Analyzer

## Milestones

- ✅ **v0 — Pipeline Local** - Phases 1-5 (shipped 2026-03-13)
- ❌ **v1.0 — Evolution API Integration** - Phases 6-8 (closed without execution 2026-03-16)
- 🚧 **v1.1 — Evolution API Go Live** - Phases 6-9 (in progress)

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
- [ ] **Phase 7: FastAPI Endpoints** - Endpoints POST /analyze/{clinic_id} e GET /jobs/{job_id} com validacao de clinic_id em sf_clinics e execucao em background
- [ ] **Phase 8: Resources and Services Inference** - Inferencia de profissionais (la_resources) e procedimentos/servicos (la_services) a partir das conversas, com schedule_type e frequencia de mencao
- [ ] **Phase 9: Pipeline Integration** - Pipeline completo end-to-end (Evolution → metricas, DSPy, Shadow DNA, blueprint) funcionando com clinic_id correto e blueprint acessivel pela Sofia

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

### Phase 7: FastAPI Endpoints
**Goal**: Frontend pode disparar analise passando clinic_id e acompanhar o progresso via API REST, com validacao fail-fast se clinic_id nao existir
**Depends on**: Phase 6
**Requirements**: API-01, API-02, API-03
**Success Criteria** (what must be TRUE):
  1. POST /analyze/{clinic_id} retorna job_id imediatamente (< 1 segundo) e a analise continua em background sem bloquear a resposta HTTP
  2. GET /jobs/{job_id} retorna status atual (pending / running / complete / failed) e percentual de progresso consultavel a qualquer momento
  3. POST /analyze/{clinic_id} com clinic_id inexistente em sf_clinics retorna HTTP 404 antes de iniciar qualquer analise ou criar qualquer job
  4. main.py suporta o novo fluxo sem quebrar endpoints ou comportamentos pre-existentes
**Plans**: 1 plan
Plans:
- [ ] 07-01-PLAN.md — FastAPI Endpoints: test stubs (TDD RED) + POST /analyze + GET /jobs enrichment (GREEN)

### Phase 8: Resources and Services Inference
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
- [ ] 08-01-PLAN.md — SQL migration (la_resources + la_services) + test stubs RED + module skeleton
- [ ] 08-02-PLAN.md — Full implementation GREEN + dspy_pipeline registration

### Phase 9: Pipeline Integration
**Goal**: Analise completa end-to-end funciona com mensagens vindas do Evolution: do ingestor ao blueprint salvo em la_blueprints com clinic_id correto para a Sofia consumir
**Depends on**: Phase 7, Phase 8
**Requirements**: PIPE-01, PIPE-02
**Success Criteria** (what must be TRUE):
  1. Dado clinic_id de uma clinica com mensagens no Evolution, o pipeline executa metricas, DSPy, deteccao de desfechos, Shadow DNA e produz blueprint_json valido
  2. Blueprint e salvo em la_blueprints com clinic_id correto — Sofia consegue consumir via polling WHERE clinic_id = UUID ORDER BY created_at DESC LIMIT 1 sem nenhuma alteracao no contrato existente
  3. la_resources e la_services sao persistidos durante a mesma execucao que salva o blueprint (analise atomica por clinic_id)
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 6 → 7 → 8 → 9
Note: Phase 8 depends on Phase 6 (not Phase 7). Phase 9 depends on both Phase 7 and Phase 8.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1–5. Pipeline Local | v0 | - | Complete | 2026-03-13 |
| 6. Evolution Ingestor | 1/1 | Complete   | 2026-03-16 | - |
| 7. FastAPI Endpoints | v1.1 | 0/1 | Not started | - |
| 8. Resources and Services Inference | 1/2 | In Progress|  | - |
| 9. Pipeline Integration | v1.1 | 0/TBD | Not started | - |
