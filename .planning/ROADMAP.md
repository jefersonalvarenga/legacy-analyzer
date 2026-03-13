# Roadmap: EasyScale Legacy Analyzer

## Milestones

- ✅ **v0 — Pipeline Local** - Phases 1-5 (shipped 2026-03-13)
- 🚧 **v1.0 — Evolution API Integration** - Phases 6-8 (in progress)

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

### v1.0 — Evolution API Integration (In Progress)

**Milestone Goal:** Substituir o upload manual de Archive.zip por sincronizacao automatica via Evolution API, com N8N orquestrando o trigger e frontend monitorando via polling.

- [ ] **Phase 6: Evolution Ingestor** - Adapter que le conversas do Evolution no Supabase e produz objetos Conversation/Message identicos ao parser existente
- [ ] **Phase 7: FastAPI Endpoints** - Endpoints POST /analyze/{clinic_id} e GET /jobs/{job_id} com validacao de clinic_id e execucao em background
- [ ] **Phase 8: Pipeline Integration** - Pipeline completo (metricas, DSPy, Shadow DNA, KC, blueprint) funcionando com mensagens vindas do Evolution, blueprint salvo com clinic_id correto

## Phase Details

### Phase 6: Evolution Ingestor
**Goal**: Sistema le conversas diretamente das tabelas do Evolution no Supabase e entrega objetos Conversation/Message prontos para o pipeline
**Depends on**: Nothing (first phase of this milestone)
**Requirements**: ING-01, ING-02, ING-03, CTR-01
**Success Criteria** (what must be TRUE):
  1. Dado um clinic_id valido, o ingestor retorna uma lista de objetos Conversation preenchidos com as mensagens daquela clinica
  2. Cada Conversation contem objetos Message com sender, timestamp e body no mesmo formato que o parser do Archive.zip produziria
  3. Conversas de outras clinicas nao aparecem no resultado (filtragem por clinic_id e isolation garantida)
  4. CTR-01 esta documentado: o fluxo N8N -> sf_clinics.onboarding_status='sync_complete' -> frontend chama API esta descrito em docs/
**Plans**: TBD

Plans:
- [ ] 06-01: Criar analyzer/evolution_ingestor.py — consulta Evolution Supabase tables, filtra por clinic_id, mapeia para Conversation/Message
- [ ] 06-02: Documentar contrato CTR-01 (fluxo N8N / sf_clinics / frontend) em docs/

### Phase 7: FastAPI Endpoints
**Goal**: Frontend e N8N podem disparar analise por clinic_id e acompanhar progresso via API REST
**Depends on**: Phase 6
**Requirements**: API-01, API-02, API-03, API-04, CTR-02
**Success Criteria** (what must be TRUE):
  1. POST /analyze/{clinic_id} retorna job_id imediatamente (resposta em < 1 segundo) e a analise continua em background
  2. GET /jobs/{job_id} retorna status atual (pending / running / complete / failed) e percentual de progresso
  3. POST /analyze/{clinic_id} com um clinic_id inexistente em sf_clinics retorna HTTP 404 antes de iniciar qualquer analise
  4. main.py suporta o novo fluxo sem quebrar endpoints pre-existentes
**Plans**: TBD

Plans:
- [ ] 07-01: Adicionar POST /analyze/{clinic_id} em main.py — validar clinic_id em sf_clinics (CTR-02), criar job, disparar background task com ingestor
- [ ] 07-02: Adicionar GET /jobs/{job_id} em main.py — consultar la_analysis_jobs e retornar status/progresso
- [ ] 07-03: Testes de integracao para ambos os endpoints (clinic_id valido, invalido, job inexistente)

### Phase 8: Pipeline Integration
**Goal**: Analise completa end-to-end funciona com mensagens do Evolution: do ingestor ao blueprint na la_blueprints, sem alterar o pipeline existente
**Depends on**: Phase 7
**Requirements**: PIPE-01, PIPE-02
**Success Criteria** (what must be TRUE):
  1. Dado clinic_id de uma clinica com mensagens no Evolution, o pipeline produz um blueprint_json valido salvo em la_blueprints com o clinic_id correto
  2. Metricas, DSPy, deteccao de desfechos, Shadow DNA, KC e blueprint executam sem erros usando mensagens vindas do Evolution (comportamento identico ao Archive.zip)
  3. Sofia pode consumir o blueprint via polling em la_blueprints WHERE clinic_id = UUID sem nenhuma alteracao no contrato existente
**Plans**: TBD

Plans:
- [ ] 08-01: Integrar evolution_ingestor no worker.py / pipeline — substituir chamada ao parser pelo ingestor quando clinic_id presente
- [ ] 08-02: Teste end-to-end com fixture de conversas sinteticas do Evolution — validar blueprint salvo e contrato Sofia

## Progress

**Execution Order:**
Phases execute in numeric order: 6 → 7 → 8

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1–5. Pipeline Local | v0 | - | Complete | 2026-03-13 |
| 6. Evolution Ingestor | v1.0 | 0/2 | Not started | - |
| 7. FastAPI Endpoints | v1.0 | 0/3 | Not started | - |
| 8. Pipeline Integration | v1.0 | 0/2 | Not started | - |
