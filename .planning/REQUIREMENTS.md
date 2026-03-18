# Requirements: EasyScale Legacy Analyzer

**Defined:** 2026-03-16
**Core Value:** Transformar conversas de WhatsApp em conhecimento estruturado (blueprint + resources + services) que a Sofia usa para atender pacientes automaticamente.

## v1.1 Requirements — Evolution API Go Live

### Ingestao

- [x] **ING-01**: LA le mensagens da tabela `Message` do Evolution WHERE instanceId = instancia do onboarding
- [x] **ING-02**: Adapter mapeia formato `Message` → objetos internos `Conversation`/`Message`
- [x] **ING-03**: Filtra conversas por clinic_id (via instancia associada ao onboarding)

### API

- [x] **API-01**: `POST /analyze/{clinic_id}` — valida clinic_id em sf_clinics, cria job, inicia analise em background, retorna job_id imediatamente
- [x] **API-02**: `GET /jobs/{job_id}` — retorna status (pending / running / complete / failed) e progresso
- [x] **API-03**: `main.py` atualizada para suportar novo fluxo sem quebrar comportamento existente

### Resources

- [x] **RES-01**: LA infere profissionais mencionados nas conversas (ex: "Dra. Ana", "Dr. Carlos") → salva em `la_resources`
- [x] **RES-02**: LA infere `schedule_type` (single / by_professional / by_room) → salva em `la_resources`

### Services

- [x] **SVC-01**: LA infere procedimentos e servicos oferecidos pela clinica (ex: implante, clareamento, ortodontia) → salva em `la_services`
- [x] **SVC-02**: `la_services` inclui frequencia de mencao (indica relevancia do servico para a clinica)

### Pipeline

- [x] **PIPE-01**: Pipeline completo (metricas, DSPy, desfechos, Shadow DNA, blueprint) funciona com mensagens do Evolution
- [x] **PIPE-02**: Blueprint salvo em `la_blueprints` com `clinic_id` correto para a Sofia consumir

## v2 Requirements

### Multi-instancia

- **MULTI-01**: Consolidacao de analises de N instancias Evolution de uma mesma Unit
- **MULTI-02**: `sf_instance_clinic_map` usada no fluxo de analise multi-instancia

### Monitoramento automatico

- **MON-01**: Worker monitora `sf_clinics` e dispara analise automaticamente (sem depender do frontend)
- **MON-02**: Re-analise incremental — processa apenas mensagens novas desde a ultima analise

### KC

- **KC-01**: Knowledge Consolidator online reativado apos validacao com clinicas reais

## Out of Scope

| Feature | Reason |
|---------|--------|
| KC (offline e online) | Suspenso integralmente — foco no go live |
| Archive.zip como fallback | So se performance do Evolution inviabilizar |
| Relacao blueprint → AgentProfile | A definir apos go live |
| Blueprint overwrite por unidade | v2+ junto com multi-instancia |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ING-01 | Phase 6 | Complete |
| ING-02 | Phase 6 | Complete |
| ING-03 | Phase 6 | Complete |
| API-01 | Phase 7 | Complete |
| API-02 | Phase 7 | Complete |
| API-03 | Phase 7 | Complete |
| RES-01 | Phase 8 | Complete |
| RES-02 | Phase 8 | Complete |
| SVC-01 | Phase 8 | Complete |
| SVC-02 | Phase 8 | Complete |
| PIPE-01 | Phase 9 | Complete |
| PIPE-02 | Phase 9 | Complete |

**Coverage:**
- v1.1 requirements: 12 total
- Mapped to phases: 12
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-16*
*Last updated: 2026-03-16 — Traceability updated for v1.1 roadmap (Phases 6-9)*
