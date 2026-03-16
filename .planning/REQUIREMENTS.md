# Requirements: EasyScale Legacy Analyzer

**Defined:** 2026-03-16
**Core Value:** Transformar conversas de WhatsApp em conhecimento estruturado (blueprint + resources + services) que a Sofia usa para atender pacientes automaticamente.

## v1.1 Requirements — Evolution API Go Live

### Ingestão

- [ ] **ING-01**: LA lê mensagens da tabela `Message` do Evolution WHERE instanceId = instância do onboarding
- [ ] **ING-02**: Adapter mapeia formato `Message` → objetos internos `Conversation`/`Message`
- [ ] **ING-03**: Filtra conversas por clinic_id (via instância associada ao onboarding)

### API

- [ ] **API-01**: `POST /analyze/{clinic_id}` — valida clinic_id em sf_clinics, cria job, inicia análise em background, retorna job_id imediatamente
- [ ] **API-02**: `GET /jobs/{job_id}` — retorna status (pending / running / complete / failed) e progresso
- [ ] **API-03**: `main.py` atualizada para suportar novo fluxo sem quebrar comportamento existente

### Resources

- [ ] **RES-01**: LA infere profissionais mencionados nas conversas (ex: "Dra. Ana", "Dr. Carlos") → salva em `la_resources`
- [ ] **RES-02**: LA infere `schedule_type` (single / by_professional / by_room) → salva em `la_resources`

### Services

- [ ] **SVC-01**: LA infere procedimentos e serviços oferecidos pela clínica (ex: implante, clareamento, ortodontia) → salva em `la_services`
- [ ] **SVC-02**: `la_services` inclui frequência de menção (indica relevância do serviço para a clínica)

### Pipeline

- [ ] **PIPE-01**: Pipeline completo (métricas, DSPy, desfechos, Shadow DNA, blueprint) funciona com mensagens do Evolution
- [ ] **PIPE-02**: Blueprint salvo em `la_blueprints` com `clinic_id` correto para a Sofia consumir

## v2 Requirements

### Multi-instância

- **MULTI-01**: Consolidação de análises de N instâncias Evolution de uma mesma Unit
- **MULTI-02**: `sf_instance_clinic_map` usada no fluxo de análise multi-instância

### Monitoramento automático

- **MON-01**: Worker monitora `sf_clinics` e dispara análise automaticamente (sem depender do frontend)
- **MON-02**: Re-análise incremental — processa apenas mensagens novas desde a última análise

### KC

- **KC-01**: Knowledge Consolidator online reativado após validação com clínicas reais

## Out of Scope

| Feature | Reason |
|---------|--------|
| KC (offline e online) | Suspenso integralmente — foco no go live |
| Archive.zip como fallback | Só se performance do Evolution inviabilizar |
| Relação blueprint → AgentProfile | A definir após go live |
| Blueprint overwrite por unidade | v2+ junto com multi-instância |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ING-01 | Phase 6 | Pending |
| ING-02 | Phase 6 | Pending |
| ING-03 | Phase 6 | Pending |
| API-01 | Phase 7 | Pending |
| API-02 | Phase 7 | Pending |
| API-03 | Phase 7 | Pending |
| RES-01 | Phase 8 | Pending |
| RES-02 | Phase 8 | Pending |
| SVC-01 | Phase 8 | Pending |
| SVC-02 | Phase 8 | Pending |
| PIPE-01 | Phase 9 | Pending |
| PIPE-02 | Phase 9 | Pending |

**Coverage:**
- v1.1 requirements: 12 total
- Mapped to phases: 12
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-16*
*Last updated: 2026-03-16 after initial definition*
