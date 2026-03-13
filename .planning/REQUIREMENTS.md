# Requirements: EasyScale Legacy Analyzer

**Defined:** 2026-03-13
**Core Value:** Transformar conversas de WhatsApp em conhecimento estruturado que a Sofia usa para atender pacientes automaticamente.

## v1.0 Requirements — Evolution API Integration

### Ingestão

- [ ] **ING-01**: Sistema lê conversas e mensagens das tabelas do Evolution no Supabase (sem depender de Archive.zip)
- [ ] **ING-02**: Parser adapta formato das tabelas do Evolution para os objetos `Conversation`/`Message` internos do pipeline
- [ ] **ING-03**: Sistema filtra conversas por `clinic_id` para processar apenas a clínica correta

### API

- [ ] **API-01**: Endpoint `POST /analyze/{clinic_id}` inicia análise completa para uma clínica
- [ ] **API-02**: Endpoint retorna `job_id` imediatamente (análise roda em background)
- [ ] **API-03**: Endpoint `GET /jobs/{job_id}` retorna status e progresso da análise
- [ ] **API-04**: FastAPI já existente em `main.py` é atualizada para suportar o novo fluxo de ingestão

### Contrato N8N / sf_clinics

- [ ] **CTR-01**: Frontend monitora `sf_clinics.onboarding_status = 'sync_complete'` (setado pelo N8N via `evolution_instance_id`) e só então chama a API do Legacy Analyzer
- [ ] **CTR-02**: Legacy Analyzer valida que o `clinic_id` recebido existe em `sf_clinics` antes de iniciar análise

### Pipeline

- [ ] **PIPE-01**: Pipeline de análise (métricas, DSPy, desfechos, Shadow DNA, KC, blueprint) funciona identicamente com mensagens vindas do Evolution
- [ ] **PIPE-02**: Blueprint resultante é salvo em `la_blueprints` com `clinic_id` correto para a Sofia consumir

## v2 Requirements

### Monitoramento automático

- **MON-01**: Worker interno monitora `sf_clinics` e dispara análise automaticamente ao detectar sync completa (sem depender do frontend chamar a API)
- **MON-02**: Re-análise incremental — processa apenas mensagens novas desde a última análise

### Fallback

- **FALL-01**: Archive.zip como fallback para clínicas sem Evolution API (ativado somente se performance do novo fluxo for inviável)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Workflow N8N | Já existe e está definido: UPDATE sf_clinics SET onboarding_status='sync_complete', onboarding_step=3 WHERE evolution_instance_id=instance |
| Frontend UI | Existe e será adaptado externamente; Legacy Analyzer só expõe API |
| Realtime subscribe no frontend | Responsabilidade do frontend; Legacy Analyzer não precisa saber disso |
| Migração de dados históricos do Archive.zip para Evolution | Fora do escopo deste milestone |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ING-01 | Phase 6 | Pending |
| ING-02 | Phase 6 | Pending |
| ING-03 | Phase 6 | Pending |
| API-01 | Phase 7 | Pending |
| API-02 | Phase 7 | Pending |
| API-03 | Phase 7 | Pending |
| API-04 | Phase 7 | Pending |
| CTR-01 | Phase 6 | Pending |
| CTR-02 | Phase 7 | Pending |
| PIPE-01 | Phase 8 | Pending |
| PIPE-02 | Phase 8 | Pending |

**Coverage:**
- v1.0 requirements: 11 total
- Mapped to phases: 11
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-13*
*Last updated: 2026-03-13 after initial definition*
