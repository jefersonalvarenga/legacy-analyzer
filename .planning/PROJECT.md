# EasyScale Legacy Analyzer

## What This Is

Ferramenta de análise de conversas de WhatsApp de clínicas odontológicas, que extrai métricas de atendimento, detecta desfechos (agendamentos, ghosting, objeções), gera blueprints estruturados de conhecimento clínico, e exporta dados de treinamento para a IA Sofia. Processa arquivos de conversas exportados do WhatsApp ou, na nova arquitetura, lê mensagens diretamente do Evolution API.

## Core Value

Transformar conversas de WhatsApp em conhecimento estruturado (blueprint) que a Sofia usa para atender automaticamente pacientes com a mesma linguagem e processo da clínica.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- ✓ Parser de Archive.zip com conversas WhatsApp no formato pt-BR — v0
- ✓ Análise semântica com DSPy (sentimento, tópicos, qualidade, resumo) — v0
- ✓ Detecção de desfechos (agendado, ghosting, objeção, pendente) — v0
- ✓ Shadow DNA extraction (tom, padrões linguísticos da clínica) — v0
- ✓ Financial KPIs (ticket médio, oportunidade perdida, recuperável com IA) — v0
- ✓ Blueprint JSON salvo em la_blueprints para Sofia consumir — v0
- ✓ Persistência completa no Supabase (mensagens, análises, relatório HTML) — v0
- ✓ Knowledge Consolidator offline mode (sem Supabase) — v0

### Active

<!-- Current scope. Building toward these. -->

- [ ] Conectar ao Evolution API para sincronizar conversas diretamente (sem upload de arquivo)
- [ ] N8N chama webhook após sync → altera flag em tabela de controle
- [ ] Frontend monitora flag via polling e inicia análise automaticamente
- [ ] Pipeline de análise adaptado para ler mensagens do Evolution API em vez de Archive.zip

### Out of Scope

- Upload manual de Archive.zip — substituído pela integração Evolution API neste milestone
- Modo offline sem Supabase para o fluxo principal — relevante apenas para testes

## Context

- **Stack atual:** Python 3.11, FastAPI, DSPy, Supabase (PostgreSQL + pgvector), OpenAI/Groq/GLM
- **Evolution API:** Plataforma de automação de WhatsApp Business. Expõe REST API para listar/ler conversas e mensagens.
- **N8N:** Orquestrador de workflows que recebe webhook do Evolution e atualiza flag no Supabase
- **Sofia:** IA de atendimento da EasyScale que consome `la_blueprints` via polling em `sf_clinics`
- **Cliente atual em produção:** Sorriso Da Gente (slug: `sgen`)
- **Archive de chats:** `/Users/jefersonalvarenga/Documents/customer-chats/sgen/chats/Archive.zip`

## Constraints

- **Tech stack:** Python + Supabase — não adicionar novos bancos ou linguagens
- **Evolution API:** Integração via REST (não SDK oficial), autenticação por API key
- **Compatibilidade:** Manter `run_local.py` funcional com Archive.zip para clínicas sem Evolution API
- **Sofia contrato:** `la_blueprints` com `blueprint_json` + `knowledge_base_mapping` + `clinic_id` — não quebrar

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| KC offline first para testes | Permite validar extração sem depender de Supabase em CI | ✓ Good |
| Blueprint salvo em la_blueprints com clinic_id | Sofia faz polling por clinic_id para carregar conhecimento | ✓ Good |
| DSPy para análise semântica | Estrutura declarativa, fácil de testar e swappar modelos | ✓ Good |
| Groq como provider padrão (free tier) | Velocidade + custo zero para desenvolvimento | ✓ Good |

## Current Milestone: v1.0 — Evolution API Integration

**Goal:** Substituir o upload manual de Archive.zip por sincronização automática via Evolution API, com N8N orquestrando o trigger e frontend monitorando via polling.

**Target features:**
- Sincronização de conversas via Evolution API
- Flag de controle no Supabase atualizada pelo N8N
- Frontend polling para detectar mensagens disponíveis e iniciar análise
- Pipeline adaptado para processar mensagens do Evolution diretamente

---
*Last updated: 2026-03-13 — Milestone v1.0 started*
