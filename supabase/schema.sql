-- ============================================================
-- EasyScale Legacy Analyzer — Supabase Schema
-- Run this in the Supabase SQL Editor
-- ============================================================

-- Enable pgvector extension for semantic search / RAG
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- CLIENTS
-- One row per EasyScale customer (clinic, business, etc.)
-- ============================================================
CREATE TABLE IF NOT EXISTS la_clients (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug        TEXT UNIQUE NOT NULL,           -- e.g. "sgen", "cliente-xyz"
    name        TEXT NOT NULL,                  -- e.g. "Sorriso Da Gente"
    sender_name TEXT,                           -- WhatsApp display name used by the clinic
    config      JSONB NOT NULL DEFAULT '{}',    -- extra per-client config
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- ANALYSIS JOBS
-- Tracks each upload + processing run
-- ============================================================
CREATE TYPE la_job_status AS ENUM ('queued', 'processing', 'done', 'error');

CREATE TABLE IF NOT EXISTS la_analysis_jobs (
    id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id                 UUID NOT NULL REFERENCES la_clients(id) ON DELETE CASCADE,
    status                    la_job_status NOT NULL DEFAULT 'queued',
    progress                  SMALLINT NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    current_step              TEXT,
    file_url                  TEXT,             -- Supabase Storage path
    original_filename         TEXT,
    total_conversations       INT NOT NULL DEFAULT 0,
    processed_conversations   INT NOT NULL DEFAULT 0,
    error_message             TEXT,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- CONVERSATIONS
-- One row per individual chat (i.e. per .txt file inside a zip)
-- ============================================================
CREATE TABLE IF NOT EXISTS la_conversations (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id                  UUID NOT NULL REFERENCES la_analysis_jobs(id) ON DELETE CASCADE,
    client_id               UUID NOT NULL REFERENCES la_clients(id) ON DELETE CASCADE,
    phone                   TEXT,               -- patient phone (may be anonymized)
    message_count           INT NOT NULL DEFAULT 0,
    clinic_message_count    INT NOT NULL DEFAULT 0,
    patient_message_count   INT NOT NULL DEFAULT 0,
    date_start              TIMESTAMPTZ,
    date_end                TIMESTAMPTZ,
    duration_days           INT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- MESSAGES
-- One row per message line inside a conversation
-- ============================================================
CREATE TABLE IF NOT EXISTS la_messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES la_conversations(id) ON DELETE CASCADE,
    client_id       UUID NOT NULL REFERENCES la_clients(id) ON DELETE CASCADE,
    sent_at         TIMESTAMPTZ NOT NULL,
    sender          TEXT NOT NULL,
    sender_type     TEXT NOT NULL CHECK (sender_type IN ('clinic', 'patient', 'system')),
    content         TEXT NOT NULL,
    embedding       VECTOR(1536),               -- text-embedding-3-small
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- CHAT ANALYSES
-- LLM-derived insights per conversation
-- ============================================================
CREATE TABLE IF NOT EXISTS la_chat_analyses (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id             UUID UNIQUE NOT NULL REFERENCES la_conversations(id) ON DELETE CASCADE,
    client_id                   UUID NOT NULL REFERENCES la_clients(id) ON DELETE CASCADE,
    job_id                      UUID NOT NULL REFERENCES la_analysis_jobs(id) ON DELETE CASCADE,

    -- Timing KPIs (seconds)
    avg_response_time_seconds   FLOAT,
    first_response_time_seconds FLOAT,
    max_response_time_seconds   FLOAT,

    -- Volume KPIs
    confirmation_rate           FLOAT,          -- 0.0 – 1.0
    reminders_needed            INT,            -- avg reminders before patient confirmed

    -- Sentiment & Quality
    sentiment_score             FLOAT,          -- -1.0 (negative) to 1.0 (positive)
    quality_score               FLOAT,          -- 0.0 – 10.0
    health_score                FLOAT,          -- 0.0 – 100.0 (composite)

    -- Semantic
    topics                      JSONB,          -- ["confirmação", "reagendamento", ...]
    flags                       JSONB,          -- ["sem_resposta", "reclamação", "erro_info"]
    summary                     TEXT,           -- 2–3 sentence LLM summary

    -- Outcome detection (Phase 2)
    outcome                     TEXT CHECK (outcome IN ('agendado','ghosting','objecao_ativa','pendente','outro')),
    outcome_confidence          FLOAT,          -- 0.0 – 1.0
    outcome_reasoning           TEXT,
    main_objection              TEXT,           -- populated when outcome = objecao_ativa

    -- Embedding of the whole conversation (for RAG / clustering)
    embedding                   VECTOR(1536),

    -- Metadata
    llm_model                   TEXT,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- ANALYSIS REPORTS
-- HTML report generated per job
-- ============================================================
CREATE TABLE IF NOT EXISTS la_analysis_reports (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID UNIQUE NOT NULL REFERENCES la_analysis_jobs(id) ON DELETE CASCADE,
    client_id       UUID NOT NULL REFERENCES la_clients(id) ON DELETE CASCADE,
    html_content    TEXT,
    kpis_summary    JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- TRAINING EXPORTS
-- JSONL / RAG chunk exports per job
-- ============================================================
CREATE TYPE la_export_format AS ENUM ('openai_jsonl', 'anthropic_jsonl', 'rag_chunks');

CREATE TABLE IF NOT EXISTS la_training_exports (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID NOT NULL REFERENCES la_analysis_jobs(id) ON DELETE CASCADE,
    client_id       UUID NOT NULL REFERENCES la_clients(id) ON DELETE CASCADE,
    format          la_export_format NOT NULL,
    file_url        TEXT,
    record_count    INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- IMPLEMENTATION BLUEPRINTS
-- One per job — the "plug & play" agent config for n8n
-- ============================================================
CREATE TABLE IF NOT EXISTS la_blueprints (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID UNIQUE NOT NULL REFERENCES la_analysis_jobs(id) ON DELETE CASCADE,
    client_id       UUID NOT NULL REFERENCES la_clients(id) ON DELETE CASCADE,
    blueprint       JSONB NOT NULL,         -- full blueprint conforming to schema
    file_url        TEXT,                   -- path to blueprint_*.json on disk / storage
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- SHADOW DNA PROFILES
-- Persistent behavioral fingerprint per client (updated each run)
-- ============================================================
CREATE TABLE IF NOT EXISTS la_shadow_dna (
    id                              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id                       UUID UNIQUE NOT NULL REFERENCES la_clients(id) ON DELETE CASCADE,
    job_id                          UUID REFERENCES la_analysis_jobs(id) ON DELETE SET NULL,

    -- Identity
    tone_classification             TEXT,
    personality_traits              JSONB,
    forbidden_terms                 JSONB,
    agent_suggested_name            TEXT,

    -- Examples
    greeting_example                TEXT,
    closing_example                 TEXT,

    -- Handoff
    handoff_keywords                JSONB,
    handoff_situations              JSONB,

    -- Local entities
    local_procedures                JSONB,
    local_insurances                JSONB,
    local_neighborhoods             JSONB,
    local_payment_conditions        JSONB,

    -- Knowledge gaps
    unresolved_queries              JSONB,

    -- Quantitative
    average_response_length_tokens  FLOAT,
    emoji_frequency                 JSONB,
    sentiment_score_distribution    JSONB,
    response_time_metrics           JSONB,
    common_objections               JSONB,
    rag_efficiency_score            FLOAT,

    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- INDEXES
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_la_jobs_client        ON la_analysis_jobs(client_id);
CREATE INDEX IF NOT EXISTS idx_la_jobs_status        ON la_analysis_jobs(status);
CREATE INDEX IF NOT EXISTS idx_la_conversations_job  ON la_conversations(job_id);
CREATE INDEX IF NOT EXISTS idx_la_conversations_cli  ON la_conversations(client_id);
CREATE INDEX IF NOT EXISTS idx_la_messages_conv      ON la_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_la_messages_sent_at   ON la_messages(sent_at);
CREATE INDEX IF NOT EXISTS idx_la_analyses_job       ON la_chat_analyses(job_id);
CREATE INDEX IF NOT EXISTS idx_la_analyses_client    ON la_chat_analyses(client_id);
CREATE INDEX IF NOT EXISTS idx_la_exports_job        ON la_training_exports(job_id);
CREATE INDEX IF NOT EXISTS idx_la_blueprints_job    ON la_blueprints(job_id);
CREATE INDEX IF NOT EXISTS idx_la_blueprints_client ON la_blueprints(client_id);
CREATE INDEX IF NOT EXISTS idx_la_shadow_dna_client ON la_shadow_dna(client_id);
CREATE INDEX IF NOT EXISTS idx_la_analyses_outcome  ON la_chat_analyses(outcome);

-- Vector similarity search indexes (cosine distance)
CREATE INDEX IF NOT EXISTS idx_la_messages_embedding
    ON la_messages USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_la_analyses_embedding
    ON la_chat_analyses USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ============================================================
-- RLS (Row Level Security) — multi-tenant
-- Enable after setting up Supabase Auth users
-- ============================================================
ALTER TABLE la_clients          ENABLE ROW LEVEL SECURITY;
ALTER TABLE la_analysis_jobs    ENABLE ROW LEVEL SECURITY;
ALTER TABLE la_conversations    ENABLE ROW LEVEL SECURITY;
ALTER TABLE la_messages         ENABLE ROW LEVEL SECURITY;
ALTER TABLE la_chat_analyses    ENABLE ROW LEVEL SECURITY;
ALTER TABLE la_analysis_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE la_training_exports ENABLE ROW LEVEL SECURITY;
ALTER TABLE la_blueprints       ENABLE ROW LEVEL SECURITY;
ALTER TABLE la_shadow_dna       ENABLE ROW LEVEL SECURITY;

-- Service role bypass (used by the FastAPI backend with service key)
-- The service key bypasses RLS automatically — no policy needed for it.

-- ============================================================
-- SEED: initial client
-- ============================================================
INSERT INTO la_clients (slug, name, sender_name, config)
VALUES (
    'sgen',
    'Sorriso Da Gente',
    'Sorriso Da Gente',
    '{"whatsapp_export_language": "pt-BR"}'
)
ON CONFLICT (slug) DO NOTHING;
