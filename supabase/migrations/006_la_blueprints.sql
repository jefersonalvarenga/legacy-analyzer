-- Migration 006: Create la_blueprints table for KnowledgeConsolidator output
-- Stores consolidated clinic knowledge base extracted from corpus-wide message analysis.

CREATE TABLE IF NOT EXISTS public.la_blueprints (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id      UUID NOT NULL REFERENCES public.la_analysis_jobs(id) ON DELETE CASCADE,
  client_id   UUID NOT NULL REFERENCES public.la_clients(id) ON DELETE CASCADE,

  -- Consolidated knowledge base (output of KnowledgeConsolidator Phase 2)
  knowledge_base_mapping JSONB NOT NULL DEFAULT '{}',

  -- Full blueprint JSON (assembled by blueprint.py — populated later)
  blueprint_json JSONB,

  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE(job_id)
);

CREATE INDEX IF NOT EXISTS idx_la_blueprints_client_id ON public.la_blueprints(client_id);
CREATE INDEX IF NOT EXISTS idx_la_blueprints_job_id    ON public.la_blueprints(job_id);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_la_blueprints_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_la_blueprints_updated_at ON public.la_blueprints;
CREATE TRIGGER trg_la_blueprints_updated_at
  BEFORE UPDATE ON public.la_blueprints
  FOR EACH ROW EXECUTE FUNCTION update_la_blueprints_updated_at();
