"""
Teste standalone do KnowledgeConsolidator contra dados existentes da sgen.
Roda direto — não precisa de um job no worker.

Uso:
  python scripts/test_knowledge_consolidator.py            # run completo
  python scripts/test_knowledge_consolidator.py --sample   # 400 msgs (validação rápida)
"""

import sys
import json
import logging
sys.path.insert(0, ".")

QUICK_SAMPLE = "--sample" in sys.argv  # 400 msgs → 5 batches para validação de qualidade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

from config import get_settings
from db import get_db
from analyzer.dspy_pipeline import configure_lm, build_lm
from analyzer.knowledge_consolidator import (
    consolidate_knowledge,
    init_knowledge_modules,
)

SGEN_CLIENT_ID = "4569fed6-7b4a-41f1-918e-ffb1e5ba0403"
SGEN_CLINIC_NAME = "Sorriso Da Gente"

# Fake job_id for standalone test (not saved to la_blueprints)
FAKE_JOB_ID = "00000000-0000-0000-0000-000000000000"


def main():
    settings = get_settings()
    db = get_db()

    logger.info("=== KnowledgeConsolidator — Teste sgen ===")
    logger.info("Extractor LLM : %s", settings.llm_model)
    logger.info("Base URL      : %s", settings.openai_base_url or "(OpenAI default)")
    logger.info("Consolidator  : %s", settings.llm_model_consolidator)
    logger.info("Mode          : %s", "QUICK SAMPLE (400 msgs)" if QUICK_SAMPLE else "FULL RUN")

    # 1. Configure DSPy + build LM instances
    fast_lm, consolidation_lm = configure_lm(
        openai_api_key=settings.llm_api_key,   # GLM_API_KEY se definido, senão OPENAI_API_KEY
        model=settings.llm_model,
        base_url=settings.openai_base_url,
        anthropic_api_key=settings.anthropic_api_key,
        consolidator_model=settings.llm_model_consolidator,
    )
    init_knowledge_modules()

    # 2. Quick message count check
    count_result = (
        db.table("la_messages")
        .select("id", count="exact")
        .eq("client_id", SGEN_CLIENT_ID)
        .eq("sender_type", "clinic")
        .execute()
    )
    total = count_result.count or 0
    logger.info("Clinic messages in DB for sgen: %d", total)

    if total == 0:
        logger.error("No clinic messages found — run a full job first to populate la_messages.")
        sys.exit(1)

    # 3. Run consolidation
    # --sample mode: override MAX_MESSAGES to 400 for quick quality validation
    if QUICK_SAMPLE:
        import analyzer.knowledge_consolidator as kc_mod
        kc_mod.MAX_MESSAGES = 2000  # ~100 conversas × 20 msgs de clínica
        logger.info("Quick sample mode: MAX_MESSAGES overridden to 2000 (~100 conversas)")

    logger.info("Starting consolidation...")
    result = consolidate_knowledge(
        client_id=SGEN_CLIENT_ID,
        clinic_name=SGEN_CLINIC_NAME,
        db=db,
        fast_lm=fast_lm,
        consolidation_lm=consolidation_lm,
    )

    # 4. Print results
    print("\n" + "=" * 60)
    print("RESULTADO — KnowledgeConsolidator (sgen)")
    print("=" * 60)
    print(json.dumps(
        {
            "confirmed_insurances": result.confirmed_insurances,
            "dropped_insurances": result.dropped_insurances,
            "confirmed_address": result.confirmed_address,
            "confirmed_hours": result.confirmed_hours,
            "confirmed_payment": result.confirmed_payment,
            "confirmed_procedures": result.confirmed_procedures,
            "notes": result.notes,
            "raw_insurance_mentions": result.raw_insurance_mentions,
            "error": result.error,
        },
        ensure_ascii=False,
        indent=2,
    ))

    if result.error:
        print(f"\n[AVISO] Erro parcial: {result.error}")
    else:
        print("\n[OK] Consolidação concluída sem erros.")


if __name__ == "__main__":
    main()
