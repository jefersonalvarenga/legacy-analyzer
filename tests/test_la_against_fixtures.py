"""
test_la_against_fixtures.py
---------------------------
Manual regression test — run the LA V2 pipeline against the 3 synthetic
fixtures stored in dev_synthetic_fixtures and assert the extracted blueprint
matches the seeded DNA.

Usage:
  python tests/test_la_against_fixtures.py             # run against all 3
  python tests/test_la_against_fixtures.py popular     # one fixture only
  python tests/test_la_against_fixtures.py --keep      # skip Message cleanup at end

Requires env: SUPABASE_URL, SUPABASE_SERVICE_KEY, GOOGLE_API_KEY (Gemini).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# Ensure repo root is importable when run as script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from db import get_db  # noqa: E402
from analyzer.dspy_pipeline import configure_lm  # noqa: E402
from analyzer.evolution_ingestor import ingest_from_evolution  # noqa: E402
from analyzer.blueprint_v2 import extract_blueprint  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
log = logging.getLogger("test-la-fixtures")


# Map fixture profile.dimensions values → expected schema enum values.
# Fixtures use legacy labels; schema uses canonical V2 labels.
TOM_VOZ_MAP = {
    "marketeiro_emoji": ["marketeiro_alegre", "informal_proximo", "cordial_amigavel"],
    "amigavel": ["cordial_amigavel", "informal_proximo"],
    "formal_clinico": ["formal_clinico"],
}

POLITICA_PRECO_MAP = {
    "aberto": ["aberto", "sinal"],
    "aberto_com_sinal": ["sinal", "aberto"],
    "faixa": ["faixa", "mix"],
    "avaliacao": ["avaliacao"],
}

EDUCACAO_TECNICA_MAP = {
    "explica_no_zap": ["explica_no_zap"],
    "mix": ["mix"],
    "guarda_pra_avaliacao": ["guarda_pra_avaliacao"],
}


def _check_in(actual: str, accepted: list[str], label: str) -> bool:
    if actual in accepted:
        log.info("  ✓ %s = %s (esperado: %s)", label, actual, accepted)
        return True
    log.error("  ✗ %s = %s (esperado: %s)", label, actual, accepted)
    return False


def _check_overlap(actual: list[str], expected: list[str], label: str, threshold: float = 0.5) -> bool:
    """≥threshold (default 50%) dos itens esperados precisam aparecer (case/substring tolerant)."""
    a_low = [str(x).lower() for x in actual]
    matched = []
    for exp in expected:
        e_low = str(exp).lower()
        if any(e_low in al or al in e_low for al in a_low):
            matched.append(exp)
    overlap = len(matched) / max(1, len(expected))
    ok = overlap >= threshold
    icon = "✓" if ok else "✗"
    log.info(
        "  %s %s overlap %.0f%% — esperados: %s, matches: %s",
        icon, label, overlap * 100, expected, matched,
    )
    return ok


def _ensure_test_clinic(db, slug: str) -> tuple[str, str]:
    """
    Returns (clinic_id, instance_id).

    Reuses an existing test clinic per slug if present; otherwise creates one
    with an Instance row in the Evolution schema (also synthetic — no real
    WhatsApp connection). Idempotent across runs.
    """
    test_name = f"LA V2 fixture test — {slug}"
    existing = (
        db.table("sf_clinics")
        .select("id, evolution_instance_id")
        .eq("name", test_name)
        .limit(1)
        .execute()
    )
    if existing.data:
        clinic_id = existing.data[0]["id"]
        instance_name = existing.data[0]["evolution_instance_id"]
    else:
        clinic_id = str(uuid.uuid4())
        instance_name = clinic_id
        db.table("sf_clinics").insert({
            "id": clinic_id,
            "name": test_name,
            "evolution_instance_id": instance_name,
            "onboarding_status": "active",
        }).execute()

    instance_row = (
        db.table("Instance")
        .select("id")
        .eq("name", instance_name)
        .limit(1)
        .execute()
    )
    if instance_row.data:
        instance_id = instance_row.data[0]["id"]
    else:
        instance_id = str(uuid.uuid4())
        db.table("Instance").insert({
            "id": instance_id,
            "name": instance_name,
            "connectionStatus": "open",
            "integration": "WHATSAPP-BAILEYS",
        }).execute()

    return clinic_id, instance_id


def _migrate_fixture(db, slug: str, instance_id: str) -> dict:
    """Re-run dev_migrate_fixture_to_message RPC; returns {deleted, inserted}."""
    res = db.rpc("dev_migrate_fixture_to_message", {
        "p_slug": slug, "p_instance_id": instance_id,
    }).execute()
    return res.data[0] if res.data else {}


def _run_one(slug: str, *, keep: bool = False) -> bool:
    db = get_db()

    # Lookup expected DNA from fixture profile.
    fix = (
        db.table("dev_synthetic_fixtures")
        .select("profile, label")
        .eq("slug", slug)
        .single()
        .execute()
    )
    profile = fix.data["profile"]
    expected = profile["dimensions"]

    log.info("=== %s ===", fix.data["label"])

    clinic_id, instance_id = _ensure_test_clinic(db, slug)
    log.info("clinic_id=%s instance_id=%s", clinic_id, instance_id)

    migrated = _migrate_fixture(db, slug, instance_id)
    log.info("migration: %s", migrated)

    log.info("ingesting via evolution_ingestor...")
    convs = ingest_from_evolution(clinic_id, fix.data["label"])
    log.info("ingested %d conversations / %d messages", len(convs), sum(c.message_count for c in convs))

    configure_lm()
    t0 = time.time()
    bp = extract_blueprint(convs, fix.data["label"])
    log.info("extracted in %.1fs", time.time() - t0)

    # AC do MVP — só fatos extraíveis (G1 e G5). Classificações categóricas
    # (tom_voz, politica_preco, educacao_tecnica) são logadas pra inspeção
    # mas não bloqueiam — prompt tuning é iteração separada.
    actual_services = [s.nome for s in bp.g1_identidade.services_catalog]
    ok_serv = _check_overlap(
        actual_services,
        expected["servicos_destaque"],
        "services_catalog vs servicos_destaque",
        threshold=0.5,
    )

    actual_profs = [p.nome for p in bp.g1_identidade.professionals]
    expected_profs = [p["name"] for p in profile.get("professionals", [])]
    ok_prof = _check_overlap(actual_profs, expected_profs, "professionals", threshold=0.5)

    # Fatos adicionais: pricing extraído + FAQ não-vazio
    ok_pricing = len(bp.g1_identidade.service_pricing) >= 1
    log.info(
        "  %s service_pricing entries=%d",
        "✓" if ok_pricing else "✗",
        len(bp.g1_identidade.service_pricing),
    )

    # Informacional — não bloqueia
    log.info("  i tom_voz=%s (fixture seed=%s)", bp.g2_tom_voz.tom_voz, expected.get("tom_voz"))
    log.info(
        "  i politica_preco=%s (fixture seed=%s)",
        bp.g3_venda.politica_preco,
        expected.get("preco_politica"),
    )
    log.info(
        "  i educacao_tecnica=%s (fixture seed=%s)",
        bp.g3_venda.educacao_tecnica,
        expected.get("educacao_tecnica"),
    )

    if not keep:
        # Cleanup synthetic Messages for this instance
        db.table("Message").delete().eq("instanceId", instance_id).eq(
            "status", "dev_synthetic"
        ).execute()

    all_ok = all([ok_serv, ok_prof, ok_pricing])
    log.info("=== %s: %s ===\n", slug, "PASS" if all_ok else "FAIL")
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tier",
        nargs="?",
        choices=["popular", "intermediario", "premium", "all"],
        default="all",
    )
    parser.add_argument("--keep", action="store_true", help="Don't cleanup Messages after run")
    args = parser.parse_args()

    tier_to_slug = {
        "popular": "bella-faces-popular",
        "intermediario": "vivance-intermediario",
        "premium": "renatalins-premium",
    }

    targets = list(tier_to_slug.values()) if args.tier == "all" else [tier_to_slug[args.tier]]

    if not os.getenv("GOOGLE_API_KEY"):
        log.error("GOOGLE_API_KEY missing — set it in .env to run this test.")
        sys.exit(2)

    results = {slug: _run_one(slug, keep=args.keep) for slug in targets}

    print()
    print("Summary:")
    for slug, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {slug}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
