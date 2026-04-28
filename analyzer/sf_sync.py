"""
sf_sync.py
----------
Sincroniza o blueprint extraído pelo LA (V2, módulo blueprint_v2) para as
tabelas operacionais sf_*. Idempotente — deleta linhas anteriores ligadas
à clínica antes de inserir.

Domínios sincronizados:
  G1 → sf_clinics (perfil), sf_resources (profissionais), sf_specialties,
       sf_clinic_services (serviços + preços), sf_resource_services (mapeamento)
  G2..G5 → sf_assistant_profile

Não toca em sf_clinics.onboarding_status (orquestração é do n8n).
Marca cada domínio como 'auto' em sf_clinics.setup_review pra UI saber
que aquele dado precisa de aprovação humana.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from analyzer.blueprint_v2 import Blueprint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_to_numeric(text: Optional[str]) -> Optional[float]:
    """Extract a numeric price from free-form text. Picks the FIRST number found.

    Examples:
        "R$ 450"                   → 450.0
        "R$ 1.100"                 → 1100.0
        "a partir de R$ 1.200"     → 1200.0
        "entre R$ 2.800 e R$ 3.900"→ 2800.0
        "valor na avaliação"       → None
    """
    if not text:
        return None
    # Find numbers like 1.234,56 or 1234,56 or 1234 (BR style)
    match = re.search(r"(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?|\d+(?:,\d{1,2})?)", text)
    if not match:
        return None
    raw = match.group(1).replace(".", "").replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


PT_TO_EN_DAY = {
    "seg": "mon", "ter": "tue", "qua": "wed", "qui": "thu",
    "sex": "fri", "sab": "sat", "dom": "sun",
}


def _parse_time_range(text: str) -> Optional[tuple[str, str]]:
    """'9h-18h' / '9:00-18:00' / '9h às 18h' → ('09:00','18:00'). None se não der."""
    if not text:
        return None
    m = re.search(r"(\d{1,2})[h:]?(\d{0,2}).*?(\d{1,2})[h:]?(\d{0,2})", text)
    if not m:
        return None
    h1, m1, h2, m2 = m.groups()
    return (
        f"{int(h1):02d}:{(m1 or '00').zfill(2)[:2]}",
        f"{int(h2):02d}:{(m2 or '00').zfill(2)[:2]}",
    )


def _to_week_schedule(hours: dict) -> dict:
    """
    Converte business_hours do blueprint (PT-BR keys, free text) pro shape canônico
    do frontend (EN keys, {enabled, shifts:[{open,close}]}).

    Shape de entrada: {seg:'9h-18h', ter:None, ...}
    Shape de saída:   {mon:{enabled:True, shifts:[{open:'09:00',close:'18:00'}]}, sun:{enabled:False, shifts:[]}, ...}
    """
    out: dict = {
        "sun": {"enabled": False, "shifts": []},
        "mon": {"enabled": False, "shifts": []},
        "tue": {"enabled": False, "shifts": []},
        "wed": {"enabled": False, "shifts": []},
        "thu": {"enabled": False, "shifts": []},
        "fri": {"enabled": False, "shifts": []},
        "sat": {"enabled": False, "shifts": []},
    }
    if not isinstance(hours, dict):
        return out
    for pt_key, en_key in PT_TO_EN_DAY.items():
        text = hours.get(pt_key)
        if not text:
            continue
        rng = _parse_time_range(text)
        if not rng:
            continue
        out[en_key] = {"enabled": True, "shifts": [{"open": rng[0], "close": rng[1]}]}
    return out


def _hours_to_open_close(hours: dict) -> tuple[Optional[str], Optional[str]]:
    """Converts {seg: '9h-18h', ...} → ('09:00', '18:00') taking the most common.

    Hoje: pega só o primeiro dia não-nulo. Tela de horário do dashboard tem
    UI dedicada per-day; aqui só populamos open_time/close_time como hint.
    """
    for day in ("seg", "ter", "qua", "qui", "sex", "sab", "dom"):
        v = hours.get(day) if isinstance(hours, dict) else None
        if not v:
            continue
        # Aceita: "9h-18h", "9:00-18:00", "9h às 18h", "9-18"
        m = re.search(r"(\d{1,2})[h:]?(\d{0,2}).*?(\d{1,2})[h:]?(\d{0,2})", v)
        if not m:
            continue
        h1, m1, h2, m2 = m.groups()
        return (
            f"{int(h1):02d}:{(m1 or '00').zfill(2)[:2]}",
            f"{int(h2):02d}:{(m2 or '00').zfill(2)[:2]}",
        )
    return None, None


# ---------------------------------------------------------------------------
# Domain syncs
# ---------------------------------------------------------------------------

def _sync_clinic_profile(db, clinic_id: str, bp: Blueprint) -> None:
    g1 = bp.g1_identidade
    open_time, close_time = _hours_to_open_close(g1.business_hours.model_dump())

    update_payload: dict = {}
    if g1.clinic_name:
        update_payload["name"] = g1.clinic_name
    if g1.clinic_address:
        update_payload["address"] = g1.clinic_address
    if g1.clinic_neighborhood:
        update_payload["neighborhood"] = g1.clinic_neighborhood
    if open_time:
        update_payload["open_time"] = open_time
    if close_time:
        update_payload["close_time"] = close_time
    if g1.business_hours:
        update_payload["schedule"] = _to_week_schedule(g1.business_hours.model_dump())

    # Payment instructions consolidated (text livre — admin edita)
    payment_lines = []
    if g1.payment_methods:
        payment_lines.append(f"Aceita: {', '.join(g1.payment_methods)}")
    if g1.installments_policy and g1.installments_policy.aceita:
        s = "Parcelamento"
        if g1.installments_policy.max_parcelas:
            s += f" até {g1.installments_policy.max_parcelas}x"
        if g1.installments_policy.juros:
            s += f" ({g1.installments_policy.juros})"
        payment_lines.append(s)
    if g1.discounts_policy:
        d = g1.discounts_policy
        if d.vista_pix:
            payment_lines.append(f"PIX/à vista: {d.vista_pix}")
        if d.primeira_consulta:
            payment_lines.append(f"Primeira consulta: {d.primeira_consulta}")
        if d.indicacao:
            payment_lines.append(f"Indicação: {d.indicacao}")
        if d.pacote_sessoes:
            payment_lines.append(f"Pacote sessões: {d.pacote_sessoes}")
    if payment_lines:
        update_payload["payment_instructions"] = "\n".join(payment_lines)

    # Marca todos os domínios extraídos como 'auto' (precisam aprovação).
    update_payload["setup_review"] = {
        "profile": "auto",
        "business_hours": "auto",
        "payment": "auto",
        "professionals": "auto",
        "services": "auto",
        "assistant_tone": "auto",
    }

    if update_payload:
        db.table("sf_clinics").update(update_payload).eq("id", clinic_id).execute()


def _sync_professionals(db, clinic_id: str, bp: Blueprint) -> dict[str, str]:
    """Insere profissionais em sf_resources. Retorna {nome → resource_id} pra
    ser usado pelo mapeamento em sf_resource_services."""
    # Idempotência: deleta antes de re-inserir.
    db.table("sf_resources").delete().eq("clinic_id", clinic_id).execute()

    name_to_id: dict[str, str] = {}
    for prof in bp.g1_identidade.professionals:
        if not prof.nome.strip():
            continue
        row = {
            "clinic_id": clinic_id,
            "name": prof.nome.strip(),
            "type": "professional",
            "is_active": True,
            "specialty": (prof.especialidades or [None])[0] if prof.especialidades else None,
            "metadata": {
                "titulo": prof.titulo,
                "especialidades": prof.especialidades or [],
            },
            "sofia_enabled": True,
            "offers_specialty": bool(prof.especialidades),
        }
        result = db.table("sf_resources").insert(row).execute()
        if result.data:
            name_to_id[prof.nome.strip().lower()] = result.data[0]["id"]
    return name_to_id


def _sync_specialties(db, clinic_id: str, bp: Blueprint) -> dict[str, str]:
    """Insere especialidades únicas em sf_specialties. Retorna {nome → id}."""
    db.table("sf_specialties").delete().eq("clinic_id", clinic_id).execute()

    seen: set[str] = set()
    name_to_id: dict[str, str] = {}
    for prof in bp.g1_identidade.professionals:
        for spec in prof.especialidades or []:
            key = spec.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            result = (
                db.table("sf_specialties")
                .insert({"clinic_id": clinic_id, "name": spec.strip()})
                .execute()
            )
            if result.data:
                name_to_id[key] = result.data[0]["id"]
    return name_to_id


def _sync_services(db, clinic_id: str, bp: Blueprint, prof_name_to_id: dict[str, str]) -> None:
    """Insere serviços em sf_clinic_services. Cruza com pricing por nome.
    Insere mapeamento profissional↔serviço em sf_resource_services."""
    db.table("sf_resource_services").delete().in_(
        "resource_id", list(prof_name_to_id.values())
    ).execute() if prof_name_to_id else None
    db.table("sf_clinic_services").delete().eq("clinic_id", clinic_id).execute()

    # Index pricing by service name (case-insensitive)
    pricing_idx: dict[str, str] = {
        p.servico.strip().lower(): p.valor_or_faixa
        for p in bp.g1_identidade.service_pricing
    }

    for svc in bp.g1_identidade.services_catalog:
        if not svc.nome.strip():
            continue
        key = svc.nome.strip().lower()
        price_text = pricing_idx.get(key)
        price_numeric = _price_to_numeric(price_text)

        row = {
            "clinic_id": clinic_id,
            "name": svc.nome.strip(),
            "category": svc.categoria,
            "duration_minutes": svc.duracao_min,
            "price": price_numeric,
            "description": price_text if price_text else None,
            "requires_evaluation": False,
            "default_flow": "agendamento",
        }
        result = db.table("sf_clinic_services").insert(row).execute()
        if not result.data:
            continue
        service_id = result.data[0]["id"]

        # Mapeamento profissional ↔ serviço
        for performer in svc.performed_by or []:
            performer_id = prof_name_to_id.get(performer.strip().lower())
            if not performer_id:
                continue
            db.table("sf_resource_services").insert({
                "resource_id": performer_id,
                "service_id": service_id,
                "is_active": True,
            }).execute()


def _sync_assistant_profile(db, clinic_id: str, bp: Blueprint) -> None:
    """Upsert sf_assistant_profile (G2..G5) — 1 row por clínica."""
    g2, g3, g4, g5 = bp.g2_tom_voz, bp.g3_venda, bp.g4_fluxo, bp.g5_conhecimento

    payload = {
        "clinic_id": clinic_id,
        # G2
        "tom_voz": g2.tom_voz,
        "nivel_formalidade": g2.nivel_formalidade,
        "uso_emoji_frequencia": g2.uso_emoji.frequencia,
        "uso_emoji_tipos": g2.uso_emoji.tipos_comuns or [],
        "comprimento_msg_tipico": g2.comprimento_msg_tipico,
        "quebra_de_msg": g2.quebra_de_msg,
        "saudacao_inicial": g2.saudacao_inicial or [],
        "despedida_padrao": g2.despedida_padrao or [],
        # G3
        "politica_preco": g3.politica_preco,
        "momento_revela_preco": g3.momento_revela_preco,
        "educacao_tecnica": g3.educacao_tecnica,
        "qualificacao_tipica": g3.qualificacao_tipica or [],
        "prova_social_uso": g3.prova_social_uso,
        "mencao_profissional": g3.mencao_profissional,
        "politica_sinal": g3.politica_sinal.model_dump() if g3.politica_sinal else {},
        "objecoes_recorrentes": [o.model_dump() for o in g3.objecoes_recorrentes],
        "contraindicacao_policy": g3.contraindicacao_policy.model_dump()
            if g3.contraindicacao_policy else {},
        # G4
        "fluxo_padrao_atendimento": g4.fluxo_padrao_atendimento or [],
        "como_confirma_agendamento": g4.como_confirma_agendamento or "",
        "follow_up_apos_silencio": g4.follow_up_apos_silencio.model_dump()
            if g4.follow_up_apos_silencio else {},
        # G5
        "faq_extraido": [f.model_dump() for f in g5.faq_extraido],
        "procedimentos_explicados": [p.model_dump() for p in g5.procedimentos_explicados],
        "casos_de_escalation": g5.casos_de_escalation or [],
    }

    db.table("sf_assistant_profile").upsert(payload, on_conflict="clinic_id").execute()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def sync_blueprint_to_sf(db, clinic_id: str, blueprint: Blueprint) -> None:
    """
    Auto-migração la_blueprints → sf_*. Idempotente. Rodar no fim do
    pipeline LA, antes de marcar job=done.

    Cada chamada lança exceção em fase específica — deixa subir pro caller
    para que run_analysis marque job=error com mensagem útil.
    """
    logger.info("[sf_sync] Start clinic=%s", clinic_id)

    _sync_clinic_profile(db, clinic_id, blueprint)
    logger.info("[sf_sync] sf_clinics profile updated")

    prof_name_to_id = _sync_professionals(db, clinic_id, blueprint)
    logger.info("[sf_sync] sf_resources synced (%d professionals)", len(prof_name_to_id))

    _sync_specialties(db, clinic_id, blueprint)
    logger.info("[sf_sync] sf_specialties synced")

    _sync_services(db, clinic_id, blueprint, prof_name_to_id)
    logger.info("[sf_sync] sf_clinic_services + sf_resource_services synced")

    _sync_assistant_profile(db, clinic_id, blueprint)
    logger.info("[sf_sync] sf_assistant_profile upserted")

    logger.info("[sf_sync] Done clinic=%s", clinic_id)
