"""
resources_inference.py
----------------------
Infers clinic professionals (la_resources) and services (la_services)
from aggregated WhatsApp conversations.

CALL ORDER DEPENDENCY: infer_and_persist_resources() MUST be called after
extract_shadow_dna() — it reads shadow_dna.local_procedures for SVC-01.
Phase 9 is responsible for enforcing this order.

schedule_type is inferred once at corpus level and stored on EACH professional
row (denormalised to mirror sf_resources schema). If no professionals are
detected, a single 'schedule_config' row carries the clinic-level schedule_type.
"""
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import dspy

logger = logging.getLogger(__name__)

VALID_SCHEDULE_TYPES = {"single", "by_professional", "by_room"}


@dataclass
class ResourcesResult:
    professionals: list[str] = field(default_factory=list)
    schedule_type: str = "single"
    error: Optional[str] = None


class ResourcesSignature(dspy.Signature):
    """
    Analise o conjunto de conversas de WhatsApp de uma clinica odontologica e identifique:
    1. Os profissionais mencionados (dentistas, doutores, especialistas) com seus titulos.
    2. O tipo de agendamento praticado pela clinica.

    Responda em portugues. Baseie-se exclusivamente nos textos fornecidos.
    """

    conversations_sample: str = dspy.InputField(
        desc="Amostra de conversas da clinica (ate 10 conversas, inicio e fim de cada uma)"
    )
    clinic_name: str = dspy.InputField(desc="Nome da clinica")

    professionals: list = dspy.OutputField(
        desc=(
            "Lista de nomes de profissionais detectados nas conversas. "
            "Inclua titulo quando presente (ex: ['Dra. Ana', 'Dr. Carlos']). "
            "Retorne lista vazia [] se nenhum profissional for identificado. "
            "NAO invente nomes — inclua apenas os que aparecem no texto."
        )
    )
    schedule_type: str = dspy.OutputField(
        desc=(
            "Tipo de agendamento: "
            "'by_professional' se a clinica agenda por profissional especifico; "
            "'by_room' se agenda por sala ou consultorio; "
            "'single' se ha apenas um profissional ou agendamento nao menciona profissionais. "
            "Retorne exatamente um dos tres valores: single, by_professional, by_room."
        )
    )


class ResourcesModule(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(ResourcesSignature)

    def forward(self, conversations_sample: str, clinic_name: str):
        return self.predict(
            conversations_sample=conversations_sample,
            clinic_name=clinic_name,
        )


_resources_module: Optional[ResourcesModule] = None


def init_resources_module() -> None:
    """Initialize the DSPy ResourcesModule. Call alongside other init_*() in configure_lm()."""
    global _resources_module
    _resources_module = ResourcesModule()


def _safe_professional_name(item) -> str:
    """Extract a string name from a DSPy output item (str or dict)."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        # Handle common shapes: {"name": "..."}, {"professional": "..."}
        return str(
            item.get("name") or item.get("professional") or next(iter(item.values()), "")
        ).strip()
    return str(item).strip()


def _filter_professionals(raw_list: list) -> list[str]:
    """Deduplicate professionals by lowercase+strip. Preserve original casing of first seen."""
    seen: set[str] = set()
    result: list[str] = []
    for item in raw_list:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _safe_list(value, default: list) -> list:
    """Handle str/list/dict DSPy output shapes. Already handles ast.literal_eval fallback."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        try:
            import ast
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed]
        except Exception:
            pass
        return [v.strip() for v in value.split(",") if v.strip()]
    return default


def extract_resources(conversations: list, clinic_name: str) -> ResourcesResult:
    """
    Extract professionals and schedule_type from a corpus of conversations using DSPy.

    Args:
        conversations: list of Conversation objects
        clinic_name:   display name of the clinic (passed to DSPy prompt)

    Returns:
        ResourcesResult with professionals list and schedule_type
    """
    if not conversations:
        logger.warning("extract_resources(): conversations list is empty — skipping DSPy call.")
        return ResourcesResult(schedule_type="single")

    if not _resources_module:
        return ResourcesResult(error="ResourcesModule not initialized.")

    try:
        from analyzer.shadow_dna import _build_sample
        sample = _build_sample(conversations, max_convs=10)

        pred = _resources_module.forward(conversations_sample=sample, clinic_name=clinic_name)

        # Extract professionals — handle list of str or list of dicts
        raw_professionals = _safe_list(pred.professionals, [])
        named_professionals = [_safe_professional_name(item) for item in raw_professionals]
        professionals = _filter_professionals([p for p in named_professionals if p])

        # Extract and validate schedule_type
        raw_schedule_type = str(pred.schedule_type).strip().lower()
        schedule_type = raw_schedule_type if raw_schedule_type in VALID_SCHEDULE_TYPES else "single"

        return ResourcesResult(professionals=professionals, schedule_type=schedule_type)

    except Exception as e:
        logger.warning("extract_resources() failed: %s", e)
        return ResourcesResult(error=str(e))


def count_service_mentions(service_names: list[str], conversations: list) -> list[dict]:
    """
    Count how many clinic messages mention each service name across all conversations.

    Args:
        service_names: list of service/procedure names (from ShadowDNA.local_procedures)
        conversations: list of Conversation objects

    Returns:
        list of dicts [{"name": str, "mention_count": int}] sorted by mention_count DESC
    """
    if not service_names:
        return []

    # Build corpus from clinic messages only (not patient messages)
    all_clinic_content = [
        msg.content.lower()
        for conv in conversations
        for msg in conv.clinic_messages
    ]

    results = []
    for service in service_names:
        name = service.strip()
        if not name:
            continue
        count = sum(1 for content in all_clinic_content if name.lower() in content)
        results.append({"name": name, "mention_count": count})

    return sorted(results, key=lambda x: x["mention_count"], reverse=True)


def persist_resources(
    db,
    clinic_id: str,
    job_id: str,
    professionals: list[str],
    schedule_type: str,
    services: list[dict],
) -> None:
    """
    Persist resources and services suggestions for a clinic.
    Replaces any previous unconfirmed suggestions (delete confirmed=FALSE, then insert).

    Args:
        db:            Supabase client
        clinic_id:     UUID from sf_clinics
        job_id:        UUID of the analysis job
        professionals: list of professional name strings
        schedule_type: 'single' | 'by_professional' | 'by_room'
        services:      list of dicts [{"name": str, "mention_count": int}]
    """
    # Step 1: Delete unconfirmed resources and services
    db.table("la_resources").delete().eq("clinic_id", clinic_id).eq("confirmed", False).execute()
    db.table("la_services").delete().eq("clinic_id", clinic_id).eq("confirmed", False).execute()

    # Step 2: Build professional rows; if none detected, create a schedule_config row
    if professionals:
        rows = [
            {
                "clinic_id": clinic_id,
                "job_id": job_id,
                "resource_type": "professional",
                "name": name,
                "schedule_type": schedule_type,
                "confirmed": False,
            }
            for name in professionals
        ]
    else:
        rows = [
            {
                "clinic_id": clinic_id,
                "job_id": job_id,
                "resource_type": "schedule_config",
                "name": "default",
                "schedule_type": schedule_type,
                "confirmed": False,
            }
        ]

    # Step 3: Insert resource rows
    db.table("la_resources").insert(rows).execute()

    # Step 4: Insert service rows (if any)
    if services:
        svc_rows = [
            {
                "clinic_id": clinic_id,
                "job_id": job_id,
                "name": svc["name"],
                "mention_count": svc["mention_count"],
                "confirmed": False,
            }
            for svc in services
        ]
        db.table("la_services").insert(svc_rows).execute()


def infer_and_persist_resources(
    conversations: list,
    clinic_name: str,
    clinic_id: str,
    job_id: str,
    shadow_dna,
    db=None,
) -> None:
    """
    Infer resources (professionals, schedule_type) and services from conversations
    and persist as suggestions to la_resources and la_services.

    MUST be called after extract_shadow_dna() — reads shadow_dna.local_procedures.

    Args:
        conversations:  list of Conversation objects (full corpus)
        clinic_name:    display name of the clinic
        clinic_id:      UUID from sf_clinics (for la_resources.clinic_id FK)
        job_id:         UUID of the analysis job (for traceability)
        shadow_dna:     ShadowDNA result — local_procedures feeds SVC-01
        db:             Supabase client (defaults to get_db())
    """
    if not conversations:
        logger.warning(
            "infer_and_persist_resources(): conversations list is empty for clinic %s — skipping.",
            clinic_id,
        )
        return

    if db is None:
        from db import get_db
        db = get_db()

    result = extract_resources(conversations, clinic_name)

    # Guard against None local_procedures
    service_names = (shadow_dna.local_procedures if shadow_dna else None) or []
    services = count_service_mentions(service_names, conversations)

    persist_resources(
        db=db,
        clinic_id=clinic_id,
        job_id=job_id,
        professionals=result.professionals,
        schedule_type=result.schedule_type,
        services=services,
    )
