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
            "Inclua titulo quando presente (ex: ['Dra. Ana', 'Dr. Carlos', 'Dr. Marcos']). "
            "Retorne lista vazia [] se nenhum profissional for identificado explicitamente. "
            "NAO invente nomes — inclua apenas os que aparecem no texto."
        )
    )
    schedule_type: str = dspy.OutputField(
        desc=(
            "Tipo de agendamento: "
            "'by_professional' se a clinica agenda por profissional especifico (ex: 'com a Dra. Ana'); "
            "'by_room' se agenda por sala ou consultorio; "
            "'single' se ha apenas um profissional ou se o agendamento nao menciona profissionais. "
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
    raise NotImplementedError


def extract_resources(conversations: list, clinic_name: str) -> ResourcesResult:
    """
    Extract professionals and schedule_type from a corpus of conversations using DSPy.

    Args:
        conversations: list of Conversation objects
        clinic_name:   display name of the clinic (passed to DSPy prompt)

    Returns:
        ResourcesResult with professionals list and schedule_type
    """
    raise NotImplementedError


def count_service_mentions(service_names: list[str], conversations: list) -> list[dict]:
    """
    Count how many clinic messages mention each service name across all conversations.

    Args:
        service_names: list of service/procedure names (from ShadowDNA.local_procedures)
        conversations: list of Conversation objects

    Returns:
        list of dicts [{"name": str, "mention_count": int}] sorted by mention_count DESC
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
