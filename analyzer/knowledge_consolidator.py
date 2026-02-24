"""
knowledge_consolidator.py
--------------------------
Two-phase corpus-wide fact extraction for a clinic's knowledge base.

Phase 1 — Bulk Extraction (fast/cheap LLM, e.g. gpt-4o-mini or GLM-4-Flash):
  Scans ALL clinic messages in batches of BATCH_SIZE.
  Each batch returns: insurances, address_hints, hours_hints, payment_hints,
  procedure_hints.

Phase 2 — Consolidation (quality LLM, e.g. claude-sonnet-4-6):
  Takes the aggregated raw extractions from Phase 1 and produces a final,
  de-duplicated, temporally-resolved knowledge base.
  Handles cases like "MetLife foi descontinuado em jan/2026".

Usage (standalone):
    from analyzer.knowledge_consolidator import consolidate_knowledge
    result = consolidate_knowledge(
        client_id="4569fed6-...",
        clinic_name="Sorriso Da Gente",
        db=get_db(),
        fast_lm=dspy.LM("openai/gpt-4o-mini", api_key=...),
        consolidation_lm=dspy.LM("anthropic/claude-sonnet-4-6", api_key=...),
    )

Called by worker.py after DSPy semantic analysis, before report building.
"""

import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import dspy

logger = logging.getLogger(__name__)

# Number of clinic messages per extraction batch.
BATCH_SIZE = 80

# Max messages to process after dedup + stratified sampling.
# 6000 = ~75 batches. After dedup, covers virtually all unique content.
MAX_MESSAGES = 6000

# Concurrent batches in Phase 1.
# Free tier (GLM-4.7-Flash): use 3 — evita 429s com RPM baixo
# Paid tier (GLM-4.7, GPT-4o-mini): pode usar 8-10
MAX_WORKERS = 3

# Supabase page size for fetching messages
DB_PAGE_SIZE = 1000


# ------------------------------------------------------------------
# DSPy Signatures
# ------------------------------------------------------------------

class FactExtractionSignature(dspy.Signature):
    """
    Analise este lote de mensagens enviadas pela equipe de uma clínica odontológica
    e extraia fatos específicos mencionados. Responda SOMENTE com o que estiver
    explicitamente no texto — não invente fatos. Se não encontrar nada, retorne
    lista vazia para o campo correspondente.
    """
    messages_text: str = dspy.InputField(
        desc=(
            "Lote de mensagens da clínica no formato:\n"
            "[DATA] REMETENTE: mensagem\n"
            "Cada linha é uma mensagem diferente."
        )
    )
    clinic_name: str = dspy.InputField(desc="Nome da clínica")

    insurances: list = dspy.OutputField(
        desc=(
            "Convênios/planos de saúde mencionados (ex: ['Amil', 'Uniodonto']). "
            "Inclua variações do mesmo nome (ex: 'Amil' e 'Amil/Santander' → anote ambas). "
            "Inclua QUALQUER menção, positiva ou negativa (ex: 'não aceitamos X' → anote X)."
        )
    )
    address_hints: list = dspy.OutputField(
        desc=(
            "Fragmentos de endereço mencionados (rua, número, bairro, cidade). "
            "Ex: ['Rua Quinze de Novembro, 916', 'centro de Indaiatuba']. "
            "Retorne lista vazia se não houver nenhum."
        )
    )
    hours_hints: list = dspy.OutputField(
        desc=(
            "Horários de funcionamento mencionados. "
            "Ex: ['Segunda a sexta 9h às 18h', 'Sábado 9h às 12h']. "
            "Retorne lista vazia se não houver nenhum."
        )
    )
    payment_hints: list = dspy.OutputField(
        desc=(
            "Condições de pagamento mencionadas (parcelamento, pix, cartão, etc.). "
            "Ex: ['12x sem juros', 'Pix com desconto']. "
            "Retorne lista vazia se não houver nenhum."
        )
    )
    procedure_hints: list = dspy.OutputField(
        desc=(
            "Procedimentos e serviços específicos mencionados pela clínica. "
            "Ex: ['Implante dentário', 'Clareamento dental', 'Ortodontia']. "
            "Retorne somente o nome do procedimento, sem preço."
        )
    )


class ConsolidationSignature(dspy.Signature):
    """
    Você é um revisor especialista. Recebeu extrações brutas de fatos de uma clínica
    odontológica coletadas de centenas de conversas de WhatsApp. Seu trabalho é produzir
    a base de conhecimento final, consolidada e precisa.

    Regras:
    - Se um fato aparece muitas vezes com variações, normalize para a forma mais completa.
    - Se houver conflito temporal (ex: convênio aceito em 2022, descontinuado em 2026),
      use a informação mais recente e anote em "notes".
    - Para convênios: agrupe aliases (ex: "Amil" e "Amil/Santander" → "Amil/Santander").
    - Remova duplicatas óbvias mas mantenha variantes relevantes.
    - Seja conservador: prefira omitir a inventar.
    """
    raw_extractions: str = dspy.InputField(
        desc="JSON com todas as extrações brutas agrupadas por campo"
    )
    clinic_name: str = dspy.InputField(desc="Nome da clínica")

    confirmed_insurances: list = dspy.OutputField(
        desc=(
            "Lista final de convênios aceitos pela clínica. "
            "Apenas convênios ATUALMENTE aceitos — exclua os descontinuados. "
            "Ex: ['Amil/Santander', 'Uniodonto', 'Alma Odonto', 'ASPMI', 'CDA', 'Omega']"
        )
    )
    dropped_insurances: list = dspy.OutputField(
        desc=(
            "Convênios que já foram aceitos mas foram descontinuados. "
            "Formato: ['NomeConvênio (descontinuado)'] ou lista vazia."
        )
    )
    confirmed_address: str = dspy.OutputField(
        desc=(
            "Endereço completo consolidado da clínica. "
            "Vazio se não houver evidência suficiente."
        )
    )
    confirmed_hours: list = dspy.OutputField(
        desc=(
            "Horários de funcionamento consolidados. "
            "Ex: ['Segunda a sexta: 9h às 18h', 'Sábado: 9h às 12h']"
        )
    )
    confirmed_payment: list = dspy.OutputField(
        desc="Métodos e condições de pagamento confirmados"
    )
    confirmed_procedures: list = dspy.OutputField(
        desc="Lista de procedimentos/serviços mais relevantes desta clínica"
    )
    notes: str = dspy.OutputField(
        desc=(
            "Notas sobre conflitos resolvidos, incertezas ou dados que precisam "
            "de confirmação humana. Ex: 'MetLife descontinuado em jan/2026 conforme "
            "mensagem da coordenadora Rafaela.' Vazio se não houver notas."
        )
    )


# ------------------------------------------------------------------
# DSPy Modules
# ------------------------------------------------------------------

class FactExtractor(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(FactExtractionSignature)

    def forward(self, messages_text: str, clinic_name: str):
        return self.predict(messages_text=messages_text, clinic_name=clinic_name)


class KnowledgeConsolidatorModule(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(ConsolidationSignature)

    def forward(self, raw_extractions: str, clinic_name: str):
        return self.predict(raw_extractions=raw_extractions, clinic_name=clinic_name)


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------

@dataclass
class ConsolidatedKnowledge:
    confirmed_insurances: list[str] = field(default_factory=list)
    dropped_insurances: list[str] = field(default_factory=list)
    confirmed_address: str = ""
    confirmed_hours: list[str] = field(default_factory=list)
    confirmed_payment: list[str] = field(default_factory=list)
    confirmed_procedures: list[str] = field(default_factory=list)
    notes: str = ""

    # Raw aggregated data before consolidation (for debugging)
    raw_insurance_mentions: dict[str, int] = field(default_factory=dict)

    error: Optional[str] = None


# ------------------------------------------------------------------
# Module instances
# ------------------------------------------------------------------

_extractor: Optional[FactExtractor] = None
_consolidator: Optional[KnowledgeConsolidatorModule] = None


def init_knowledge_modules():
    global _extractor, _consolidator
    _extractor = FactExtractor()
    _consolidator = KnowledgeConsolidatorModule()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _safe_list(value, default: list) -> list:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        try:
            import ast
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass
        return [v.strip() for v in value.split(",") if v.strip()]
    return default


def _build_batch_text(messages: list[dict]) -> str:
    """Format a list of DB message rows into a text block for the LLM."""
    lines = []
    for msg in messages:
        sent_at = str(msg.get("sent_at", ""))[:16]  # "2023-05-10 14:32"
        sender = msg.get("sender", "Clínica")
        content = (msg.get("content") or "").strip()
        if content:
            lines.append(f"[{sent_at}] {sender}: {content}")
    return "\n".join(lines)


def _count_mentions(texts: list[str], items: list[str]) -> dict[str, int]:
    """Count how many times each item appears across all text."""
    combined = " ".join(t.lower() for t in texts)
    return {
        item: combined.count(item.lower())
        for item in items
        if combined.count(item.lower()) > 0
    }


# ------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------

def consolidate_knowledge(
    client_id: str,
    clinic_name: str,
    db,
    fast_lm=None,
    consolidation_lm=None,
) -> ConsolidatedKnowledge:
    """
    Run two-phase knowledge consolidation for a clinic.

    Args:
        client_id:         UUID of the client in la_clients
        clinic_name:       Display name for LLM prompts
        db:                Supabase client (from get_db())
        fast_lm:           DSPy LM for bulk extraction (Phase 1)
        consolidation_lm:  DSPy LM for final consolidation (Phase 2)

    Returns:
        ConsolidatedKnowledge dataclass
    """
    if not _extractor or not _consolidator:
        return ConsolidatedKnowledge(
            error="KnowledgeConsolidator not initialized. Call init_knowledge_modules() first."
        )

    result = ConsolidatedKnowledge()

    # ------------------------------------------------------------------
    # Fetch ALL clinic messages from Supabase (paginated)
    # ------------------------------------------------------------------
    logger.info("[KC] Fetching clinic messages for client %s...", client_id[:8])
    messages: list[dict] = []
    try:
        offset = 0
        while True:
            page = (
                db.table("la_messages")
                .select("content, sender, sender_type, sent_at")
                .eq("client_id", client_id)
                .eq("sender_type", "clinic")
                .order("sent_at")
                .range(offset, offset + DB_PAGE_SIZE - 1)
                .execute()
            )
            rows = page.data or []
            messages.extend(rows)
            if len(rows) < DB_PAGE_SIZE:
                break
            offset += DB_PAGE_SIZE
    except Exception as e:
        result.error = f"DB fetch error: {e}"
        logger.error("[KC] %s", result.error)
        return result

    if not messages:
        result.error = "No clinic messages found in la_messages for this client."
        logger.warning("[KC] %s", result.error)
        return result

    total_raw = len(messages)

    # ------------------------------------------------------------------
    # Deduplicate + stratified sample
    # Clinic messages have many exact duplicates (boilerplate greetings, etc.)
    # Stratified sample preserves temporal coverage (catches MetLife-style drops)
    # ------------------------------------------------------------------
    seen_content: set[str] = set()
    unique_messages: list[dict] = []
    for m in messages:
        c = (m.get("content") or "").strip()
        if c and c not in seen_content:
            seen_content.add(c)
            unique_messages.append(m)

    total_unique = len(unique_messages)
    logger.info("[KC] %d total → %d unique messages after dedup.", total_raw, total_unique)

    # Stratified sample: first 20% + last 20% (time-aware) + random middle
    if total_unique > MAX_MESSAGES:
        head_n = MAX_MESSAGES // 5          # 20% from start (historical)
        tail_n = MAX_MESSAGES // 5          # 20% from end (recent changes)
        mid_n  = MAX_MESSAGES - head_n - tail_n
        head   = unique_messages[:head_n]
        tail   = unique_messages[-tail_n:]
        middle = unique_messages[head_n:-tail_n]
        mid_sample = random.sample(middle, min(mid_n, len(middle)))
        sampled = head + mid_sample + tail
        sampled.sort(key=lambda m: m.get("sent_at", ""))  # restore chronological order
        logger.info("[KC] Sampled %d/%d unique messages (stratified).", len(sampled), total_unique)
    else:
        sampled = unique_messages

    total_msgs = len(sampled)
    logger.info("[KC] Processing %d messages in batches of %d.", total_msgs, BATCH_SIZE)

    # ------------------------------------------------------------------
    # Phase 1: Parallel bulk extraction (ThreadPoolExecutor)
    # ------------------------------------------------------------------
    all_insurances: list[str] = []
    all_address_hints: list[str] = []
    all_hours_hints: list[str] = []
    all_payment_hints: list[str] = []
    all_procedure_hints: list[str] = []

    batches = [sampled[i: i + BATCH_SIZE] for i in range(0, total_msgs, BATCH_SIZE)]
    total_batches = len(batches)
    logger.info("[KC] Phase 1: %d batches, %d workers...", total_batches, MAX_WORKERS)

    def _extract_batch(args: tuple) -> tuple[int, dict | None]:
        batch_idx, batch = args
        batch_text = _build_batch_text(batch)
        if not batch_text.strip():
            return batch_idx, None
        try:
            if fast_lm:
                with dspy.context(lm=fast_lm):
                    pred = _extractor(messages_text=batch_text, clinic_name=clinic_name)
            else:
                pred = _extractor(messages_text=batch_text, clinic_name=clinic_name)
            return batch_idx, {
                "insurances":     _safe_list(pred.insurances, []),
                "address_hints":  _safe_list(pred.address_hints, []),
                "hours_hints":    _safe_list(pred.hours_hints, []),
                "payment_hints":  _safe_list(pred.payment_hints, []),
                "procedure_hints": _safe_list(pred.procedure_hints, []),
            }
        except Exception as e:
            logger.warning("[KC] Batch %d/%d failed: %s", batch_idx + 1, total_batches, e)
            return batch_idx, None

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_extract_batch, (i, b)): i
            for i, b in enumerate(batches)
        }
        for future in as_completed(futures):
            _, extraction = future.result()
            completed += 1
            if extraction:
                all_insurances.extend(extraction["insurances"])
                all_address_hints.extend(extraction["address_hints"])
                all_hours_hints.extend(extraction["hours_hints"])
                all_payment_hints.extend(extraction["payment_hints"])
                all_procedure_hints.extend(extraction["procedure_hints"])
            if completed % 10 == 0 or completed == total_batches:
                logger.info("[KC] Phase 1: %d/%d batches done", completed, total_batches)

    # Count insurance mentions across all content for confidence signal
    all_content = [m.get("content", "") for m in messages]
    unique_insurances = list(dict.fromkeys(all_insurances))  # dedupe preserving order
    result.raw_insurance_mentions = _count_mentions(all_content, unique_insurances)

    logger.info(
        "[KC] Phase 1 done. Raw: %d insurance names, %d address hints, %d hour hints.",
        len(unique_insurances), len(all_address_hints), len(all_hours_hints),
    )

    # ------------------------------------------------------------------
    # Phase 2: Consolidation (quality LLM — Claude)
    # ------------------------------------------------------------------
    raw_extractions = json.dumps(
        {
            "insurances": all_insurances,
            "insurance_mention_counts": result.raw_insurance_mentions,
            "address_hints": list(dict.fromkeys(all_address_hints)),
            "hours_hints": list(dict.fromkeys(all_hours_hints)),
            "payment_hints": list(dict.fromkeys(all_payment_hints)),
            "procedure_hints": list(dict.fromkeys(all_procedure_hints)),
        },
        ensure_ascii=False,
        indent=2,
    )

    logger.info("[KC] Running Phase 2 consolidation (quality LLM)...")
    try:
        if consolidation_lm:
            with dspy.context(lm=consolidation_lm):
                consolidated = _consolidator(
                    raw_extractions=raw_extractions,
                    clinic_name=clinic_name,
                )
        else:
            consolidated = _consolidator(
                raw_extractions=raw_extractions,
                clinic_name=clinic_name,
            )

        result.confirmed_insurances = _safe_list(consolidated.confirmed_insurances, [])
        result.dropped_insurances = _safe_list(consolidated.dropped_insurances, [])
        result.confirmed_address = str(consolidated.confirmed_address or "").strip()
        result.confirmed_hours = _safe_list(consolidated.confirmed_hours, [])
        result.confirmed_payment = _safe_list(consolidated.confirmed_payment, [])
        result.confirmed_procedures = _safe_list(consolidated.confirmed_procedures, [])
        result.notes = str(consolidated.notes or "").strip()

    except Exception as e:
        result.error = f"Phase 2 consolidation failed: {e}"
        logger.error("[KC] %s", result.error)
        # Graceful fallback: use raw deduplicated data
        result.confirmed_insurances = list(dict.fromkeys(all_insurances))
        result.confirmed_address = all_address_hints[0] if all_address_hints else ""
        result.confirmed_hours = list(dict.fromkeys(all_hours_hints))
        result.confirmed_payment = list(dict.fromkeys(all_payment_hints))
        result.confirmed_procedures = list(dict.fromkeys(all_procedure_hints))

    logger.info(
        "[KC] Done. Insurances: %s | Address: %s | Hours: %d",
        result.confirmed_insurances,
        result.confirmed_address or "(none)",
        len(result.confirmed_hours),
    )

    return result


# ------------------------------------------------------------------
# Supabase persistence
# ------------------------------------------------------------------

def save_knowledge_to_supabase(
    db,
    job_id: str,
    client_id: str,
    knowledge: ConsolidatedKnowledge,
) -> Optional[str]:
    """
    Persist ConsolidatedKnowledge to la_blueprints (knowledge_base_mapping field).
    Returns the upserted row ID or None on failure.
    """
    if knowledge.error and not knowledge.confirmed_insurances:
        logger.warning("[KC] Skipping Supabase save — extraction had errors and no data.")
        return None

    payload = {
        "confirmed_insurances": knowledge.confirmed_insurances,
        "dropped_insurances": knowledge.dropped_insurances,
        "confirmed_address": knowledge.confirmed_address,
        "confirmed_hours": knowledge.confirmed_hours,
        "confirmed_payment": knowledge.confirmed_payment,
        "confirmed_procedures": knowledge.confirmed_procedures,
        "notes": knowledge.notes,
        "raw_insurance_mentions": knowledge.raw_insurance_mentions,
        "extractor_error": knowledge.error,
    }

    try:
        result = (
            db.table("la_blueprints")
            .upsert(
                {
                    "job_id": job_id,
                    "client_id": client_id,
                    "knowledge_base_mapping": payload,
                },
                on_conflict="job_id",
            )
            .execute()
        )
        if result.data:
            row_id = result.data[0].get("id")
            logger.info("[KC] Saved to la_blueprints: %s", row_id)
            return row_id
    except Exception as e:
        logger.error("[KC] Failed to save to la_blueprints: %s", e)

    return None
