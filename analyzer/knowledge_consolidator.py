"""
knowledge_consolidator.py
--------------------------
Two-phase corpus-wide fact extraction for a clinic's knowledge base.

Architecture: SQL as punching bag → LLM only sees signal, not noise.

Phase 0 — SQL pre-filter (Supabase, zero cost):
  Fetches only messages that mention keywords relevant to each fact category.
  30k messages → ~500-800 relevant messages (15-20x reduction).
  Also fetches temporal bookends (first + last 150 msgs) to catch
  changes over time (e.g. "MetLife descontinuado jan/2026").

Phase 1 — Bulk Extraction (fast/cheap LLM, e.g. GLM-4.7-Flash):
  Processes the ~500-800 filtered messages in batches of BATCH_SIZE.
  Each batch extracts: insurances, address_hints, hours_hints,
  payment_hints, procedure_hints.

Phase 2 — Consolidation (quality LLM, e.g. claude-sonnet-4-6):
  Single call. Deduplicates, normalizes aliases, resolves temporal
  conflicts, produces final structured knowledge base.

Usage:
    from analyzer.knowledge_consolidator import consolidate_knowledge
    result = consolidate_knowledge(
        client_id="4569fed6-...",
        clinic_name="Sorriso Da Gente",
        db=get_db(),
        fast_lm=...,
        consolidation_lm=...,
    )
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import dspy

logger = logging.getLogger(__name__)

# Messages per LLM batch
BATCH_SIZE = 80

# Concurrent LLM batches in Phase 1
# Free tier (GLM-4.7-Flash): 1 — RPM muito baixo
# Paid tier: 6-8
MAX_WORKERS = 1

# Max messages fetched per category SQL query (per pattern chunk)
CATEGORY_LIMIT = 500

# How many ILIKE patterns per Supabase OR query (URL length constraint)
PATTERNS_PER_QUERY = 10

# Temporal bookend: first + last N clinic messages (catches drift over time)
TEMPORAL_BOOKEND_N = 150


# ------------------------------------------------------------------
# SQL keyword patterns per fact category (pt-BR dental clinic)
# ------------------------------------------------------------------

CATEGORY_PATTERNS: dict[str, list[str]] = {
    "insurance": [
        "convênio", "convenio", "plano de saúde", "plano de saude",
        "amil", "uniodonto", "metlife", "met life", "aspmi", "omega",
        "cda", "alma odonto", "alma", "bradesco", "sulamerica", "unimed",
        "hapvida", "odontoprev", "porto seguro", "são cristóvão",
        "aceita", "aceitamos", "não aceitamos",
    ],
    "address": [
        "rua ", "avenida ", "av. ", "r. ", "endereço", "endereco",
        "localiza", "centro", "cep", "bairro", "sala ", "número",
        "fica no", "estamos na", "estamos no",
    ],
    "hours": [
        "horário", "horario", "funcionamos", "atendemos", "atendimento",
        "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
        "sexta-feira", "sábado", "sabado", "domingo",
        "às 8h", "às 9h", "às 18h", "às 17h", "de manhã", "à tarde",
    ],
    "payment": [
        "parcela", "pix", "cartão", "cartao", "dinheiro", "boleto",
        "juros", " vezes", "nota fiscal", "pagamento", "particular",
        "avaliação gratuita", "avaliacao gratuita", "50 reais", "r$",
    ],
    "procedure": [
        "implante", "ortodontia", "aparelho", "clareamento", "canal",
        "extração", "extracao", "prótese", "protese", "faceta",
        "limpeza", "radiografia", "tomografia", "coroa", "cirurgia",
        "restauração", "restauracao", "consulta", "avaliação", "avaliacao",
        "periodontia", "gengiva", "bruxismo", "placa",
    ],
}


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

    # SQL fetch stats per category (for logging/debugging)
    fetch_stats: dict[str, int] = field(default_factory=dict)

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
# SQL pre-filter helpers (Phase 0)
# ------------------------------------------------------------------

def _fetch_category_messages(
    db,
    client_id: str,
    patterns: list[str],
) -> list[dict]:
    """
    Fetch clinic messages matching any of the keyword patterns via ILIKE.
    Splits into chunks of PATTERNS_PER_QUERY to respect URL length limits.
    Returns list of {content, sender, sent_at} dicts, deduped by content.
    """
    seen: set[str] = set()
    results: list[dict] = []

    for i in range(0, len(patterns), PATTERNS_PER_QUERY):
        chunk = patterns[i: i + PATTERNS_PER_QUERY]
        # PostgREST OR filter: "content.ilike.*kw1*,content.ilike.*kw2*,..."
        filter_str = ",".join(f"content.ilike.*{p}*" for p in chunk)
        try:
            page = (
                db.table("la_messages")
                .select("content, sender, sent_at")
                .eq("client_id", client_id)
                .eq("sender_type", "clinic")
                .or_(filter_str)
                .order("sent_at")
                .limit(CATEGORY_LIMIT)
                .execute()
            )
            for row in (page.data or []):
                content = (row.get("content") or "").strip()
                if content and content not in seen:
                    seen.add(content)
                    results.append(row)
        except Exception as e:
            logger.warning("[KC] Category fetch chunk %d failed: %s", i, e)

    return results


def _fetch_temporal_bookends(db, client_id: str, n: int) -> list[dict]:
    """
    Fetch first N + last N clinic messages ordered by time.
    Catches historical facts AND recent changes (e.g. dropped insurances).
    """
    results: list[dict] = []
    seen: set[str] = set()

    # First N (historical baseline)
    try:
        head = (
            db.table("la_messages")
            .select("content, sender, sent_at")
            .eq("client_id", client_id)
            .eq("sender_type", "clinic")
            .order("sent_at", desc=False)
            .limit(n)
            .execute()
        )
        for row in (head.data or []):
            content = (row.get("content") or "").strip()
            if content and content not in seen:
                seen.add(content)
                results.append(row)
    except Exception as e:
        logger.warning("[KC] Temporal head fetch failed: %s", e)

    # Last N (recent — catches drops/changes)
    try:
        tail = (
            db.table("la_messages")
            .select("content, sender, sent_at")
            .eq("client_id", client_id)
            .eq("sender_type", "clinic")
            .order("sent_at", desc=True)
            .limit(n)
            .execute()
        )
        for row in (tail.data or []):
            content = (row.get("content") or "").strip()
            if content and content not in seen:
                seen.add(content)
                results.append(row)
    except Exception as e:
        logger.warning("[KC] Temporal tail fetch failed: %s", e)

    return results


# ------------------------------------------------------------------
# LLM helpers (Phase 1)
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
    lines = []
    for msg in messages:
        sent_at = str(msg.get("sent_at", ""))[:16]
        sender = msg.get("sender", "Clínica")
        content = (msg.get("content") or "").strip()
        if content:
            lines.append(f"[{sent_at}] {sender}: {content}")
    return "\n".join(lines)


def _count_mentions(texts: list[str], items: list[str]) -> dict[str, int]:
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
    Run three-phase knowledge consolidation for a clinic.

    Phase 0: SQL pre-filter — fetches only relevant messages per category
    Phase 1: LLM bulk extraction on filtered set (~500-800 msgs vs 30k)
    Phase 2: Claude consolidation — dedup, normalize, resolve conflicts
    """
    if not _extractor or not _consolidator:
        return ConsolidatedKnowledge(
            error="KnowledgeConsolidator not initialized. Call init_knowledge_modules() first."
        )

    result = ConsolidatedKnowledge()

    # ------------------------------------------------------------------
    # Phase 0: SQL pre-filter per category
    # ------------------------------------------------------------------
    logger.info("[KC] Phase 0 — SQL pre-filter for client %s...", client_id[:8])

    all_messages: list[dict] = []
    seen_content: set[str] = set()
    fetch_stats: dict[str, int] = {}

    def _add_unique(rows: list[dict]):
        added = 0
        for row in rows:
            content = (row.get("content") or "").strip()
            if content and content not in seen_content:
                seen_content.add(content)
                all_messages.append(row)
                added += 1
        return added

    for category, patterns in CATEGORY_PATTERNS.items():
        rows = _fetch_category_messages(db, client_id, patterns)
        added = _add_unique(rows)
        fetch_stats[category] = added
        logger.info("[KC]   %-12s → %d relevant messages fetched (%d unique added)", category, len(rows), added)

    # Temporal bookends (first + last 150 msgs)
    bookend_rows = _fetch_temporal_bookends(db, client_id, TEMPORAL_BOOKEND_N)
    bookend_added = _add_unique(bookend_rows)
    fetch_stats["temporal"] = bookend_added
    logger.info("[KC]   %-12s → %d unique messages added", "temporal", bookend_added)

    result.fetch_stats = fetch_stats

    if not all_messages:
        result.error = "No messages matched any category patterns."
        logger.error("[KC] %s", result.error)
        return result

    # Sort chronologically for coherent batch context
    all_messages.sort(key=lambda m: m.get("sent_at", ""))

    total_msgs = len(all_messages)
    logger.info("[KC] Phase 0 done — %d unique relevant messages (from corpus).", total_msgs)

    # ------------------------------------------------------------------
    # Phase 1: LLM bulk extraction on filtered set
    # ------------------------------------------------------------------
    all_insurances: list[str] = []
    all_address_hints: list[str] = []
    all_hours_hints: list[str] = []
    all_payment_hints: list[str] = []
    all_procedure_hints: list[str] = []

    batches = [all_messages[i: i + BATCH_SIZE] for i in range(0, total_msgs, BATCH_SIZE)]
    total_batches = len(batches)
    logger.info("[KC] Phase 1 — %d batches × %d msgs, %d worker(s)...", total_batches, BATCH_SIZE, MAX_WORKERS)

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
                "insurances":      _safe_list(pred.insurances, []),
                "address_hints":   _safe_list(pred.address_hints, []),
                "hours_hints":     _safe_list(pred.hours_hints, []),
                "payment_hints":   _safe_list(pred.payment_hints, []),
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
            if completed % 5 == 0 or completed == total_batches:
                logger.info("[KC] Phase 1: %d/%d batches done", completed, total_batches)

    # Insurance mention counts across filtered set
    all_content = [m.get("content", "") for m in all_messages]
    unique_insurances = list(dict.fromkeys(all_insurances))
    result.raw_insurance_mentions = _count_mentions(all_content, unique_insurances)

    logger.info(
        "[KC] Phase 1 done — %d insurance names, %d address hints, %d hour hints.",
        len(unique_insurances), len(all_address_hints), len(all_hours_hints),
    )

    # ------------------------------------------------------------------
    # Phase 2: Consolidation (Claude)
    # ------------------------------------------------------------------
    raw_extractions = json.dumps(
        {
            "insurances": all_insurances,
            "insurance_mention_counts": result.raw_insurance_mentions,
            "address_hints": list(dict.fromkeys(all_address_hints)),
            "hours_hints":   list(dict.fromkeys(all_hours_hints)),
            "payment_hints": list(dict.fromkeys(all_payment_hints)),
            "procedure_hints": list(dict.fromkeys(all_procedure_hints)),
        },
        ensure_ascii=False,
        indent=2,
    )

    logger.info("[KC] Phase 2 — consolidation (quality LLM)...")
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
        result.dropped_insurances   = _safe_list(consolidated.dropped_insurances, [])
        result.confirmed_address    = str(consolidated.confirmed_address or "").strip()
        result.confirmed_hours      = _safe_list(consolidated.confirmed_hours, [])
        result.confirmed_payment    = _safe_list(consolidated.confirmed_payment, [])
        result.confirmed_procedures = _safe_list(consolidated.confirmed_procedures, [])
        result.notes                = str(consolidated.notes or "").strip()

    except Exception as e:
        result.error = f"Phase 2 consolidation failed: {e}"
        logger.error("[KC] %s", result.error)
        # Graceful fallback
        result.confirmed_insurances = list(dict.fromkeys(all_insurances))
        result.confirmed_address    = all_address_hints[0] if all_address_hints else ""
        result.confirmed_hours      = list(dict.fromkeys(all_hours_hints))
        result.confirmed_payment    = list(dict.fromkeys(all_payment_hints))
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
    """Persist ConsolidatedKnowledge to la_blueprints."""
    if knowledge.error and not knowledge.confirmed_insurances:
        logger.warning("[KC] Skipping save — extraction had errors and no data.")
        return None

    payload = {
        "confirmed_insurances": knowledge.confirmed_insurances,
        "dropped_insurances":   knowledge.dropped_insurances,
        "confirmed_address":    knowledge.confirmed_address,
        "confirmed_hours":      knowledge.confirmed_hours,
        "confirmed_payment":    knowledge.confirmed_payment,
        "confirmed_procedures": knowledge.confirmed_procedures,
        "notes":                knowledge.notes,
        "raw_insurance_mentions": knowledge.raw_insurance_mentions,
        "fetch_stats":          knowledge.fetch_stats,
        "extractor_error":      knowledge.error,
    }

    try:
        res = (
            db.table("la_blueprints")
            .upsert(
                {"job_id": job_id, "client_id": client_id, "knowledge_base_mapping": payload},
                on_conflict="job_id",
            )
            .execute()
        )
        if res.data:
            row_id = res.data[0].get("id")
            logger.info("[KC] Saved to la_blueprints: %s", row_id)
            return row_id
    except Exception as e:
        logger.error("[KC] Failed to save to la_blueprints: %s", e)

    return None
