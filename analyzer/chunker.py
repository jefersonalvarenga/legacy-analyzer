"""
chunker.py
----------
Chunking + merging para o pipeline LA V2.

Quando uma clínica tem mais conversas do que cabe num único call do Gemini,
divide em chunks de N conversas inteiras (sem quebrar conversa no meio),
extrai blueprint parcial pra cada um e mergeia no final.

Estratégia de merge:
  - Listas (services, professionals, FAQ, objeções, etc): união deduplicada
    por nome/chave (case-insensitive).
  - Strings categóricas (tom_voz, politica_preco, etc): voto majoritário
    (modo) entre os chunks. Empate → primeiro valor encontrado.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Callable, Optional

from analyzer.blueprint_v2 import (
    Blueprint,
    BusinessHours,
    ContraindicacaoPolicy,
    DiscountsPolicy,
    FaqItem,
    FollowUpApsSilencio,
    G1Identidade,
    G2TomVoz,
    G3Venda,
    G4Fluxo,
    G5Conhecimento,
    G6InteligenciaComercial,
    InstallmentsPolicy,
    Objecao,
    OrigemPacienteDistribuicao,
    PoliticaSinal,
    ProcedimentoExplicado,
    Professional,
    ServiceItem,
    ServicePrice,
    UsoEmoji,
    extract_blueprint,
)
from analyzer.parser import Conversation

logger = logging.getLogger(__name__)


DEFAULT_MAX_CONVS_PER_CHUNK = 1000


def chunk_conversations(
    conversations: list[Conversation],
    max_per_chunk: int = DEFAULT_MAX_CONVS_PER_CHUNK,
) -> list[list[Conversation]]:
    """
    Divide conversas em chunks de no máximo N conversas. Conversa nunca é
    dividida no meio (chunk = lista de conversas inteiras).
    """
    if not conversations:
        return []
    chunks: list[list[Conversation]] = []
    for i in range(0, len(conversations), max_per_chunk):
        chunks.append(conversations[i : i + max_per_chunk])
    return chunks


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def _mode(values: list[Optional[str]]) -> Optional[str]:
    """Valor mais frequente, ignorando None/vazio. Empate → primeiro."""
    cleaned = [v for v in values if v]
    if not cleaned:
        return None
    counter = Counter(cleaned)
    return counter.most_common(1)[0][0]


def _dedupe_by_key(items: list, key_fn) -> list:
    """Mantém ordem de primeira aparição, dedupe por chave (case-insensitive)."""
    seen: set[str] = set()
    out: list = []
    for item in items:
        k = key_fn(item)
        if not k:
            continue
        norm = str(k).strip().lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(item)
    return out


def _merge_business_hours(parts: list[BusinessHours]) -> BusinessHours:
    """Para cada dia, pega o primeiro valor não-nulo entre os chunks."""
    out = BusinessHours()
    for day in ("seg", "ter", "qua", "qui", "sex", "sab", "dom"):
        for part in parts:
            v = getattr(part, day, None)
            if v:
                setattr(out, day, v)
                break
    return out


def _merge_g1(g1s: list[G1Identidade]) -> G1Identidade:
    return G1Identidade(
        clinic_name=next((g.clinic_name for g in g1s if g.clinic_name), ""),
        clinic_address=next((g.clinic_address for g in g1s if g.clinic_address), None),
        clinic_neighborhood=next((g.clinic_neighborhood for g in g1s if g.clinic_neighborhood), None),
        business_hours=_merge_business_hours([g.business_hours for g in g1s]),
        professionals=_dedupe_by_key(
            [p for g in g1s for p in g.professionals],
            lambda p: p.nome,
        ),
        services_catalog=_dedupe_by_key(
            [s for g in g1s for s in g.services_catalog],
            lambda s: s.nome,
        ),
        service_pricing=_dedupe_by_key(
            [p for g in g1s for p in g.service_pricing],
            lambda p: p.servico,
        ),
        payment_methods=list({m for g in g1s for m in g.payment_methods}),
        installments_policy=next(
            (g.installments_policy for g in g1s if g.installments_policy.aceita),
            InstallmentsPolicy(),
        ),
        discounts_policy=DiscountsPolicy(
            vista_pix=next((g.discounts_policy.vista_pix for g in g1s if g.discounts_policy.vista_pix), None),
            primeira_consulta=next((g.discounts_policy.primeira_consulta for g in g1s if g.discounts_policy.primeira_consulta), None),
            indicacao=next((g.discounts_policy.indicacao for g in g1s if g.discounts_policy.indicacao), None),
            pacote_sessoes=next((g.discounts_policy.pacote_sessoes for g in g1s if g.discounts_policy.pacote_sessoes), None),
        ),
    )


def _merge_g2(g2s: list[G2TomVoz]) -> G2TomVoz:
    # Listas: união deduplicada
    saudacoes = _dedupe_by_key([s for g in g2s for s in g.saudacao_inicial], lambda s: s)
    despedidas = _dedupe_by_key([s for g in g2s for s in g.despedida_padrao], lambda s: s)
    emoji_tipos = list({t for g in g2s for t in g.uso_emoji.tipos_comuns})
    return G2TomVoz(
        tom_voz=_mode([g.tom_voz for g in g2s]) or "cordial_amigavel",
        nivel_formalidade=_mode([g.nivel_formalidade for g in g2s]) or "voce",
        uso_emoji=UsoEmoji(
            frequencia=_mode([g.uso_emoji.frequencia for g in g2s]) or "media",
            tipos_comuns=emoji_tipos,
        ),
        comprimento_msg_tipico=_mode([g.comprimento_msg_tipico for g in g2s]) or "medio_explicativo",
        quebra_de_msg=_mode([g.quebra_de_msg for g in g2s]) or "mix",
        saudacao_inicial=saudacoes[:8],
        despedida_padrao=despedidas[:8],
    )


def _merge_g3(g3s: list[G3Venda]) -> G3Venda:
    qualif = _dedupe_by_key([q for g in g3s for q in g.qualificacao_tipica], lambda q: q)
    objecoes = _dedupe_by_key(
        [o for g in g3s for o in g.objecoes_recorrentes],
        lambda o: o.objecao,
    )
    # politica_sinal: pega o primeiro que tenha usa_sinal=True; senão default.
    sinal = next(
        (g.politica_sinal for g in g3s if g.politica_sinal.usa_sinal),
        PoliticaSinal(),
    )
    # contraindicacao: modo
    cps = [g.contraindicacao_policy for g in g3s if g.contraindicacao_policy]
    cp_default = ContraindicacaoPolicy()
    cp = ContraindicacaoPolicy(
        deteccao=_mode([c.deteccao for c in cps]) or cp_default.deteccao,
        acao=_mode([c.acao for c in cps]) or cp_default.acao,
    )
    return G3Venda(
        politica_preco=_mode([g.politica_preco for g in g3s]) or "mix",
        momento_revela_preco=_mode([g.momento_revela_preco for g in g3s]) or "apos_qualificacao",
        educacao_tecnica=_mode([g.educacao_tecnica for g in g3s]) or "mix",
        qualificacao_tipica=qualif,
        prova_social_uso=_mode([g.prova_social_uso for g in g3s]) or "sob_demanda",
        mencao_profissional=_mode([g.mencao_profissional for g in g3s]) or "nao_nomeia",
        politica_sinal=sinal,
        objecoes_recorrentes=objecoes,
        contraindicacao_policy=cp,
    )


def _merge_g4(g4s: list[G4Fluxo]) -> G4Fluxo:
    fluxos = [f for g in g4s for f in g.fluxo_padrao_atendimento]
    confirma = next(
        (g.como_confirma_agendamento for g in g4s if g.como_confirma_agendamento),
        "",
    )
    follow_ups = [g.follow_up_apos_silencio for g in g4s]
    fu = FollowUpApsSilencio(
        tenta_quantas_vezes=max((f.tenta_quantas_vezes for f in follow_ups), default=0),
        intervalo_horas=next((f.intervalo_horas for f in follow_ups if f.intervalo_horas), None),
        tom=_mode([f.tom for f in follow_ups]),
    )
    return G4Fluxo(
        fluxo_padrao_atendimento=list(dict.fromkeys(fluxos)),  # dedupe preservando ordem
        como_confirma_agendamento=confirma,
        follow_up_apos_silencio=fu,
    )


def _merge_g5(g5s: list[G5Conhecimento]) -> G5Conhecimento:
    return G5Conhecimento(
        faq_extraido=_dedupe_by_key(
            [f for g in g5s for f in g.faq_extraido],
            lambda f: f.pergunta_padrao,
        ),
        procedimentos_explicados=_dedupe_by_key(
            [p for g in g5s for p in g.procedimentos_explicados],
            lambda p: p.procedimento,
        ),
        casos_de_escalation=_dedupe_by_key(
            [c for g in g5s for c in g.casos_de_escalation],
            lambda c: c,
        ),
    )


def _merge_g6(g6s: list[G6InteligenciaComercial]) -> G6InteligenciaComercial:
    """Média ponderada das origens. Se nenhum chunk tem dado, retorna zeros."""
    keys = ("google_ads", "instagram", "indicacao", "google_organico", "retorno", "outros")
    sums = {k: 0.0 for k in keys}
    n = 0
    for g in g6s:
        d = g.origem_paciente_distribuicao
        if not any(getattr(d, k) for k in keys):
            continue
        for k in keys:
            sums[k] += getattr(d, k)
        n += 1
    if n == 0:
        return G6InteligenciaComercial()
    return G6InteligenciaComercial(
        origem_paciente_distribuicao=OrigemPacienteDistribuicao(
            **{k: sums[k] / n for k in keys}
        )
    )


def merge_blueprints(parts: list[Blueprint]) -> Blueprint:
    """Combina N blueprints parciais num blueprint final."""
    if not parts:
        raise ValueError("merge_blueprints: lista vazia")
    if len(parts) == 1:
        return parts[0]
    return Blueprint(
        g1_identidade=_merge_g1([p.g1_identidade for p in parts]),
        g2_tom_voz=_merge_g2([p.g2_tom_voz for p in parts]),
        g3_venda=_merge_g3([p.g3_venda for p in parts]),
        g4_fluxo=_merge_g4([p.g4_fluxo for p in parts]),
        g5_conhecimento=_merge_g5([p.g5_conhecimento for p in parts]),
        g6_inteligencia_comercial=_merge_g6([p.g6_inteligencia_comercial for p in parts]),
    )


# ---------------------------------------------------------------------------
# Chunked extraction with progress callback
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, int, int], None]
"""(done, total, eta_seconds) — chamado após cada chunk. eta_seconds é
estimativa do tempo restante baseada nos chunks já processados."""


def extract_blueprint_chunked(
    conversations: list[Conversation],
    clinic_name: str,
    *,
    max_per_chunk: int = DEFAULT_MAX_CONVS_PER_CHUNK,
    on_progress: Optional[ProgressCallback] = None,
) -> Blueprint:
    """
    Versão chunked do extract_blueprint. Se o input cabe em 1 chunk, equivale
    a 1 call único. Caso contrário, processa em série e mergeia no final.
    """
    import time

    chunks = chunk_conversations(conversations, max_per_chunk)
    total = len(chunks)
    if total == 0:
        raise ValueError("extract_blueprint_chunked: zero conversas")

    logger.info("[chunker] %d conversations → %d chunk(s)", len(conversations), total)

    # ETA inicial: 35s por chunk (média observada em testes).
    INITIAL_PER_CHUNK = 35
    if on_progress:
        on_progress(0, total, total * INITIAL_PER_CHUNK)

    parts: list[Blueprint] = []
    elapsed_total = 0.0
    for idx, chunk in enumerate(chunks):
        t0 = time.time()
        logger.info(
            "[chunker] chunk %d/%d (%d conversations)",
            idx + 1, total, len(chunk),
        )
        bp = extract_blueprint(chunk, clinic_name)
        parts.append(bp)

        elapsed = time.time() - t0
        elapsed_total += elapsed
        avg_per_chunk = elapsed_total / (idx + 1)
        remaining = total - (idx + 1)
        eta = int(avg_per_chunk * remaining)
        logger.info(
            "[chunker] chunk %d done in %.1fs — eta=%ds",
            idx + 1, elapsed, eta,
        )
        if on_progress:
            on_progress(idx + 1, total, eta)

    if total == 1:
        return parts[0]

    logger.info("[chunker] merging %d partial blueprints", total)
    return merge_blueprints(parts)
