"""
blueprint_v2.py
---------------
Legacy Analyzer V2 — pipeline de 1 call. Extrai DNA da clínica em JSON estruturado
seguindo implementation_blueprint_schema.json (G1..G6, ~25 campos).

Tudo numa signature DSPy única. Provider via configure_lm() — Gemini 2.5 Flash
default, fallback OpenAI/Anthropic via env LLM_PROVIDER.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal, Optional

import dspy
from pydantic import BaseModel, Field

from analyzer.parser import Conversation

logger = logging.getLogger(__name__)


# ----- G1 — Identidade objetiva -----

class Professional(BaseModel):
    nome: str
    titulo: Optional[str] = None
    especialidades: list[str] = Field(default_factory=list)


class ServiceItem(BaseModel):
    nome: str
    categoria: Optional[str] = None
    duracao_min: Optional[int] = None
    contraindicacoes_mencionadas: list[str] = Field(default_factory=list)


class ServicePrice(BaseModel):
    servico: str
    valor_or_faixa: str
    condicao: Optional[str] = None


class BusinessHours(BaseModel):
    seg: Optional[str] = None
    ter: Optional[str] = None
    qua: Optional[str] = None
    qui: Optional[str] = None
    sex: Optional[str] = None
    sab: Optional[str] = None
    dom: Optional[str] = None


class InstallmentsPolicy(BaseModel):
    aceita: bool = False
    max_parcelas: Optional[int] = None
    juros: Optional[str] = None


class DiscountsPolicy(BaseModel):
    vista_pix: Optional[str] = None
    primeira_consulta: Optional[str] = None
    indicacao: Optional[str] = None
    pacote_sessoes: Optional[str] = None


class G1Identidade(BaseModel):
    clinic_name: str
    clinic_address: Optional[str] = None
    clinic_neighborhood: Optional[str] = None
    business_hours: BusinessHours = Field(default_factory=BusinessHours)
    professionals: list[Professional] = Field(default_factory=list)
    services_catalog: list[ServiceItem] = Field(default_factory=list)
    service_pricing: list[ServicePrice] = Field(default_factory=list)
    payment_methods: list[
        Literal["pix", "credito", "debito", "dinheiro", "boleto", "transferencia", "wellhub_gympass"]
    ] = Field(default_factory=list)
    installments_policy: InstallmentsPolicy = Field(default_factory=InstallmentsPolicy)
    discounts_policy: DiscountsPolicy = Field(default_factory=DiscountsPolicy)


# ----- G2 — Tom e voz -----

class UsoEmoji(BaseModel):
    frequencia: Literal["alta", "media", "baixa", "zero"]
    tipos_comuns: list[str] = Field(default_factory=list)


class G2TomVoz(BaseModel):
    tom_voz: Literal["formal_clinico", "cordial_amigavel", "marketeiro_alegre", "informal_proximo"]
    nivel_formalidade: Literal["voce", "senhor_senhora", "mix"]
    uso_emoji: UsoEmoji
    comprimento_msg_tipico: Literal["curto_objetivo", "medio_explicativo", "longo_caloroso"]
    quebra_de_msg: Literal["uma_msg_longa", "varias_msgs_curtas", "mix"]
    saudacao_inicial: list[str] = Field(default_factory=list, min_length=1, max_length=8)
    despedida_padrao: list[str] = Field(default_factory=list, min_length=1, max_length=8)


# ----- G3 — Comportamento de venda -----

class PoliticaSinal(BaseModel):
    usa_sinal: bool = False
    valor_ou_percentual: Optional[str] = None
    momento: Optional[str] = None
    abatimento: Optional[str] = None


class Objecao(BaseModel):
    objecao: str
    resposta_padrao_da_clinica: str


class ContraindicacaoPolicy(BaseModel):
    deteccao: Literal["triagem_estruturada", "triagem_leve", "so_avaliacao", "nao_aborda"]
    acao: Literal["escala_avaliacao", "orienta_e_marca", "marca_direto", "recusa_atender"]


class G3Venda(BaseModel):
    politica_preco: Literal["aberto", "faixa", "avaliacao", "sinal"]
    momento_revela_preco: Literal["imediato", "apos_qualificacao", "apos_anamnese", "so_avaliacao"]
    educacao_tecnica: Literal["explica_no_zap", "mix", "guarda_pra_avaliacao"]
    qualificacao_tipica: list[str] = Field(default_factory=list)
    prova_social_uso: Literal["proativa_antes_depois", "sob_demanda", "nao_usa"]
    mencao_profissional: Literal["nomeia_dermato", "nomeia_biomedica", "nao_nomeia", "por_servico"]
    politica_sinal: PoliticaSinal = Field(default_factory=PoliticaSinal)
    objecoes_recorrentes: list[Objecao] = Field(default_factory=list)
    contraindicacao_policy: ContraindicacaoPolicy


# ----- G4 — Fluxo conversacional -----

class FollowUpApsSilencio(BaseModel):
    tenta_quantas_vezes: int = 0
    intervalo_horas: Optional[float] = None
    tom: Optional[str] = None


class G4Fluxo(BaseModel):
    fluxo_padrao_atendimento: list[
        Literal[
            "greeting", "qualifica_area", "qualifica_objetivo", "educa", "preco",
            "agenda", "sinal", "confirmacao", "follow_up", "escala_humano",
        ]
    ] = Field(default_factory=list)
    como_confirma_agendamento: str
    follow_up_apos_silencio: FollowUpApsSilencio = Field(default_factory=FollowUpApsSilencio)


# ----- G5 — Conhecimento operacional -----

class FaqItem(BaseModel):
    pergunta_padrao: str
    resposta_padrao_da_clinica: str


class ProcedimentoExplicado(BaseModel):
    procedimento: str
    explicacao: str
    beneficios_destacados: list[str] = Field(default_factory=list)
    contraindicacoes_mencionadas: list[str] = Field(default_factory=list)


class G5Conhecimento(BaseModel):
    faq_extraido: list[FaqItem] = Field(default_factory=list)
    procedimentos_explicados: list[ProcedimentoExplicado] = Field(default_factory=list)
    casos_de_escalation: list[str] = Field(default_factory=list)


# ----- G6 — Inteligência comercial -----

class OrigemPacienteDistribuicao(BaseModel):
    google_ads: float = 0.0
    instagram: float = 0.0
    indicacao: float = 0.0
    google_organico: float = 0.0
    retorno: float = 0.0
    outros: float = 0.0


class G6InteligenciaComercial(BaseModel):
    origem_paciente_distribuicao: OrigemPacienteDistribuicao = Field(
        default_factory=OrigemPacienteDistribuicao
    )


# ----- Blueprint completo -----

class Blueprint(BaseModel):
    g1_identidade: G1Identidade
    g2_tom_voz: G2TomVoz
    g3_venda: G3Venda
    g4_fluxo: G4Fluxo
    g5_conhecimento: G5Conhecimento
    g6_inteligencia_comercial: G6InteligenciaComercial


# ----- DSPy signature -----

class BlueprintSignature(dspy.Signature):
    """
    Você recebe TODAS as conversas WhatsApp de uma clínica de estética e extrai
    o DNA da clínica em JSON estruturado.

    Princípios:
    - Identidade (G1) é fato observável: copie o que está nas mensagens, não invente.
      Campos com null quando a conversa não menciona.
    - Tom/voz (G2): atendente fala assim em geral? Use exemplos REAIS de saudação e despedida
      extraídos das mensagens (não invente formato).
    - Venda (G3): observe o COMO da clínica vender — política de preço (aberto, faixa, avaliação),
      momento de revelar, prova social, sinal, objeções recorrentes e respostas dadas.
    - Fluxo (G4): a sequência mais comum de etapas que a atendente segue.
    - Conhecimento (G5): perguntas frequentes com respostas REAIS da clínica, procedimentos
      explicados como a clínica explica, gatilhos que escalam pra humano.
    - Comercial (G6): origem do paciente quando dá pra inferir das mensagens iniciais.

    Não opine sobre qualidade. Não dê notas. Apenas catalogue.
    """

    clinic_name: str = dspy.InputField(desc="Nome da clínica como referência.")
    conversations_text: str = dspy.InputField(
        desc="Todas as conversas concatenadas com delimitadores '=== CONVERSA <id> ===' entre elas. "
        "Cada mensagem em formato 'YYYY-MM-DD HH:MM | <clinica|paciente> | <texto>'."
    )

    blueprint: Blueprint = dspy.OutputField(
        desc="JSON estruturado com 6 grupos: g1_identidade, g2_tom_voz, g3_venda, g4_fluxo, "
        "g5_conhecimento, g6_inteligencia_comercial. Siga os enums e tipos exatos do schema."
    )


# ----- Helpers -----

def _conversation_to_text(conv: Conversation) -> str:
    """Renderiza uma conversa em formato compacto pro LLM."""
    lines = []
    for m in conv.messages:
        if m.sender_type == "system":
            continue
        side = "clinica" if m.sender_type == "clinic" else "paciente"
        ts = m.sent_at.strftime("%Y-%m-%d %H:%M")
        text = m.content.replace("\n", " ").strip()
        if text:
            lines.append(f"{ts} | {side} | {text}")
    return "\n".join(lines)


def build_corpus(conversations: list[Conversation]) -> str:
    """Concatena todas as conversas com delimitadores claros."""
    blocks = []
    for i, conv in enumerate(conversations, start=1):
        body = _conversation_to_text(conv)
        if not body:
            continue
        blocks.append(f"=== CONVERSA {i} ({conv.phone}) ===\n{body}")
    return "\n\n".join(blocks)


def extract_blueprint(
    conversations: list[Conversation],
    clinic_name: str,
) -> Blueprint:
    """
    Roda 1 call DSPy com todas as conversas concatenadas e retorna o blueprint.

    Pré-condição: dspy.settings.lm já configurado (via configure_lm).
    Pré-condição: número de conversas cabe no contexto. Chunking entra em Fase 2 se >2500.
    """
    if not conversations:
        raise ValueError("extract_blueprint: lista de conversas vazia.")

    corpus = build_corpus(conversations)
    if not corpus.strip():
        raise ValueError("extract_blueprint: corpus vazio (todas as conversas sem mensagens).")

    extractor = dspy.Predict(BlueprintSignature)
    result = extractor(clinic_name=clinic_name, conversations_text=corpus)
    return result.blueprint


def to_storage_dict(
    bp: Blueprint,
    *,
    clinic_id: str,
    clinic_name: str,
    conversation_count: int,
    message_count: int,
    llm_provider: str,
    llm_model: str,
    analyzer_version: str = "2.0.0",
) -> dict:
    """Serializa Blueprint + metadata pro shape gravado em la_blueprints.blueprint."""
    return {
        "metadata": {
            "clinic_id": clinic_id,
            "clinic_name": clinic_name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "analyzer_version": analyzer_version,
            "conversation_count": conversation_count,
            "message_count": message_count,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
        },
        **bp.model_dump(),
    }
