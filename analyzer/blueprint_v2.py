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
from typing import Optional

import dspy
from pydantic import BaseModel, Field, model_validator

from analyzer.parser import Conversation

logger = logging.getLogger(__name__)


# ----- G1 — Identidade objetiva -----

class Professional(BaseModel):
    nome: str
    titulo: Optional[str] = None
    especialidades: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value):
        if isinstance(value, str):
            return {"nome": value}
        if isinstance(value, list):
            return {"nome": value[0] if value else ""}
        return value


class ServiceItem(BaseModel):
    nome: str
    categoria: Optional[str] = None
    duracao_min: Optional[int] = None
    # Lista de profissionais que executam esse serviço (nome livre — usar
    # o mesmo nome que aparece em professionals[].nome quando possível).
    performed_by: list[str] = Field(default_factory=list)
    contraindicacoes_mencionadas: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value):
        # Tolera Gemini retornando string solta ou lista no lugar do objeto.
        if isinstance(value, str):
            return {"nome": value}
        if isinstance(value, list):
            return {"nome": value[0] if value else ""}
        return value


class ServicePrice(BaseModel):
    servico: str
    valor_or_faixa: str
    condicao: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value):
        if isinstance(value, str):
            return {"servico": value, "valor_or_faixa": ""}
        if isinstance(value, list):
            return {"servico": value[0] if value else "", "valor_or_faixa": ""}
        return value


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
    # Sugestões: pix | credito | debito | dinheiro | boleto | transferencia | wellhub_gympass.
    # Modelo pode retornar qualquer string — consumer normaliza.
    payment_methods: list[str] = Field(default_factory=list)
    installments_policy: InstallmentsPolicy = Field(default_factory=InstallmentsPolicy)
    discounts_policy: DiscountsPolicy = Field(default_factory=DiscountsPolicy)


# ----- G2 — Tom e voz -----

class UsoEmoji(BaseModel):
    # Sugestões: alta | media | baixa | zero
    frequencia: str = "media"
    tipos_comuns: list[str] = Field(default_factory=list)


class G2TomVoz(BaseModel):
    # tom_voz sugerido: formal_clinico | cordial_amigavel | marketeiro_alegre | informal_proximo
    tom_voz: str = "cordial_amigavel"
    # nivel_formalidade sugerido: voce | senhor_senhora | mix
    nivel_formalidade: str = "voce"
    uso_emoji: UsoEmoji = Field(default_factory=UsoEmoji)
    # comprimento_msg_tipico sugerido: curto_objetivo | medio_explicativo | longo_caloroso
    comprimento_msg_tipico: str = "medio_explicativo"
    # quebra_de_msg sugerido: uma_msg_longa | varias_msgs_curtas | mix
    quebra_de_msg: str = "mix"
    saudacao_inicial: list[str] = Field(default_factory=list)
    despedida_padrao: list[str] = Field(default_factory=list)


# ----- G3 — Comportamento de venda -----

class PoliticaSinal(BaseModel):
    usa_sinal: bool = False
    valor_ou_percentual: Optional[str] = None
    momento: Optional[str] = None
    abatimento: Optional[str] = None


class Objecao(BaseModel):
    objecao: str
    resposta_padrao_da_clinica: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value):
        if isinstance(value, str):
            return {"objecao": value, "resposta_padrao_da_clinica": ""}
        if isinstance(value, list):
            return {"objecao": value[0] if value else "", "resposta_padrao_da_clinica": ""}
        return value


class ContraindicacaoPolicy(BaseModel):
    # deteccao sugerido: triagem_estruturada | triagem_leve | so_avaliacao | nao_aborda
    deteccao: str = "nao_aborda"
    # acao sugerido: escala_avaliacao | orienta_e_marca | marca_direto | recusa_atender
    acao: str = "marca_direto"


class G3Venda(BaseModel):
    # politica_preco sugerido: aberto | faixa | avaliacao | sinal | mix
    politica_preco: str = "mix"
    # momento_revela_preco sugerido: imediato | apos_qualificacao | apos_anamnese | so_avaliacao
    momento_revela_preco: str = "apos_qualificacao"
    # educacao_tecnica sugerido: explica_no_zap | mix | guarda_pra_avaliacao
    educacao_tecnica: str = "mix"
    qualificacao_tipica: list[str] = Field(default_factory=list)
    # prova_social_uso sugerido: proativa_antes_depois | sob_demanda | nao_usa
    prova_social_uso: str = "sob_demanda"
    # mencao_profissional sugerido: nomeia_dermato | nomeia_biomedica | nao_nomeia | por_servico
    mencao_profissional: str = "nao_nomeia"
    politica_sinal: PoliticaSinal = Field(default_factory=PoliticaSinal)
    objecoes_recorrentes: list[Objecao] = Field(default_factory=list)
    contraindicacao_policy: ContraindicacaoPolicy = Field(default_factory=ContraindicacaoPolicy)


# ----- G4 — Fluxo conversacional -----

class FollowUpApsSilencio(BaseModel):
    tenta_quantas_vezes: int = 0
    intervalo_horas: Optional[float] = None
    tom: Optional[str] = None


class G4Fluxo(BaseModel):
    # Sugestões: greeting | qualifica_area | qualifica_objetivo | educa | preco | prova_social |
    # agenda | sinal | confirmacao | follow_up | escala_humano | anamnese.
    fluxo_padrao_atendimento: list[str] = Field(default_factory=list)
    como_confirma_agendamento: str = ""
    follow_up_apos_silencio: FollowUpApsSilencio = Field(default_factory=FollowUpApsSilencio)


# ----- G5 — Conhecimento operacional -----

class FaqItem(BaseModel):
    pergunta_padrao: str
    resposta_padrao_da_clinica: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value):
        if isinstance(value, str):
            return {"pergunta_padrao": value, "resposta_padrao_da_clinica": ""}
        if isinstance(value, list):
            return {"pergunta_padrao": value[0] if value else "", "resposta_padrao_da_clinica": ""}
        return value


class ProcedimentoExplicado(BaseModel):
    procedimento: str
    explicacao: str = ""
    beneficios_destacados: list[str] = Field(default_factory=list)
    contraindicacoes_mencionadas: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value):
        if isinstance(value, str):
            return {"procedimento": value, "explicacao": ""}
        if isinstance(value, list):
            return {"procedimento": value[0] if value else "", "explicacao": ""}
        return value


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
    g2_tom_voz: G2TomVoz = Field(default_factory=G2TomVoz)
    g3_venda: G3Venda = Field(default_factory=G3Venda)
    g4_fluxo: G4Fluxo = Field(default_factory=G4Fluxo)
    g5_conhecimento: G5Conhecimento = Field(default_factory=G5Conhecimento)
    g6_inteligencia_comercial: G6InteligenciaComercial = Field(
        default_factory=G6InteligenciaComercial
    )


# ----- DSPy signature -----

class BlueprintSignature(dspy.Signature):
    """
    Você recebe TODAS as conversas WhatsApp entre pacientes e atendentes de uma
    clínica de estética. Sua tarefa: extrair o DNA da clínica para que um
    assistente de IA replique o padrão de atendimento sem soar genérico.

    OBRIGATÓRIO: o output JSON deve conter os 6 grupos preenchidos
    (g1_identidade, g2_tom_voz, g3_venda, g4_fluxo, g5_conhecimento,
    g6_inteligencia_comercial). NUNCA omita um grupo. Quando a evidência for
    fraca, escolha o valor mais próximo nas sugestões e siga em frente.

    Como preencher cada grupo:

    G1 — IDENTIDADE (fatos): clinic_name, endereço, bairro, business_hours por
    dia, professionals com nome+título+especialidades, services_catalog com
    nome de cada serviço/procedimento mencionado E performed_by (lista de
    profissionais que executam — usar mesmos nomes de professionals[].nome
    quando a conversa diz "a Dra. Ana faz tal procedimento"), service_pricing
    com valores citados (texto livre tipo "R$ 450" ou "a partir de R$ 1.200"),
    payment_methods, installments_policy, discounts_policy. Use null se não
    citado, mas tente preencher tudo o que aparecer ao menos 1x.

    G2 — TOM/VOZ: como a atendente fala?
      tom_voz: "marketeiro_alegre" se há emojis em quase toda msg da
      atendente + frases curtas em rajada + "amor"/"linda"/"corre".
      "cordial_amigavel" se 0–1 emoji ocasional, frases médias, profissional
      mas calorosa. "formal_clinico" se ZERO emoji, "você"/"a senhora",
      menciona Dra. proativa, frases completas. "informal_proximo" pra
      casos sem emoji mas muito coloquial.
      saudacao_inicial e despedida_padrao: 3 a 8 exemplos REAIS extraídos
      das mensagens (não invente — copie literal). Não use mais de 10.

    G3 — VENDA: como vende?
      politica_preco: "aberto" se dá valor exato logo. "sinal" se dá valor +
      pede sinal. "faixa" se "a partir de R$X" / "entre X e Y". "avaliacao"
      se "o valor a gente passa na consulta". "mix" se varia entre serviços.
      educacao_tecnica: "explica_no_zap" se atendente explica procedimento
      no chat. "guarda_pra_avaliacao" se sempre adia. "mix" se varia.
      objecoes_recorrentes: top 3-5 objeções (ex: "tá caro", "vou pensar")
      com a resposta REAL da clínica.
      contraindicacao_policy: como a clínica trata contraindicações
      (gestante, isotretinoína, etc).

    G4 — FLUXO: ordem típica de etapas. Ex: greeting → qualifica_area →
    educa → preco → agenda → sinal → confirmacao. Use as etapas que
    realmente aparecem.
    como_confirma_agendamento: copie um exemplo real de mensagem de
    confirmação da clínica.

    G5 — CONHECIMENTO: faq_extraido (top 5-10 perguntas frequentes com
    resposta REAL da clínica), procedimentos_explicados (como a clínica
    explica cada procedimento), casos_de_escalation (gatilhos que sempre
    vão pra humano).

    G6 — COMERCIAL: origem_paciente_distribuicao em % (some 100). Olhe
    as primeiras mensagens — paciente menciona "vi no instagram", "me
    indicaram", "vi anúncio". Estime distribuição.

    Princípios gerais:
    - Não opine sobre qualidade do atendimento.
    - Use exemplos REAIS quando o campo pede ("saudacao_inicial",
      "objecoes_recorrentes", "como_confirma_agendamento", "faq_extraido"):
      copie literal das mensagens, não invente.
    - Quando incerto, prefira a sugestão mais conservadora a deixar vazio.
    """

    clinic_name: str = dspy.InputField(desc="Nome da clínica como referência.")
    conversations_text: str = dspy.InputField(
        desc="Todas as conversas concatenadas com delimitadores '=== CONVERSA <id> ===' entre elas. "
        "Cada mensagem em formato 'YYYY-MM-DD HH:MM | <clinica|paciente> | <texto>'."
    )

    blueprint: Blueprint = dspy.OutputField(
        desc="Blueprint completo com OS 6 GRUPOS PREENCHIDOS: g1_identidade, "
        "g2_tom_voz, g3_venda, g4_fluxo, g5_conhecimento, g6_inteligencia_comercial."
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
