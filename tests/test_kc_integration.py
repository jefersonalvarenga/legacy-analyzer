# tests/test_kc_integration.py
"""
Testes do Knowledge Consolidator — integração e modo offline.
"""
import pytest
from analyzer.parser import Conversation, Message
from datetime import datetime


def _make_message(content: str, sender_type: str = "clinic") -> Message:
    return Message(
        sent_at=datetime(2025, 1, 1, 9, 0),
        sender="Clínica" if sender_type == "clinic" else "Paciente",
        sender_type=sender_type,
        content=content,
    )


def _make_conv(clinic_msgs: list[str], patient_msgs: list[str] = None) -> Conversation:
    messages = []
    for i, txt in enumerate(clinic_msgs):
        messages.append(_make_message(txt, "clinic"))
    for i, txt in enumerate(patient_msgs or []):
        messages.append(_make_message(txt, "patient"))

    conv = Conversation(source_filename="test_chat.zip", phone="11999990000")
    conv.messages = messages
    return conv


class TestConsolidateKnowledgeOffline:
    def test_extracts_insurance_from_clinic_messages(self):
        """KC offline deve detectar convênios mencionados nas mensagens."""
        from analyzer.knowledge_consolidator import consolidate_knowledge_offline

        convs = [
            _make_conv(["Aceitamos Amil, Uniodonto e Bradesco."]),
            _make_conv(["Trabalhamos com convênio Amil e Sulamerica."]),
        ]

        result = consolidate_knowledge_offline(convs, clinic_name="Clínica Teste")

        assert isinstance(result.confirmed_insurances, list)
        # Amil deve aparecer (mencionado 2x)
        insurance_str = " ".join(result.confirmed_insurances).lower()
        assert "amil" in insurance_str

    def test_extracts_payment_from_clinic_messages(self):
        """KC offline deve detectar formas de pagamento."""
        from analyzer.knowledge_consolidator import consolidate_knowledge_offline

        convs = [
            _make_conv(["Parcelamos em até 12x sem juros no cartão. Pix com 5% de desconto."]),
            _make_conv(["Aceitamos PIX e cartão de crédito."]),
        ]

        result = consolidate_knowledge_offline(convs, clinic_name="Clínica Teste")

        assert isinstance(result.confirmed_payment, list)

    def test_extracts_procedures_from_clinic_messages(self):
        """KC offline deve detectar procedimentos."""
        from analyzer.knowledge_consolidator import consolidate_knowledge_offline

        convs = [
            _make_conv(["Realizamos implante dentário, clareamento e ortodontia."]),
            _make_conv(["Fazemos extração, canal e prótese."]),
        ]

        result = consolidate_knowledge_offline(convs, clinic_name="Clínica Teste")

        assert isinstance(result.confirmed_procedures, list)

    def test_returns_empty_lists_on_empty_conversations(self):
        """KC offline não deve lançar exceção com corpus vazio."""
        from analyzer.knowledge_consolidator import consolidate_knowledge_offline

        result = consolidate_knowledge_offline([], clinic_name="Vazia")

        assert result.confirmed_insurances == []
        assert result.confirmed_procedures == []
        assert result.confirmed_payment == []

    def test_ignores_patient_messages(self):
        """KC offline só deve analisar mensagens da clínica."""
        from analyzer.knowledge_consolidator import consolidate_knowledge_offline

        convs = [
            _make_conv(
                clinic_msgs=["Realizamos implante dentário."],
                patient_msgs=["Fazem limpeza? Aceitam Amil?"],  # perguntas do paciente
            )
        ]

        result = consolidate_knowledge_offline(convs, clinic_name="Clínica Teste")

        # Paciente perguntou "aceitam Amil?" → KC não deve confirmar Amil
        # (só afirmações da clínica valem)
        insurance_str = " ".join(result.confirmed_insurances).lower()
        assert "amil" not in insurance_str, (
            "KC não deve confirmar Amil a partir de pergunta do paciente — "
            f"insurances encontrados: {result.confirmed_insurances}"
        )
        # Implante deve estar (clínica afirmou)
        proc_str = " ".join(result.confirmed_procedures).lower()
        assert "implante" in proc_str, (
            f"Implante afirmado pela clínica deve aparecer nos procedures: {result.confirmed_procedures}"
        )
