# tests/test_sofia_contract.py
"""
Testes de contrato entre Legacy Analyzer e Sofia.

Sofia consome la_blueprints assim:
  SELECT * FROM la_blueprints WHERE clinic_id = '<uuid>' ORDER BY created_at DESC LIMIT 1

Estes testes garantem que o blueprint salvo satisfaz o contrato.
"""
import json
import pytest


# Campos obrigatórios no blueprint_json (contrato com Sofia)
REQUIRED_BLUEPRINT_KEYS = {
    "metadata",
    "agent_identity",
    "knowledge_base_mapping",
    "conversational_flow",
    "shadow_dna_profile",
    "outcome_summary",
    "financial_kpis",
}

REQUIRED_METADATA_KEYS = {"client_slug", "client_name", "generated_at", "conversation_count"}

REQUIRED_KNOWLEDGE_KEYS = {
    "confirmed_procedures",
    "detected_insurances",
    "payment_methods",
    "unresolved_queries",
}

REQUIRED_AGENT_IDENTITY_KEYS = {"name", "personality_traits", "forbidden_terms"}

REQUIRED_CONVERSATIONAL_FLOW_KEYS = {
    "greeting_style",
    "closing_style",
    "handoff_trigger",
    "attendance_flow",
}


def _load_blueprint_fixture(tmp_path, synthetic_archive_path) -> dict:
    """
    Roda o pipeline completo (sem Supabase, sem embeddings) e retorna o blueprint dict.
    """
    import subprocess
    import sys
    from pathlib import Path

    output_dir = tmp_path / "la_output"
    result = subprocess.run(
        [
            sys.executable, "run_local.py",
            "--archive", str(synthetic_archive_path),
            "--client-slug", "lumina_test",
            "--client-name", "Lumina Estética Avançada",
            "--sender-name", "Lumina Estética Avançada",
            "--output", str(output_dir),
            "--no-embeddings",
            "--no-supabase",
            "--limit", "5",
        ],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"Pipeline falhou:\n{result.stderr}\n{result.stdout}"

    exports_dir = output_dir / "exports"
    blueprint_files = list(exports_dir.glob("blueprint_lumina_test_*.json"))
    assert blueprint_files, f"Nenhum blueprint gerado em {exports_dir}"

    latest = max(blueprint_files, key=lambda p: p.stat().st_mtime)
    return json.loads(latest.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def blueprint(tmp_path_factory, synthetic_archive_path):
    tmp = tmp_path_factory.mktemp("sofia_contract")
    return _load_blueprint_fixture(tmp, synthetic_archive_path)


class TestBlueprintTopLevelKeys:
    def test_has_all_required_top_level_keys(self, blueprint):
        missing = REQUIRED_BLUEPRINT_KEYS - set(blueprint.keys())
        assert not missing, f"Blueprint faltando chaves: {missing}"

    def test_metadata_has_required_fields(self, blueprint):
        missing = REQUIRED_METADATA_KEYS - set(blueprint["metadata"].keys())
        assert not missing, f"metadata faltando campos: {missing}"

    def test_conversation_count_positive(self, blueprint):
        assert blueprint["metadata"]["conversation_count"] > 0

    def test_rag_efficiency_is_percentage(self, blueprint):
        score = blueprint["metadata"].get("rag_efficiency_score", 0)
        assert 0 <= score <= 100, f"rag_efficiency_score fora de range: {score}"


class TestKnowledgeBaseMapping:
    def test_has_all_required_knowledge_keys(self, blueprint):
        kb = blueprint["knowledge_base_mapping"]
        missing = REQUIRED_KNOWLEDGE_KEYS - set(kb.keys())
        assert not missing, f"knowledge_base_mapping faltando: {missing}"

    def test_confirmed_procedures_is_list(self, blueprint):
        procs = blueprint["knowledge_base_mapping"]["confirmed_procedures"]
        assert isinstance(procs, list), "confirmed_procedures deve ser lista"

    def test_detected_insurances_is_list(self, blueprint):
        ins = blueprint["knowledge_base_mapping"]["detected_insurances"]
        assert isinstance(ins, list), "detected_insurances deve ser lista"

    def test_payment_methods_is_list(self, blueprint):
        pm = blueprint["knowledge_base_mapping"]["payment_methods"]
        assert isinstance(pm, list), "payment_methods deve ser lista"

    def test_unresolved_queries_is_list(self, blueprint):
        uq = blueprint["knowledge_base_mapping"]["unresolved_queries"]
        assert isinstance(uq, list), "unresolved_queries deve ser lista"


class TestAgentIdentity:
    def test_has_required_agent_identity_keys(self, blueprint):
        missing = REQUIRED_AGENT_IDENTITY_KEYS - set(blueprint["agent_identity"].keys())
        assert not missing, f"agent_identity faltando: {missing}"

    def test_agent_name_is_nonempty_string(self, blueprint):
        name = blueprint["agent_identity"]["name"]
        assert isinstance(name, str) and name.strip(), "agent_identity.name deve ser string não-vazia"

    def test_personality_traits_is_list(self, blueprint):
        traits = blueprint["agent_identity"]["personality_traits"]
        assert isinstance(traits, list)

    def test_forbidden_terms_is_list(self, blueprint):
        forbidden = blueprint["agent_identity"]["forbidden_terms"]
        assert isinstance(forbidden, list)


class TestConversationalFlow:
    def test_has_required_flow_keys(self, blueprint):
        cf = blueprint["conversational_flow"]
        missing = REQUIRED_CONVERSATIONAL_FLOW_KEYS - set(cf.keys())
        assert not missing, f"conversational_flow faltando: {missing}"

    def test_attendance_flow_is_list(self, blueprint):
        flow = blueprint["conversational_flow"]["attendance_flow"]
        assert isinstance(flow, list), "attendance_flow deve ser lista"

    def test_greeting_style_is_nonempty(self, blueprint):
        gs = blueprint["conversational_flow"]["greeting_style"]
        assert isinstance(gs, dict) and gs, "greeting_style deve ser dict não-vazio"

    def test_handoff_trigger_has_keywords(self, blueprint):
        ht = blueprint["conversational_flow"]["handoff_trigger"]
        assert "keywords" in ht and isinstance(ht["keywords"], list)


class TestOutcomeAndFinancials:
    def test_outcome_summary_has_counts(self, blueprint):
        os_ = blueprint["outcome_summary"]
        for key in ("agendado", "ghosting", "objecao_ativa", "pendente"):
            assert key in os_, f"outcome_summary faltando: {key}"
            assert isinstance(os_[key], int)

    def test_conversion_rate_is_valid(self, blueprint):
        rate = blueprint["outcome_summary"].get("conversion_rate", -1)
        assert 0.0 <= rate <= 1.0, f"conversion_rate inválida: {rate}"

    def test_financial_kpis_have_ticket_medio(self, blueprint):
        fk = blueprint["financial_kpis"]
        assert "ticket_medio" in fk
        assert isinstance(fk["ticket_medio"], (int, float))
        assert fk["ticket_medio"] >= 0


class TestShadowDNAProfile:
    def test_shadow_dna_profile_exists(self, blueprint):
        assert "shadow_dna_profile" in blueprint

    def test_tone_classification_nonempty(self, blueprint):
        tone = blueprint["shadow_dna_profile"].get("tone_classification", "")
        assert tone.strip(), "tone_classification deve ser não-vazio"
