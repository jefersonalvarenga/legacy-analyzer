# tests/conftest.py
"""
Fixtures compartilhadas para todos os testes.

Requer variáveis de ambiente (pode usar .env.test):
  SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY (ou OPENAI_API_KEY)
"""
import os
import pytest
from pathlib import Path

# Carrega .env.test se existir, senão .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env.test", override=False)
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

import dspy


def _build_groq_lm() -> dspy.LM:
    """Instancia LM Groq a partir de GROQ_API_KEY."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY não definido — pulando teste LLM")
    return dspy.LM(
        model="groq/llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.0,
        max_tokens=512,
    )


@pytest.fixture(scope="session", autouse=True)
def configure_dspy_lm():
    """
    Configura dspy.settings.lm globalmente para todos os testes que precisam de LLM.
    Usa Groq (llama-3.1-8b-instant) por ser gratuito e rápido.
    Pula silenciosamente se GROQ_API_KEY não estiver disponível.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return  # testes que precisam de LLM vão falhar individualmente
    lm = dspy.LM(
        model="groq/llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.0,
        max_tokens=512,
    )
    dspy.configure(lm=lm)


SYNTHETIC_ARCHIVE = Path(__file__).parent / "fixtures" / "lumina_synthetic.zip"
TEST_CLIENT_SLUG = "lumina_test"
TEST_CLIENT_NAME = "Lumina Estética Avançada"
TEST_SENDER_NAME = "Lumina Estética Avançada"


@pytest.fixture(scope="session")
def synthetic_archive_path() -> Path:
    """
    Retorna o caminho do arquivo sintético.
    Se não existir, gera com generate_synthetic_archive.py (10 conversas).
    """
    if SYNTHETIC_ARCHIVE.exists():
        return SYNTHETIC_ARCHIVE

    SYNTHETIC_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    import subprocess, sys
    result = subprocess.run(
        [
            sys.executable, "scripts/generate_synthetic_archive.py",
            "--count", "10",
            "--output", str(SYNTHETIC_ARCHIVE),
            "--provider", os.getenv("TEST_LLM_PROVIDER", "groq"),
        ],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Falha ao gerar arquivo sintético:\n{result.stderr}"
    return SYNTHETIC_ARCHIVE
