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
