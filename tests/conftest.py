# tests/conftest.py
"""
Fixtures compartilhadas para todos os testes.

Carrega .env.test (preferido) ou .env do repo.
"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env.test", override=False)
load_dotenv(Path(__file__).parent.parent / ".env", override=False)
