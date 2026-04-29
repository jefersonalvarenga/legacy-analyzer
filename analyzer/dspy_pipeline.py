"""
dspy_pipeline.py
----------------
DSPy LM configuration for the Legacy Analyzer V2.

A única função desse módulo é `configure_lm()` — escolhe o provider via
env LLM_PROVIDER e configura `dspy.settings.lm` global. Toda a lógica de
extração mora em `analyzer/blueprint_v2.py`.
"""

from __future__ import annotations

import logging
from typing import Optional

import dspy

from config import get_settings

logger = logging.getLogger(__name__)


def _resolve_model(provider: str, override: Optional[str]) -> str:
    """Pick the model string DSPy will pass to LiteLLM."""
    if override:
        return override
    defaults = {
        "gemini": "gemini/gemini-2.5-flash",
        "openai": "openai/gpt-4o-mini",
        "anthropic": "anthropic/claude-haiku-4-5-20251001",
        "glm": "glm-4-flash",
    }
    return defaults.get(provider, defaults["gemini"])


def configure_lm(force_provider: Optional[str] = None) -> tuple[str, str]:
    """
    Configure dspy.settings.lm based on env (or override).

    Returns (provider, model) actually used. Idempotent — calling twice with
    the same provider rebuilds the LM (cheap, just rebinds settings).

    Raises ValueError if the provider's API key is missing.
    """
    s = get_settings()
    provider = (force_provider or s.llm_provider or "gemini").lower()
    # Use LLM_MODEL override only if it looks consistent with the chosen provider,
    # otherwise fall back to the provider's default.
    override = s.llm_model if (s.llm_model and provider.split("_")[0] in s.llm_model.lower()) else None
    model = _resolve_model(provider, override)

    if provider == "gemini":
        if not s.google_api_key:
            raise ValueError("LLM_PROVIDER=gemini but GOOGLE_API_KEY is not set.")
        lm = dspy.LM(model=model, api_key=s.google_api_key, max_tokens=16000)

    elif provider == "openai":
        if not s.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set.")
        lm = dspy.LM(model=model, api_key=s.openai_api_key, max_tokens=16000)

    elif provider == "anthropic":
        if not s.anthropic_api_key:
            raise ValueError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set.")
        lm = dspy.LM(model=model, api_key=s.anthropic_api_key, max_tokens=16000)

    elif provider == "glm":
        if not s.glm_api_key:
            raise ValueError("LLM_PROVIDER=glm but GLM_API_KEY is not set.")
        lm = dspy.LM(
            model=model,
            api_key=s.glm_api_key,
            api_base=s.openai_base_url or "https://open.bigmodel.cn/api/paas/v4/",
            max_tokens=16000,
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER='{provider}'. Use gemini|openai|anthropic|glm.")

    dspy.configure(lm=lm)
    logger.info("[dspy] LM configured: provider=%s model=%s", provider, model)
    return provider, model
