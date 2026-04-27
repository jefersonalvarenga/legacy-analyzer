from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # Supabase
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_service_key: str = Field(..., env="SUPABASE_SERVICE_KEY")

    # OpenAI — usado exclusivamente para embeddings (text-embedding-3-small)
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")

    # LLM API key — usado para o modelo extrator (LLM_MODEL).
    # Se GLM_API_KEY estiver definido, usa ele. Caso contrário, cai no OPENAI_API_KEY.
    # Isso permite usar GLM-5 para extração mantendo OpenAI só para embeddings.
    glm_api_key: Optional[str] = Field(None, env="GLM_API_KEY")

    @property
    def llm_api_key(self) -> str:
        """API key efetiva para o modelo extrator (GLM_API_KEY > OPENAI_API_KEY)."""
        return self.glm_api_key or self.openai_api_key

    # Application
    app_env: str = Field("development", env="APP_ENV")
    app_port: int = Field(8000, env="APP_PORT")
    app_host: str = Field("0.0.0.0", env="APP_HOST")

    # Worker
    worker_poll_interval: int = Field(5, env="WORKER_POLL_INTERVAL")

    # OpenAI-compatible base URL (for GLM-4, Ollama, etc.)
    # Example for Zhipu GLM-4: https://open.bigmodel.cn/api/paas/v4/
    openai_base_url: Optional[str] = Field(None, env="OPENAI_BASE_URL")

    # Anthropic (used by KnowledgeConsolidator as final reviewer)
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")

    # Google Gemini (optional — used by generate_synthetic_archive.py)
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")

    # Groq (optional — free tier, used for per-conversation analysis at scale)
    groq_api_key: Optional[str] = Field(None, env="GROQ_API_KEY")

    # Models
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")

    # Provider selection for the blueprint extractor.
    # Values: "gemini" (default — free tier viable), "openai", "anthropic", "glm"
    llm_provider: str = Field("gemini", env="LLM_PROVIDER")

    # Default model per provider — overridden by LLM_MODEL env if set.
    llm_model: str = Field("gemini/gemini-2.5-flash", env="LLM_MODEL")

    # Report output
    reports_output_dir: str = Field("./legacy-analyzer", env="REPORTS_OUTPUT_DIR")

    # WhatsApp notification — enviada quando um job de análise é concluído ou falha
    # NOTIFY_WEBHOOK_URL: URL do webhook n8n (ex: https://n8n.easyscale.co/webhook/la-notify)
    # NOTIFY_PHONE: número no formato internacional sem + (ex: 5511999999999)
    notify_webhook_url: Optional[str] = Field(None, env="NOTIFY_WEBHOOK_URL")
    notify_phone: Optional[str] = Field(None, env="NOTIFY_PHONE")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# EasyScale brand colors
BRAND = {
    "primary": "#635BFF",
    "secondary": "#00AFE1",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "dark": "#0F172A",
    "surface": "#1E293B",
    "text": "#F8FAFC",
    "text_muted": "#94A3B8",
}
