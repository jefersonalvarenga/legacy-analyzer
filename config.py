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

    # Models
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    llm_model: str = Field("gpt-4o-mini", env="LLM_MODEL")
    llm_model_heavy: str = Field("gpt-4o", env="LLM_MODEL_HEAVY")
    # Consolidation LM: reviewer model for KnowledgeConsolidator final pass
    # Set to "anthropic/claude-sonnet-4-6" when ANTHROPIC_API_KEY is set
    # Falls back to llm_model_heavy if not configured
    llm_model_consolidator: str = Field(
        "anthropic/claude-sonnet-4-6", env="LLM_MODEL_CONSOLIDATOR"
    )

    # Report output
    reports_output_dir: str = Field("./legacy-analyzer", env="REPORTS_OUTPUT_DIR")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


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
