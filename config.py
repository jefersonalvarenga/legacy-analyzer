from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # Supabase
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_service_key: str = Field(..., env="SUPABASE_SERVICE_KEY")

    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")

    # Application
    app_env: str = Field("development", env="APP_ENV")
    app_port: int = Field(8000, env="APP_PORT")
    app_host: str = Field("0.0.0.0", env="APP_HOST")

    # Worker
    worker_poll_interval: int = Field(5, env="WORKER_POLL_INTERVAL")

    # Models
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    llm_model: str = Field("gpt-4o-mini", env="LLM_MODEL")
    llm_model_heavy: str = Field("gpt-4o", env="LLM_MODEL_HEAVY")

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
