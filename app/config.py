"""Environment configuration. Credentials are never read from .env.example."""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", allow_inf_nan=False)
    database_url: str = "postgresql://underwriting:underwriting@localhost:5432/underwriting"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_timeout_seconds: float = Field(default=30, gt=0)
    max_agent_retries: int = Field(default=2, ge=0, le=2)
    retry_backoff_seconds: float = Field(default=0.25, ge=0, le=10)
    min_recommendation_confidence: float = Field(default=0.75, ge=0, le=1)
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    trace_dir: str = ".runs"
    console_logs: Literal["progress", "json", "quiet"] = "progress"
