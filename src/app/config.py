from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, sourced from environment variables.

    The .env file is a development convenience resolved from the process
    CWD — run from the repo root. Deployed environments (Docker, CI) must
    supply real environment variables.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Generation defaults to local Ollama (no API keys). Anthropic is the
    # opt-in "bring your own key" path — its key is required only then.
    llm_provider: Literal["ollama", "anthropic"] = "ollama"
    ollama_model: str = "gemma4"
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-5"
    db_path: str = "data/rag.sqlite3"
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    max_upload_bytes: int = 5_000_000
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "qwen3-embedding"
    ollama_rerank_model: str = "gemma4"
    qdrant_collection: str = "chunks"

    @model_validator(mode="after")
    def _require_anthropic_key_when_selected(self) -> "Settings":
        """The Anthropic path needs a non-empty key; the default Ollama
        path needs none, so embedder-only/eval flows boot with no key."""
        if self.llm_provider == "anthropic":
            key = self.anthropic_api_key
            if key is None or not key.get_secret_value():
                raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
