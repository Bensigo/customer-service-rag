from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, sourced from environment variables.

    The .env file is a development convenience resolved from the process
    CWD — run from the repo root. Deployed environments (Docker, CI) must
    supply real environment variables.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: SecretStr = Field(min_length=1)
    anthropic_model: str = "claude-sonnet-5"
    db_path: str = "data/rag.sqlite3"
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    max_upload_bytes: int = 5_000_000


@lru_cache
def get_settings() -> Settings:
    return Settings()
