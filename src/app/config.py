from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-5"
    db_path: str = "data/rag.sqlite3"
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    max_upload_bytes: int = 5_000_000


@lru_cache
def get_settings() -> Settings:
    return Settings()
