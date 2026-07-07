import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


def test_settings_defaults():
    settings = Settings(anthropic_api_key="test-key")

    assert settings.anthropic_model == "claude-sonnet-5"
    assert settings.db_path == "data/rag.sqlite3"
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.cache_ttl_seconds == 3600
    assert settings.max_upload_bytes == 5_000_000
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_embed_model == "qwen3-embedding"
    assert settings.ollama_rerank_model == "gemma4"


def test_settings_reads_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("CACHE_TTL_SECONDS", "120")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:11434")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    settings = Settings()

    assert settings.anthropic_api_key.get_secret_value() == "env-key"
    assert settings.anthropic_model == "claude-haiku-4-5-20251001"
    assert settings.cache_ttl_seconds == 120
    assert settings.ollama_base_url == "http://ollama.internal:11434"
    assert settings.ollama_embed_model == "nomic-embed-text"


def test_settings_missing_api_key_raises():
    with pytest.raises(ValidationError):
        Settings()


def test_settings_empty_api_key_raises():
    with pytest.raises(ValidationError):
        Settings(anthropic_api_key="")


def test_settings_repr_does_not_leak_api_key():
    settings = Settings(anthropic_api_key="sk-super-secret")

    assert "sk-super-secret" not in repr(settings)
    assert "sk-super-secret" not in str(settings.model_dump())


def test_get_settings_returns_cached_instance(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "cached-key")

    assert get_settings() is get_settings()
