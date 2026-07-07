import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings

SETTINGS_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "DB_PATH",
    "QDRANT_URL",
    "REDIS_URL",
    "CACHE_TTL_SECONDS",
    "MAX_UPLOAD_BYTES",
]


@pytest.fixture(autouse=True)
def clean_settings_env(monkeypatch):
    for var in SETTINGS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_settings_defaults():
    settings = Settings(_env_file=None, anthropic_api_key="test-key")

    assert settings.anthropic_model == "claude-sonnet-5"
    assert settings.db_path == "data/rag.sqlite3"
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.cache_ttl_seconds == 3600
    assert settings.max_upload_bytes == 5_000_000


def test_settings_reads_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("CACHE_TTL_SECONDS", "120")

    settings = Settings(_env_file=None)

    assert settings.anthropic_api_key == "env-key"
    assert settings.anthropic_model == "claude-haiku-4-5-20251001"
    assert settings.cache_ttl_seconds == 120


def test_settings_missing_api_key_raises():
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_get_settings_returns_cached_instance(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "cached-key")
    get_settings.cache_clear()

    assert get_settings() is get_settings()
