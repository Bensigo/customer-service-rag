import os
import sys

import pytest

from app.config import Settings, get_settings

_SETTINGS_FIELDS = {name.lower() for name in Settings.model_fields}


@pytest.fixture(autouse=True)
def hermetic_settings(monkeypatch, tmp_path):
    """Isolate every test from ambient configuration.

    Clears shell env vars in any casing (pydantic-settings matches
    case-insensitively), moves CWD to tmp_path so a developer's repo-root
    .env is never read, and clears the get_settings cache on both sides so
    no test sees another test's Settings.
    """
    for key in list(os.environ):
        if key.lower() in _SETTINGS_FIELDS:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
