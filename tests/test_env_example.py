from pathlib import Path

from dotenv import dotenv_values

from app.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_env_example_matches_settings_fields():
    example_keys = set(dotenv_values(REPO_ROOT / ".env.example"))

    assert example_keys == {name.upper() for name in Settings.model_fields}
