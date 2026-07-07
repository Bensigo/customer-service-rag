"""Live integration test for the Ollama generation client (issue #18).

Hits real gemma4 through ``create_llm_client`` + ``LangChainOllamaClient``
and asserts a grounded prompt yields a non-empty answer — the running
check that ``reasoning=False`` (Ollama ``think:false``) makes the model
answer instead of returning empty hidden-thinking. Auto-skips when Ollama
is unreachable or the model isn't pulled; credentials in OLLAMA_BASE_URL
never reach the skip message.

Tiny by design: one short prompt (gemma4 is slow, ~9.6GB).
"""

import json
import os
import urllib.request
from urllib.parse import urlsplit

import pytest

from app.chat.llm import create_llm_client
from app.config import Settings
from app.models import Message

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4")


def _safe_url(url: str) -> str:
    """scheme://host:port only, dropping any userinfo so credentials in
    OLLAMA_BASE_URL never reach a skip message or CI log."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{host}{port}" if host else "(redacted)"


def _ollama_model_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2) as response:
            names = [entry["name"] for entry in json.load(response)["models"]]
    except Exception:
        return False
    return any(name == OLLAMA_MODEL or name.startswith(f"{OLLAMA_MODEL}:") for name in names)


@pytest.mark.integration
def test_live_gemma4_answers_a_grounded_prompt():
    if not _ollama_model_available():
        pytest.skip(
            f"Ollama not reachable at {_safe_url(OLLAMA_BASE_URL)} or {OLLAMA_MODEL} not pulled"
        )

    settings = Settings(
        llm_provider="ollama",
        ollama_base_url=OLLAMA_BASE_URL,
        ollama_model=OLLAMA_MODEL,
    )
    client = create_llm_client(settings)
    messages = [
        Message(
            role="system",
            content=(
                "Answer only from this source. Source: To reset your password, "
                "open the login page and click 'Forgot password'."
            ),
        ),
        Message(role="user", content="How do I reset my password?"),
    ]

    try:
        answer = client.complete(messages)
    finally:
        client.close()  # release the httpx socket (strict warnings => error)

    assert isinstance(answer, str)
    assert answer.strip(), "gemma4 returned an empty answer (is reasoning/think disabled?)"
