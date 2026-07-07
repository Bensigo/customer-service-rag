"""End-to-end /ready probe against the real wired lifespan (issue #21).

Drives GET /ready through create_app()'s lifespan, which builds the real
ChunkStore (SQLite), Qdrant VectorStore, and Redis clients. Skipped when
Ollama or Qdrant is unreachable (the lifespan probes Ollama for the embed
dimension on startup), so it runs locally only — CI has Qdrant but no
Ollama. Not added to CI.
"""

import os
import urllib.request
import uuid
from urllib.parse import urlsplit

import httpx2
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.main import create_app

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
MODEL = "nomic-embed-text"

pytestmark = pytest.mark.integration


def _safe_url(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{host}{port}" if host else "(redacted)"


def _ollama_model_available() -> bool:
    try:
        response = httpx2.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        response.raise_for_status()
        names = [entry["name"] for entry in response.json()["models"]]
    except Exception:
        return False
    return any(name == MODEL or name.startswith(f"{MODEL}:") for name in names)


def _qdrant_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{QDRANT_URL}/readyz", timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture
def app_env(tmp_path, monkeypatch, request):
    if not _ollama_model_available():
        pytest.skip(f"Ollama not reachable at {_safe_url(OLLAMA_BASE_URL)} or {MODEL} not pulled")
    if not _qdrant_reachable():
        pytest.skip(f"Qdrant not reachable at {_safe_url(QDRANT_URL)}")

    collection = f"test21_ready_{uuid.uuid4().hex[:12]}"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-used")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "rag.sqlite3"))
    monkeypatch.setenv("QDRANT_URL", QDRANT_URL)
    monkeypatch.setenv("REDIS_URL", REDIS_URL)
    monkeypatch.setenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", MODEL)
    monkeypatch.setenv("QDRANT_COLLECTION", collection)

    def drop_collection() -> None:
        qdrant = QdrantClient(url=QDRANT_URL)
        try:
            if qdrant.collection_exists(collection):
                qdrant.delete_collection(collection)
        finally:
            qdrant.close()

    request.addfinalizer(drop_collection)
    return collection


def test_ready_200_when_all_dependencies_up(app_env):
    with TestClient(create_app()) as client:
        response = client.get("/ready")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ready"
    assert body["dependencies"] == {"sqlite": "ok", "qdrant": "ok", "redis": "ok"}
