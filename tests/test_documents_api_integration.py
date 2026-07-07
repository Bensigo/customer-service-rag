"""End-to-end upload API test against the real wired lifespan (issue #12).

Drives POST/PUT /documents through create_app()'s lifespan, which builds
the real ChunkStore + Bm25Index (second connection to the same WAL db) +
Ollama Embedder + Qdrant VectorStore. Skipped when Ollama or Qdrant is
unreachable or the embed model is not pulled, so it runs locally only
(CI has Qdrant but no Ollama). Not added to CI.

Each run uses a fresh temp db and a per-run Qdrant collection so repeated
or concurrent runs never collide; the collection is dropped in teardown.
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

# Read at collection time, before hermetic_settings scrubs the env.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
MODEL = "nomic-embed-text"

pytestmark = pytest.mark.integration


def _safe_url(url: str) -> str:
    """scheme://host:port only, dropping any userinfo so credentials embedded
    in a service URL never reach a skip message or CI log."""
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

    # Per-run collection so repeated/concurrent runs never collide; the
    # lifespan reads it from QDRANT_COLLECTION (a Settings field).
    collection = f"test12_api_{uuid.uuid4().hex[:12]}"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-used-by-upload")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "rag.sqlite3"))
    monkeypatch.setenv("QDRANT_URL", QDRANT_URL)
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


def test_upload_then_update_through_real_lifespan(app_env):
    with TestClient(create_app()) as client:
        first = client.post(
            "/documents",
            files={
                "file": (
                    "password-faq.txt",
                    b"To reset your password open the login page and click forgot password.",
                    "text/plain",
                )
            },
        )
        assert first.status_code == 201, first.text
        body = first.json()
        assert body["doc_id"] == "password-faq"
        assert body["version"] == 1
        assert body["chunk_count"] >= 1

        second = client.put(
            "/documents/password-faq",
            files={
                "file": (
                    "password-faq.txt",
                    b"Password resets now happen in the mobile app under Security settings.",
                    "text/plain",
                )
            },
        )
        assert second.status_code == 200, second.text
        assert second.json()["version"] == 2
