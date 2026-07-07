"""End-to-end ingestion pipeline test against live Ollama and Qdrant.

Skipped when either backend is unreachable or the embed model is not
pulled, so this runs locally only (CI has a Qdrant service but no
Ollama — the required-Qdrant enforcement lives in test_vector_store).
Wires the pipeline exactly as the app will: one ChunkStore connection
plus a second sqlite3 connection to the same WAL database for Bm25Index,
and a per-run Qdrant collection sized from the live embedder.
"""

import os
import sqlite3
import urllib.request
import uuid

import httpx2
import pytest
from app.ingestion.pipeline import IngestionService, NoopCacheInvalidator
from qdrant_client import QdrantClient

from app.retrieval.embedder import Embedder, OllamaEmbeddingsClient
from app.stores.bm25_index import Bm25Index
from app.stores.chunk_store import ChunkStore
from app.stores.vector_store import VectorStore

# Read at collection time, before the hermetic_settings fixture scrubs env.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
MODEL = "nomic-embed-text"

pytestmark = pytest.mark.integration

V1_TEXT = (
    "To reset your password, open the login page and click the forgot-password "
    "link. A reset email arrives within five minutes."
)
V2_TEXT = (
    "Password resets moved to the mobile app: open Settings, choose Security, "
    "and tap the reset button. No email is sent anymore."
)


def _ollama_model_available() -> bool:
    """Probe for a live Ollama with the test model pulled; any failure means skip."""
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
def rig(tmp_path, request):
    # Probed here, not at import, so unit-only runs never touch the network.
    if not _ollama_model_available():
        pytest.skip(f"Ollama not reachable at {OLLAMA_BASE_URL} or {MODEL} not pulled")
    if not _qdrant_reachable():
        pytest.skip(f"Qdrant not reachable at {QDRANT_URL}")

    db_path = str(tmp_path / "rag.sqlite3")
    chunk_store = ChunkStore(db_path)
    # second connection to the same WAL database, as the app will wire it
    conn = sqlite3.connect(db_path)
    bm25 = Bm25Index(conn)
    client = OllamaEmbeddingsClient(base_url=OLLAMA_BASE_URL, model=MODEL)
    embedder = Embedder(client=client, model=MODEL)
    collection = f"test11_e2e_{uuid.uuid4().hex[:12]}"
    vector_store = VectorStore(QDRANT_URL, vector_size=embedder.dim(), collection=collection)

    def teardown() -> None:
        vector_store.close()
        embedder.close()
        conn.close()
        chunk_store.close()
        qdrant = QdrantClient(url=QDRANT_URL)
        try:
            qdrant.delete_collection(collection)
        finally:
            qdrant.close()

    # registered before ensure_collection, so everything is released even
    # when collection bootstrap itself fails
    request.addfinalizer(teardown)
    vector_store.ensure_collection()
    service = IngestionService(chunk_store, bm25, embedder, vector_store, NoopCacheInvalidator())
    yield {
        "service": service,
        "chunk_store": chunk_store,
        "bm25": bm25,
        "embedder": embedder,
        "vector_store": vector_store,
    }


def _version_of(chunk_id: str) -> int:
    # ids are "{doc_id}:{version}:{seq}" and this test's doc id has no colon
    return int(chunk_id.split(":")[1])


def test_update_supersedes_old_version_in_both_search_indexes(rig):
    first = rig["service"].ingest("faq-password", "Password FAQ", V1_TEXT)
    second = rig["service"].ingest("faq-password", "Password FAQ", V2_TEXT)

    assert (first.version, second.version) == (1, 2)

    bm25_hits = rig["bm25"].search("reset password email", k=10)
    assert bm25_hits, "BM25 returned nothing for the new version"
    assert all(_version_of(chunk_id) == 2 for chunk_id, _ in bm25_hits)

    query_vector = rig["embedder"].embed_query("how do I reset my password?")
    vector_hits = rig["vector_store"].search(query_vector, k=10)
    assert vector_hits, "Qdrant returned nothing for the new version"
    assert all(_version_of(chunk_id) == 2 for chunk_id, _ in vector_hits)

    # hits hydrate back to the new version's text via the chunk store...
    hydrated = rig["chunk_store"].get_chunks_by_ids([chunk_id for chunk_id, _ in bm25_hits])
    assert hydrated and all("mobile app" in chunk.text for chunk in hydrated)
    # ...and the superseded version's rows were reclaimed
    assert rig["chunk_store"].get_chunks("faq-password", 1) == []
