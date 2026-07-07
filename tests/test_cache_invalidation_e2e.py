"""The money test for issue #20: a document update evicts its cached answers.

End-to-end across the two pipelines, over the real backing services
(Redis, Qdrant, Ollama embeddings) with ONE fake substituted — a
deterministic LLM, so the answer text is fixed and version-distinguishing
without a live generation model. The reranker is a passthrough for the
same reason (keep the assertion about eviction, not about gemma4 scoring);
retrieval, embedding, vector search, ingestion, and the cache/invalidator
are all the real thing.

The full loop this issue closes:

    (a) ingest v1 of a document via the real IngestionService
    (b) POST /chat a first-turn question  -> MISS, answer cached
        (assert cache:{fp} + a doc_tag:{doc_id} member exist in Redis)
    (c) POST the SAME question again       -> HIT (retriever NOT re-run;
        cached: true; the answer is v1's, verbatim from the cache)
    (d) PUT an UPDATED v2 of the document  -> invalidate_document runs
        (assert cache:{fp} is GONE and the doc_tag set is gone)
    (e) POST the SAME question a third time -> MISS again (retriever
        re-runs; cached: false); the answer + sources reflect v2 (the
        chunk ids are version 2 and the text is v2's).

Skipped cleanly when any service is unreachable; run locally only (CI has
no Ollama). Secrets/credentials never reach a skip message (_safe_url).
"""

import os
import urllib.request
import uuid
from urllib.parse import urlsplit

import httpx2
import pytest
import redis
from fastapi import Request
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.api.chat import get_llm_client, get_reranker, get_retriever
from app.chat.fingerprint import fingerprint
from app.main import create_app

pytestmark = pytest.mark.integration

# Read before the hermetic_settings fixture scrubs env at test time.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
EMBED_MODEL = "nomic-embed-text"

# Two versions of one FAQ. The answer token ("portal"/"mobile app") differs
# so a stale cached answer is observably v1's, and v2's chunk ids prove the
# fresh retrieval hit the new version.
V1_TEXT = (
    "To reset your password, open the login page and click the forgot-password "
    "link. A reset email arrives within five minutes via the web portal."
)
V2_TEXT = (
    "Password resets moved to the mobile app: open Settings, choose Security, "
    "and tap the reset button in the mobile app. No email is sent anymore."
)
QUESTION = "how do I reset my password?"


def _safe_url(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{host}{port}" if host else "(redacted)"


def _ollama_embed_available() -> bool:
    try:
        response = httpx2.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        response.raise_for_status()
        names = [entry["name"] for entry in response.json()["models"]]
    except Exception:
        return False
    return any(name == EMBED_MODEL or name.startswith(f"{EMBED_MODEL}:") for name in names)


def _qdrant_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{QDRANT_URL}/readyz", timeout=2):
            return True
    except OSError:
        return False


def _redis_reachable() -> bool:
    client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2, socket_timeout=2)
    try:
        client.ping()
        return True
    except (redis.RedisError, OSError):
        return False
    finally:
        client.close()


class _EchoSourcesLLM:
    """Deterministic stand-in for the generation client.

    Returns a fixed answer built from the escaped <source> text in the
    system prompt, so the answer is (a) stable — identical inputs give
    identical output, which the cache relies on — and (b) version-aware:
    it names the resolution channel found in the retrieved chunks, so a
    stale (cached v1) answer reads differently from a fresh v2 one.
    """

    def complete(self, messages):
        sources_text = " ".join(m.content for m in messages if m.role == "system")
        if "mobile app" in sources_text:
            channel = "the mobile app"
        elif "portal" in sources_text:
            channel = "the web portal"
        else:
            channel = "our help center"
        return f"Reset your password via {channel}."

    def close(self):
        pass


class _PassthroughReranker:
    """Keeps the fused order and count; no live gemma4 scoring."""

    def rerank(self, query, candidates, top_n=5):
        return list(candidates)[:top_n]


class _SpyRetriever:
    """Wraps the app's real RetrieverPool and counts retrieve() calls, so
    a cache HIT can be proven by the count NOT advancing."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def retrieve(self, query, *, k_each=20, top_n=12):
        self.calls += 1
        return self._inner.retrieve(query, k_each=k_each, top_n=top_n)


def _versions_in(sources) -> set[int]:
    # chunk ids are "{doc_id}:{version}:{seq}"; this doc id has no colon.
    return {int(s["chunk_id"].split(":")[1]) for s in sources}


@pytest.fixture
def app_client(monkeypatch, tmp_path):
    """A live app (real Redis/Qdrant/Ollama-embed) with a deterministic LLM
    and passthrough reranker, on a throwaway Qdrant collection + isolated
    SQLite db, cleaned up afterward."""
    if not _ollama_embed_available():
        pytest.skip(f"Ollama/{EMBED_MODEL} not available at {_safe_url(OLLAMA_BASE_URL)}")
    if not _qdrant_reachable():
        pytest.skip(f"Qdrant not reachable at {_safe_url(QDRANT_URL)}")
    if not _redis_reachable():
        pytest.skip(f"Redis not reachable at {_safe_url(REDIS_URL)}")

    collection = f"test20_e2e_{uuid.uuid4().hex[:12]}"
    db_path = str(tmp_path / "rag.sqlite3")
    # Configure the app via env; hermetic_settings (autouse) already cleared
    # the get_settings cache, so create_app() reads these.
    monkeypatch.setenv("REDIS_URL", REDIS_URL)
    monkeypatch.setenv("QDRANT_URL", QDRANT_URL)
    monkeypatch.setenv("QDRANT_COLLECTION", collection)
    monkeypatch.setenv("DB_PATH", db_path)
    monkeypatch.setenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", EMBED_MODEL)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    app = create_app()
    spy_holder: dict[str, _SpyRetriever] = {}

    def _spy_retriever(request: Request):
        # Wrap the real pool the lifespan built, once, and reuse it.
        if "spy" not in spy_holder:
            spy_holder["spy"] = _SpyRetriever(request.app.state.retriever)
        return spy_holder["spy"]

    app.dependency_overrides[get_llm_client] = lambda: _EchoSourcesLLM()
    app.dependency_overrides[get_reranker] = lambda: _PassthroughReranker()
    app.dependency_overrides[get_retriever] = _spy_retriever

    raw_redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        with TestClient(app) as client:
            yield client, spy_holder, raw_redis, collection
    finally:
        app.dependency_overrides.clear()
        # Drop the throwaway Qdrant collection.
        qdrant = QdrantClient(url=QDRANT_URL)
        try:
            qdrant.delete_collection(collection)
        finally:
            qdrant.close()
        raw_redis.close()


def _ingest(client, doc_id, text, *, put=False):
    files = {"file": (f"{doc_id}.txt", text, "text/plain")}
    if put:
        response = client.put(f"/documents/{doc_id}", files=files)
    else:
        response = client.post("/documents", files=files, data={"doc_id": doc_id})
    assert response.status_code in (200, 201), response.text
    return response.json()


def _chat(client, session_id):
    return client.post("/chat", json={"session_id": session_id, "message": QUESTION})


def test_doc_update_evicts_cached_answer(app_client):
    client, spy_holder, raw_redis, _collection = app_client
    run = uuid.uuid4().hex[:8]
    doc_id = f"faq-pw-{run}"
    # Unique session ids per run: a reused id would carry 24h-TTL history from
    # a prior run in real Redis, making the question a follow-up (fingerprint
    # None) that bypasses the cache entirely — the cache must see a first turn.
    sessions = [f"sess-{run}-{n}" for n in range(3)]

    fp = fingerprint(QUESTION, [])
    assert fp is not None
    cache_key = f"cache:{fp}"
    tag_key = f"doc_tag:{doc_id}"
    # Clean slate for this fingerprint/doc (a prior run may have left keys).
    raw_redis.delete(cache_key, tag_key)

    # (a) ingest v1
    v1 = _ingest(client, doc_id, V1_TEXT)
    assert v1["version"] == 1

    # (b) first chat -> MISS, answer cached
    first = _chat(client, sessions[0])
    assert first.status_code == 200, first.text
    body1 = first.json()
    assert body1["cached"] is False
    assert "portal" in body1["answer"].lower()
    assert _versions_in(body1["sources"]) == {1}
    spy = spy_holder["spy"]
    assert spy.calls == 1, "retriever should have run once on the miss"
    # The cache entry and its doc-tag member now exist in Redis.
    assert raw_redis.exists(cache_key) == 1
    assert raw_redis.sismember(tag_key, cache_key)

    # (c) same question again -> HIT, pipeline NOT re-run
    second = _chat(client, sessions[1])  # different session => still first-turn
    assert second.status_code == 200, second.text
    body2 = second.json()
    assert body2["cached"] is True
    assert body2["answer"] == body1["answer"]  # verbatim v1 answer from cache
    assert spy.calls == 1, "a cache HIT must NOT re-run retrieval"

    # (d) update to v2 -> ingestion triggers invalidate_document
    v2 = _ingest(client, doc_id, V2_TEXT, put=True)
    assert v2["version"] == 2
    # The cache key AND its doc-tag set are gone (evicted + pruned).
    assert raw_redis.exists(cache_key) == 0, "updated doc must evict its cached answer"
    assert raw_redis.exists(tag_key) == 0, "tag set must be pruned on invalidation"

    # (e) same question a third time -> MISS again, answer reflects v2
    third = _chat(client, sessions[2])
    assert third.status_code == 200, third.text
    body3 = third.json()
    assert body3["cached"] is False, "after eviction the same question must MISS"
    assert spy.calls == 2, "the post-eviction miss must re-run retrieval"
    assert "mobile app" in body3["answer"].lower()
    # Sources now cite the NEW version's chunks only.
    assert _versions_in(body3["sources"]) == {2}
    # And the fresh answer is cached again under the same fingerprint.
    assert raw_redis.exists(cache_key) == 1
