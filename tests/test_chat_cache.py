"""Read-through response cache wired into POST /chat (issue #19).

These run offline: the route's collaborators (session store, retriever,
reranker, LLM, and now the response cache) are FastAPI dependency
overrides, so the tests exercise the cache decision at the HTTP boundary
without Redis/Ollama/Qdrant.

Behaviour under test:

- A second IDENTICAL first-turn message is a cache HIT: retrieval,
  rerank, and the LLM are NOT called, and the response carries
  ``cached: true``. On a hit both turns are still appended so the
  conversation continues (documented choice), but the expensive pipeline
  is skipped.
- A follow-up (non-empty history) has no fingerprint, so it BYPASSES the
  cache entirely and runs the full pipeline.
- On a MISS the answer is written back with exactly the source doc-ids of
  the cited chunks.
- Cache failure is fail-open: a raising cache still yields a live answer.
"""

import threading

import pytest
from fastapi.testclient import TestClient

from app.api.chat import (
    get_llm_client,
    get_reranker,
    get_response_cache,
    get_retriever,
    get_session_store,
)
from app.main import create_app
from app.models import Chunk, RetrievedChunk, Turn
from app.stores.cache import CachedResponse


def _retrieved(chunk_id, text, *, doc_id="doc-1", title="Title", score=1.0):
    chunk = Chunk(id=chunk_id, doc_id=doc_id, version=1, seq=0, text=text, title=title)
    return RetrievedChunk(chunk=chunk, score=score, sources=frozenset({"bm25"}))


class FakeSessionStore:
    def __init__(self):
        self.turns: dict[str, list[Turn]] = {}

    def get_history(self, session_id, limit=10):
        return list(self.turns.get(session_id, []))

    def append_turn(self, session_id, turn):
        self.turns.setdefault(session_id, []).append(turn)


class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = 0

    def retrieve(self, query, *, k_each=20, top_n=12):
        self.calls += 1
        return list(self._chunks)


class FakeReranker:
    def __init__(self):
        self.calls = 0

    def rerank(self, query, candidates, top_n=5):
        self.calls += 1
        return list(candidates)[:top_n]

    def rerank_scored(self, query, candidates, top_n=5):
        self.calls += 1
        # High score => relevance gate passes; these specs cover caching.
        return list(candidates)[:top_n], 10.0


class FakeLLMClient:
    def __init__(self, *, reply="grounded answer"):
        self.reply = reply
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        return self.reply


class FakeCache:
    """In-memory stand-in for ResponseCache; records call counts and tags."""

    def __init__(self):
        self.store: dict[str, CachedResponse] = {}
        self.get_calls = 0
        self.set_calls = 0
        self.set_tags: list[tuple[str, list[str]]] = []

    def get(self, fp):
        self.get_calls += 1
        return self.store.get(fp)

    def set(self, fp, response, source_doc_ids, ttl):
        self.set_calls += 1
        self.set_tags.append((fp, list(source_doc_ids)))
        self.store[fp] = response


class RaisingCache:
    """Fail-open probe: every method raises, chat must still answer."""

    def get(self, fp):
        raise RuntimeError("cache get boom")

    def set(self, fp, response, source_doc_ids, ttl):
        raise RuntimeError("cache set boom")


def _build_app(*, chunks, llm=None, sessions=None, reranker=None, retriever=None, cache=None):
    app = create_app()
    app.state.ingestion_service = object()
    app.state.max_upload_bytes = 5_000_000

    sessions = sessions or FakeSessionStore()
    retriever = retriever or FakeRetriever(chunks)
    reranker = reranker or FakeReranker()
    llm = llm or FakeLLMClient()
    cache = cache or FakeCache()

    app.dependency_overrides[get_session_store] = lambda: sessions
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_reranker] = lambda: reranker
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[get_response_cache] = lambda: cache
    app.state._fakes = {
        "sessions": sessions,
        "retriever": retriever,
        "reranker": reranker,
        "llm": llm,
        "cache": cache,
    }
    return app


@pytest.fixture
def two_chunks():
    return [
        _retrieved("doc-1:1:0", "Reset via the portal.", doc_id="doc-1", title="Password Reset"),
        _retrieved("doc-2:1:0", "Contact billing.", doc_id="doc-2", title="Billing"),
    ]


def _post(client, session_id="session-abcd", message="how do I reset my password?"):
    return client.post("/chat", json={"session_id": session_id, "message": message})


def test_second_identical_first_turn_is_cache_hit_skips_pipeline(two_chunks):
    llm = FakeLLMClient(reply="Use the portal to reset.")
    retriever = FakeRetriever(two_chunks)
    reranker = FakeReranker()
    cache = FakeCache()
    app = _build_app(
        chunks=two_chunks, llm=llm, retriever=retriever, reranker=reranker, cache=cache
    )
    with TestClient(app) as client:
        first = _post(client, session_id="session-one1")
        # A DIFFERENT session so history stays empty (first-turn) but the
        # normalized question — hence fingerprint — is identical.
        second = _post(client, session_id="session-two2")
    app.dependency_overrides.clear()

    assert first.status_code == 200
    assert first.json()["cached"] is False
    # Second call is a hit: same answer, cached flag set.
    assert second.status_code == 200
    body = second.json()
    assert body["answer"] == "Use the portal to reset."
    assert body["cached"] is True
    assert body["sources"] == [
        {"chunk_id": "doc-1:1:0", "doc_id": "doc-1", "title": "Password Reset"},
        {"chunk_id": "doc-2:1:0", "doc_id": "doc-2", "title": "Billing"},
    ]
    # The expensive pipeline ran ONCE (the miss), not on the hit.
    assert retriever.calls == 1
    assert reranker.calls == 1
    assert llm.calls == 1


def test_cache_hit_still_appends_turns(two_chunks):
    sessions = FakeSessionStore()
    cache = FakeCache()
    app = _build_app(chunks=two_chunks, cache=cache, sessions=sessions)
    with TestClient(app) as client:
        _post(client, session_id="session-miss1", message="reset my password")
        _post(client, session_id="session-hit01", message="reset my password")
    app.dependency_overrides.clear()

    # On the hit the conversation still advances: user + assistant appended.
    assert sessions.turns["session-hit01"] == [
        Turn(role="user", content="reset my password"),
        Turn(role="assistant", content="grounded answer"),
    ]


def test_followup_turn_bypasses_cache(two_chunks):
    # Non-empty history => fingerprint None => cache never consulted, full
    # pipeline runs.
    sessions = FakeSessionStore()
    sessions.turns["session-conv1"] = [
        Turn(role="user", content="earlier"),
        Turn(role="assistant", content="reply"),
    ]
    cache = FakeCache()
    retriever = FakeRetriever(two_chunks)
    llm = FakeLLMClient(reply="follow-up answer")
    app = _build_app(
        chunks=two_chunks, cache=cache, sessions=sessions, retriever=retriever, llm=llm
    )
    with TestClient(app) as client:
        response = _post(client, session_id="session-conv1", message="follow up question")
    app.dependency_overrides.clear()

    body = response.json()
    assert body["cached"] is False
    # Cache was never touched (no fingerprint for a follow-up turn).
    assert cache.get_calls == 0
    assert cache.set_calls == 0
    # Full pipeline ran.
    assert retriever.calls == 1
    assert llm.calls == 1


def test_miss_writes_back_with_source_doc_id_tags(two_chunks):
    cache = FakeCache()
    app = _build_app(chunks=two_chunks, cache=cache)
    with TestClient(app) as client:
        response = _post(client, session_id="session-set01", message="reset my password")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert cache.set_calls == 1
    fp, doc_ids = cache.set_tags[0]
    # Exactly the (unique) source docs of the cited chunks — no duplicates.
    assert sorted(doc_ids) == ["doc-1", "doc-2"]


def test_no_context_escalation_is_not_cached():
    # A grounded refusal (no retrieval) must not populate the cache: it is
    # not a real answer, and caching it would suppress future retrieval.
    cache = FakeCache()
    app = _build_app(chunks=[], cache=cache)
    with TestClient(app) as client:
        response = _post(client, session_id="session-esc01", message="unknown topic")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["cached"] is False
    assert cache.set_calls == 0


def test_cache_get_failure_is_fail_open(two_chunks):
    # A raising cache on GET must not break chat: fall through to the pipeline.
    llm = FakeLLMClient(reply="live answer")
    app = _build_app(chunks=two_chunks, llm=llm, cache=RaisingCache())
    with TestClient(app) as client:
        response = _post(client, session_id="session-fo001", message="reset my password")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "live answer"
    assert body["cached"] is False


def test_cache_calls_run_off_the_event_loop_thread(two_chunks):
    class SpyCache(FakeCache):
        def __init__(self):
            super().__init__()
            self.thread_names = []

        def get(self, fp):
            self.thread_names.append(threading.current_thread().name)
            return super().get(fp)

        def set(self, fp, response, source_doc_ids, ttl):
            self.thread_names.append(threading.current_thread().name)
            super().set(fp, response, source_doc_ids, ttl)

    cache = SpyCache()
    app = _build_app(chunks=two_chunks, cache=cache)
    with TestClient(app) as client:
        assert _post(client, message="reset my password").status_code == 200
    app.dependency_overrides.clear()

    # One get (miss) + one set, both offloaded to the anyio worker pool —
    # redis-py is blocking and must never run on the event loop.
    assert cache.thread_names
    assert all("worker" in name.lower() for name in cache.thread_names)
