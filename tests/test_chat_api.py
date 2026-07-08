"""Unit specs for the grounded chat API (issue #18).

These run everywhere with no live services: the route's collaborators —
session store, retriever, reranker, and LLM client — are replaced with
fakes via FastAPI dependency overrides, so the tests exercise the HTTP
boundary and the retrieve -> rerank -> assemble -> generate -> cite flow
without touching Ollama, Qdrant, or Redis.

The grounded-refusal path (no retrieval => no LLM call), the 503 failure
policy, and the "generation runs off the event loop" property are the
behaviour-critical specs and are asserted directly here.
"""

import threading

import pytest
from fastapi.testclient import TestClient

from app.api.chat import (
    ESCALATION_ANSWER,
    get_llm_client,
    get_reranker,
    get_response_cache,
    get_retriever,
    get_session_store,
)
from app.chat.llm import LLMError
from app.main import create_app
from app.models import Chunk, RetrievedChunk, Turn


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
    """Returns preset chunks; records the query and its thread."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.queries = []
        self.thread_names = []

    def retrieve(self, query, *, k_each=20, top_n=12):
        self.queries.append(query)
        self.thread_names.append(threading.current_thread().name)
        return list(self._chunks)


class FakeReranker:
    """Returns the top_n candidates unchanged (records the call).

    ``top_score`` is the best relevance score reported by rerank_scored:
    it defaults high so the relevance gate (#50) passes and the pipeline
    answers; a test sets it low to exercise a weak-retrieval refusal, or
    None to exercise the full fail-open path (reranker scored nothing).
    """

    def __init__(self, *, top_score=10.0):
        self.calls = []
        self._top_score = top_score

    def rerank(self, query, candidates, top_n=5):
        self.calls.append((query, list(candidates)))
        return list(candidates)[:top_n]

    def rerank_scored(self, query, candidates, top_n=5):
        self.calls.append((query, list(candidates)))
        return list(candidates)[:top_n], self._top_score


class FakeLLMClient:
    def __init__(self, *, reply="grounded answer", raises=None):
        self.reply = reply
        self.raises = raises
        self.calls = []
        self.thread_names = []

    def complete(self, messages):
        self.calls.append(list(messages))
        self.thread_names.append(threading.current_thread().name)
        if self.raises is not None:
            raise self.raises
        return self.reply


class MissingCache:
    """Always-miss response cache: these specs cover the live pipeline, so
    the cache never intercepts (get returns None; set is a no-op). The
    cache's own behaviour is covered in test_chat_cache.py."""

    def get(self, fp):
        return None

    def set(self, fp, response, source_doc_ids, ttl):
        pass


def _build_app(*, chunks=None, llm=None, sessions=None, reranker=None):
    app = create_app()
    # Pre-stash an ingestion_service so the lifespan skips building live
    # resources (no Ollama/Qdrant/Redis in the unit suite).
    app.state.ingestion_service = object()
    app.state.max_upload_bytes = 5_000_000

    sessions = sessions or FakeSessionStore()
    retriever = FakeRetriever(chunks if chunks is not None else [])
    reranker = reranker or FakeReranker()
    llm = llm or FakeLLMClient()

    app.dependency_overrides[get_session_store] = lambda: sessions
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_reranker] = lambda: reranker
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[get_response_cache] = lambda: MissingCache()
    app.state._fakes = {
        "sessions": sessions,
        "retriever": retriever,
        "reranker": reranker,
        "llm": llm,
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


def test_returns_answer_and_sources_from_reranked_chunks(two_chunks):
    app = _build_app(chunks=two_chunks, llm=FakeLLMClient(reply="Use the portal to reset."))
    with TestClient(app) as client:
        response = _post(client)
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Use the portal to reset."
    assert body["sources"] == [
        {"chunk_id": "doc-1:1:0", "doc_id": "doc-1", "title": "Password Reset"},
        {"chunk_id": "doc-2:1:0", "doc_id": "doc-2", "title": "Billing"},
    ]


def test_no_retrieval_results_returns_escalation_without_llm_call():
    llm = FakeLLMClient()
    app = _build_app(chunks=[], llm=llm)
    with TestClient(app) as client:
        response = _post(client)
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == ESCALATION_ANSWER
    assert body["sources"] == []
    # A support bot must refuse rather than hallucinate: LLM never invoked.
    assert llm.calls == []


def test_low_relevance_returns_escalation_without_llm_call(two_chunks):
    # Retrieval found chunks, but the reranker judged even the best one below
    # the floor (weak/irrelevant). Refuse rather than answer from it (#50).
    llm = FakeLLMClient()
    app = _build_app(chunks=two_chunks, llm=llm, reranker=FakeReranker(top_score=1.0))
    app.state.min_rerank_score = 5.0  # floor above the weak score
    with TestClient(app) as client:
        response = _post(client)
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == ESCALATION_ANSWER
    assert body["sources"] == []
    assert llm.calls == []


def test_relevance_at_floor_answers(two_chunks):
    # The gate refuses on score < floor, so a score equal to the floor still
    # answers — the boundary is inclusive of the floor.
    llm = FakeLLMClient(reply="Use the portal to reset.")
    app = _build_app(chunks=two_chunks, llm=llm, reranker=FakeReranker(top_score=5.0))
    app.state.min_rerank_score = 5.0
    with TestClient(app) as client:
        response = _post(client)
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["answer"] == "Use the portal to reset."
    assert len(llm.calls) == 1


def test_reranker_full_fail_open_still_answers(two_chunks):
    # The reranker scored NOTHING (top_score None, e.g. Ollama down). With no
    # relevance signal the gate must NOT refuse — a reranker outage can't take
    # chat offline; it degrades to answering from the fused order.
    llm = FakeLLMClient(reply="answer from fused order")
    app = _build_app(chunks=two_chunks, llm=llm, reranker=FakeReranker(top_score=None))
    app.state.min_rerank_score = 5.0
    with TestClient(app) as client:
        response = _post(client)
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["answer"] == "answer from fused order"
    assert len(llm.calls) == 1


def test_llm_failure_returns_503_and_appends_no_turns(two_chunks):
    llm = FakeLLMClient(raises=LLMError("generation failed"))
    sessions = FakeSessionStore()
    app = _build_app(chunks=two_chunks, llm=llm, sessions=sessions)
    with TestClient(app) as client:
        response = _post(client, session_id="session-xyz1")
    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"error": "generation_unavailable"}
    # Failed generation leaves the session exactly as it was: no turns.
    assert sessions.turns.get("session-xyz1", []) == []


def test_turns_appended_in_order_after_success(two_chunks):
    sessions = FakeSessionStore()
    app = _build_app(
        chunks=two_chunks, llm=FakeLLMClient(reply="Here is the answer."), sessions=sessions
    )
    with TestClient(app) as client:
        response = _post(client, session_id="session-conv1", message="my question")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert sessions.turns["session-conv1"] == [
        Turn(role="user", content="my question"),
        Turn(role="assistant", content="Here is the answer."),
    ]


def test_history_is_passed_to_generation(two_chunks):
    sessions = FakeSessionStore()
    sessions.turns["session-hist1"] = [
        Turn(role="user", content="earlier question"),
        Turn(role="assistant", content="earlier answer"),
    ]
    llm = FakeLLMClient(reply="ok")
    app = _build_app(chunks=two_chunks, llm=llm, sessions=sessions)
    with TestClient(app) as client:
        _post(client, session_id="session-hist1", message="follow up")
    app.dependency_overrides.clear()

    # The prior turns land in the assembled prompt sent to the LLM.
    sent = llm.calls[0]
    contents = [m.content for m in sent]
    assert "earlier question" in contents
    assert "earlier answer" in contents


@pytest.mark.parametrize("bad_message", ["", " " * 5, "x" * 2001])
def test_message_length_validation_returns_422(bad_message):
    app = _build_app(chunks=[])
    with TestClient(app) as client:
        response = client.post("/chat", json={"session_id": "session-abcd", "message": bad_message})
    app.dependency_overrides.clear()

    assert response.status_code == 422


def test_invalid_session_id_returns_422():
    app = _build_app(chunks=[])
    with TestClient(app) as client:
        response = client.post("/chat", json={"session_id": "short", "message": "hi there"})
    app.dependency_overrides.clear()

    assert response.status_code == 422


def test_generation_runs_off_the_event_loop_thread(two_chunks):
    llm = FakeLLMClient(reply="ok")
    app = _build_app(chunks=two_chunks, llm=llm)
    with TestClient(app) as client:
        _post(client)
    app.dependency_overrides.clear()

    retriever = app.state._fakes["retriever"]
    assert llm.thread_names and retriever.thread_names
    for name in llm.thread_names + retriever.thread_names:
        assert name != "MainThread"


def test_empty_generation_returns_503_and_appends_no_turns(two_chunks):
    # A model that returns a blank answer (the thinking-model empty-output
    # failure mode) must not surface a 200 with an empty answer.
    sessions = FakeSessionStore()
    app = _build_app(chunks=two_chunks, llm=FakeLLMClient(reply="   "), sessions=sessions)
    with TestClient(app) as client:
        response = _post(client)
    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"error": "generation_unavailable"}
    assert sessions.turns == {}  # untouched, so a retry is clean


def test_redis_calls_run_off_the_event_loop_thread(two_chunks):
    # get_history + append_turn are blocking sync redis-py calls; they must
    # be offloaded like retrieve/rerank/generate, or a slow Redis freezes the
    # event loop. Assert they run off MainThread (deterministic — a timing
    # test can't reliably detect event-loop blocking through TestClient).
    class SpySessions(FakeSessionStore):
        def __init__(self):
            super().__init__()
            self.thread_names = []

        def get_history(self, session_id, limit=10):
            self.thread_names.append(threading.current_thread().name)
            return super().get_history(session_id, limit=limit)

        def append_turn(self, session_id, turn):
            self.thread_names.append(threading.current_thread().name)
            super().append_turn(session_id, turn)

    sessions = SpySessions()
    app = _build_app(chunks=two_chunks, llm=FakeLLMClient(reply="ok"), sessions=sessions)
    with TestClient(app) as client:
        assert _post(client).status_code == 200
    app.dependency_overrides.clear()

    # 1 get_history + 2 append_turn = 3 calls, all offloaded to the anyio
    # worker pool (event-loop work runs on an "asyncio-portal-*" thread; only
    # anyio.to_thread work lands on an "AnyIO worker thread").
    assert len(sessions.thread_names) == 3
    assert all("worker" in name.lower() for name in sessions.thread_names)


def test_slow_generation_does_not_block_concurrent_health(two_chunks):
    release = threading.Event()

    class SlowLLM(FakeLLMClient):
        def complete(self, messages):
            release.wait(timeout=5)
            return "done"

    app = _build_app(chunks=two_chunks, llm=SlowLLM())
    with TestClient(app) as client:
        result = {}

        def do_chat():
            result["response"] = _post(client)

        chatter = threading.Thread(target=do_chat)
        chatter.start()
        try:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}
        finally:
            release.set()
            chatter.join(timeout=5)
    app.dependency_overrides.clear()

    assert result["response"].status_code == 200
