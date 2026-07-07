"""HTTP-boundary specs for observability (issue #21).

Request-id middleware, the one-summary-line-per-chat-request log, the
content-free-logging PII guarantee, and the /ready readiness probe —
driven through the app with fake collaborators so no live service is
touched. The /ready all-up integration path lives in
test_observability_integration.py.
"""

import json
import logging

import pytest
import structlog
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
from app.observability import configure_logging
from app.stores.cache import CachedResponse

# --- fakes (mirror the chat contracts, no I/O) ---


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

    def retrieve(self, query, *, k_each=20, top_n=12):
        return list(self._chunks)


class FakeReranker:
    def rerank(self, query, candidates, top_n=5):
        return list(candidates)[:top_n]


class FakeLLMClient:
    def __init__(self, *, reply="grounded answer"):
        self.reply = reply

    def complete(self, messages):
        return self.reply


class FakeCache:
    def __init__(self):
        self.store: dict[str, CachedResponse] = {}

    def get(self, fp):
        return self.store.get(fp)

    def set(self, fp, response, source_doc_ids, ttl):
        self.store[fp] = response


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
    return app


@pytest.fixture
def two_chunks():
    return [
        _retrieved("doc-1:1:0", "Reset via the portal.", doc_id="doc-1", title="Password Reset"),
        _retrieved("doc-2:1:0", "Contact billing.", doc_id="doc-2", title="Billing"),
    ]


@pytest.fixture
def captured_logs():
    """Capture every structlog event as a parsed dict for the duration of a test.

    Uses structlog's own capture processor so we see exactly the event
    dicts the JSON renderer would serialize — the source of truth for both
    the summary-line assertions and the PII scan.
    """
    # Ensure the app's base config is already installed and the idempotent
    # guard is tripped, so create_app() inside the test does NOT reconfigure
    # structlog and clobber the capture processor we splice in below.
    configure_logging()
    cap = structlog.testing.LogCapture()
    old = structlog.get_config()["processors"]
    # Replace the terminal renderer (JSONRenderer) with the capture sink,
    # keeping the contextvars merge so request_id still rides along.
    structlog.configure(processors=[*old[:-1], cap])
    try:
        yield cap.entries
    finally:
        structlog.configure(processors=old)


def _post(client, session_id="session-abcd", message="how do I reset my password?"):
    return client.post("/chat", json={"session_id": session_id, "message": message})


# --- request-id middleware ---


def test_request_id_generated_and_echoed(two_chunks):
    app = _build_app(chunks=two_chunks)
    with TestClient(app) as client:
        response = _post(client)
    rid = response.headers.get("X-Request-ID")
    assert rid is not None
    assert len(rid) == 32  # generated uuid4 hex


def test_request_id_propagated_from_header(two_chunks, captured_logs):
    app = _build_app(chunks=two_chunks)
    provided = "client-supplied-42"
    with TestClient(app) as client:
        response = _post(client)  # warm any first-request cost off the record
        assert response.status_code == 200
        captured_logs.clear()
        response = client.post(
            "/chat",
            json={"session_id": "session-xyz1", "message": "another question"},
            headers={"X-Request-ID": provided},
        )
    assert response.headers.get("X-Request-ID") == provided
    # every log line emitted during that request carries the provided id
    assert captured_logs, "expected at least one structured log line"
    assert all(entry.get("request_id") == provided for entry in captured_logs)


# --- one summary line per chat request ---


def test_chat_request_emits_one_summary_line_with_stage_timings(two_chunks, captured_logs):
    app = _build_app(chunks=two_chunks)
    with TestClient(app) as client:
        response = _post(client, session_id="session-sum1", message="reset my password")
    assert response.status_code == 200

    summaries = [e for e in captured_logs if e.get("event") == "request_summary"]
    assert len(summaries) == 1, f"expected exactly one summary line, got {len(summaries)}"
    s = summaries[0]
    assert s["route"] == "/chat"
    assert s["status"] == 200
    assert isinstance(s["cache_hit"], bool)
    assert s["cache_hit"] is False
    # a live (miss) request ran retrieval, rerank, and the LLM
    for key in ("retrieval_ms", "rerank_ms", "llm_ms", "total_ms"):
        assert key in s
        assert isinstance(s[key], (int, float))
    assert "request_id" in s


def test_cache_hit_summary_skips_stage_timings(two_chunks, captured_logs):
    app = _build_app(chunks=two_chunks)
    with TestClient(app) as client:
        # miss populates the cache
        _post(client, session_id="session-m1x1", message="reset my password")
        captured_logs.clear()
        # identical first-turn question in a fresh session => cache HIT
        response = _post(client, session_id="session-h1x1", message="reset my password")
    assert response.status_code == 200
    assert response.json()["cached"] is True

    summaries = [e for e in captured_logs if e.get("event") == "request_summary"]
    assert len(summaries) == 1
    s = summaries[0]
    assert s["cache_hit"] is True
    # a cache hit skips retrieval/rerank/llm => reported as 0 or omitted
    assert s.get("retrieval_ms", 0) == 0
    assert s.get("rerank_ms", 0) == 0
    assert s.get("llm_ms", 0) == 0


# --- content is NEVER logged ---


def test_no_message_or_document_content_in_logs(two_chunks, captured_logs, caplog):
    """Drive a chat and an upload with sentinel PII strings and confirm none
    of that content appears in ANY captured log line — across BOTH the
    structlog summary events and the handlers' stdlib logging (chat.py /
    documents.py log via logging.getLogger, which the structlog capture alone
    would not see)."""
    caplog.set_level(logging.DEBUG)
    secret_question = "SENTINEL_QUESTION_ssn_123_45_6789"
    secret_answer = "SENTINEL_ANSWER_card_4111_1111_1111_1111"
    secret_doc = b"SENTINEL_DOCUMENT_email_alice@example.com lives at 10 Downing St"

    from app.api.documents import get_ingestion_service
    from app.ingestion.pipeline import IngestResult

    class FakeIngestion:
        def ingest(self, doc_id, title, text):
            return IngestResult(doc_id=doc_id, version=1, chunk_count=1)

    llm = FakeLLMClient(reply=secret_answer)
    retriever = FakeRetriever(
        [_retrieved("doc-1:1:0", "SENTINEL_CHUNK_TEXT_secret", doc_id="doc-1", title="T")]
    )
    app = _build_app(chunks=two_chunks, llm=llm, retriever=retriever)
    app.dependency_overrides[get_ingestion_service] = lambda: FakeIngestion()

    with TestClient(app) as client:
        chat_resp = client.post(
            "/chat", json={"session_id": "session-pii1", "message": secret_question}
        )
        assert chat_resp.status_code == 200
        up_resp = client.post(
            "/documents",
            files={"file": ("faq.txt", secret_doc, "text/plain")},
        )
        assert up_resp.status_code == 201

    # Scan both logging systems: the structlog events AND the stdlib records
    # the handlers emit (caplog), so a future content-logging call in either
    # is caught.
    blob = json.dumps(captured_logs, default=str) + "\n" + caplog.text
    for sentinel in [
        secret_question,
        secret_answer,
        "SENTINEL_CHUNK_TEXT_secret",
        "SENTINEL_DOCUMENT",
        "alice@example.com",
        "4111",
    ]:
        assert sentinel not in blob, f"content leaked into logs: {sentinel!r}"


# --- /ready readiness probe ---


def test_ready_503_names_failing_dependency_without_leaking_url():
    """A down dependency => 503 naming it, and never leaking a URL/credential.

    Uses fake store handles on app.state so no real service is needed.
    """
    app = create_app()
    app.state.ingestion_service = object()
    app.state.max_upload_bytes = 5_000_000

    class OkConn:
        def execute(self, *a, **k):
            return None

    class OkQdrant:
        def get_collections(self):
            return None

    class DownRedis:
        def ping(self):
            raise RuntimeError("Error connecting to redis://user:pw@10.0.0.1:6379")

    # observability resolves its check handles from these state slots
    app.state.ready_sqlite = OkConn()
    app.state.ready_qdrant = OkQdrant()
    app.state.ready_redis = DownRedis()

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["dependencies"]["sqlite"] == "ok"
    assert body["dependencies"]["qdrant"] == "ok"
    assert body["dependencies"]["redis"] == "down"
    # no connection string / credentials leaked anywhere in the response
    text = response.text
    assert "pw" not in text
    assert "10.0.0.1" not in text
    assert "6379" not in text


def test_ready_200_when_all_dependencies_up_fakes():
    app = create_app()
    app.state.ingestion_service = object()
    app.state.max_upload_bytes = 5_000_000

    class OkConn:
        def execute(self, *a, **k):
            return None

    class OkQdrant:
        def get_collections(self):
            return None

    class OkRedis:
        def ping(self):
            return True

    app.state.ready_sqlite = OkConn()
    app.state.ready_qdrant = OkQdrant()
    app.state.ready_redis = OkRedis()

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["dependencies"] == {"sqlite": "ok", "qdrant": "ok", "redis": "ok"}


# --- /health stays a dumb liveness check ---


def test_health_does_not_touch_dependencies():
    app = create_app()
    app.state.ingestion_service = object()
    app.state.max_upload_bytes = 5_000_000

    class Boom:
        def __getattr__(self, name):
            raise AssertionError(f"/health must not touch dependencies (accessed {name})")

    app.state.ready_sqlite = Boom()
    app.state.ready_qdrant = Boom()
    app.state.ready_redis = Boom()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
