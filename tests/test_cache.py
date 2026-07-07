"""Specs for the response cache (issue #19).

Two layers of test:

- **Unit / fail-open** — a fake redis client (no network) proves the
  serialization round-trip and, critically, that ANY redis error on
  get or set is swallowed (miss / skipped write) rather than raised: a
  cache outage must never take chat down. These run everywhere, unmarked.

- **Integration** — against a live Redis (mirroring test_sessions.py):
  the JSON payload, the TTL, and the ``doc_tag:{doc_id}`` eviction sets
  #20 consumes. Marked ``integration``; skipped when Redis is
  unreachable, unless REDIS_REQUIRED=1 (CI) makes that a hard failure.

The cache stores customer-authored answers, so — like the session store
— nothing here is ever logged; warnings on failure carry the operation
name only, never keys or content.
"""

import logging
import os
import time
import uuid
from urllib.parse import urlsplit

import pytest
import redis

from app.models import SourceRef
from app.stores.cache import CachedResponse, ResponseCache

_REDIS_URL = os.environ.get("REDIS_URL", "")
_REDIS_REQUIRED = os.environ.get("REDIS_REQUIRED") == "1"


def _safe_url(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{host}{port}" if host else "(redacted)"


def _unavailable(reason: str) -> None:
    if _REDIS_REQUIRED:
        pytest.fail(f"REDIS_REQUIRED=1 but integration tests cannot run: {reason}", pytrace=False)
    pytest.skip(reason)


# --- fail-open unit tests: a fake client that raises on every command ---


class _RaisingClient:
    """Every Redis command raises: proves get/set fail open."""

    def get(self, key):
        raise redis.ConnectionError("redis down")

    def pipeline(self):
        raise redis.ConnectionError("redis down")

    def close(self):
        pass


def _cache_with_client(client) -> ResponseCache:
    cache = ResponseCache.__new__(ResponseCache)
    cache._client = client
    return cache


def _response() -> CachedResponse:
    return CachedResponse(
        answer="Use the portal to reset.",
        sources=[SourceRef(chunk_id="doc-1:1:0", doc_id="doc-1", title="Password Reset")],
    )


def test_get_fails_open_returns_none_and_logs_warning(caplog):
    cache = _cache_with_client(_RaisingClient())
    with caplog.at_level(logging.WARNING):
        result = cache.get("deadbeef")
    assert result is None
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_set_fails_open_skips_write_and_logs_warning(caplog):
    cache = _cache_with_client(_RaisingClient())
    with caplog.at_level(logging.WARNING):
        # Must not raise even though every command raises.
        cache.set("deadbeef", _response(), source_doc_ids=["doc-1"], ttl=3600)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_fail_open_does_not_log_answer_or_key_content(caplog):
    cache = _cache_with_client(_RaisingClient())
    with caplog.at_level(logging.WARNING):
        cache.get("secretfp")
        cache.set("secretfp", _response(), source_doc_ids=["doc-1"], ttl=3600)
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "Use the portal to reset." not in logged
    assert "secretfp" not in logged


def test_cached_response_json_roundtrip_is_lossless():
    # The dataclass must survive JSON serialization (what Redis stores).
    original = _response()
    restored = CachedResponse.from_json(original.to_json())
    assert restored == original


class _StubValueClient:
    """Returns a preset raw value on get; never raises."""

    def __init__(self, value):
        self._value = value

    def get(self, key):
        return self._value

    def close(self):
        pass


@pytest.mark.parametrize(
    "corrupt",
    [
        "not json at all",
        '{"answer": "x"}',  # missing "sources" -> KeyError
        '{"answer": "x", "sources": 5}',  # non-list "sources" -> TypeError
    ],
)
def test_get_corrupt_or_wrong_shaped_value_is_a_miss(corrupt, caplog):
    # A foreign/corrupt/wrong-shaped value at cache:{fp} must degrade to a
    # miss, never crash (defense in depth: only this cache writes the key).
    cache = _cache_with_client(_StubValueClient(corrupt))
    with caplog.at_level(logging.WARNING):
        assert cache.get("deadbeef") is None
    assert any(record.levelno == logging.WARNING for record in caplog.records)


# --- integration: real Redis ---


class _IntegrationBase:
    pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def redis_client():
    if not _REDIS_URL:
        _unavailable("REDIS_URL is not set")
    client = redis.Redis.from_url(
        _REDIS_URL, decode_responses=True, socket_connect_timeout=2, socket_timeout=2
    )
    try:
        client.ping()
    except (redis.RedisError, OSError) as error:
        client.close()
        _unavailable(f"Redis unreachable at {_safe_url(_REDIS_URL)}: {error}")
    yield client
    client.close()


@pytest.fixture
def cache(redis_client):
    cache = ResponseCache(_REDIS_URL)
    yield cache
    cache.close()


@pytest.fixture
def fp(redis_client):
    """A unique fingerprint, with its cache key and doc-tag sets cleaned up."""
    value = f"test19_{uuid.uuid4().hex}"
    yield value
    # Clean up the entry and any doc-tag sets this test created.
    keys = [f"cache:{value}"]
    for doc_id in ("doc-1", "doc-2"):
        keys.append(f"doc_tag:{doc_id}")
    redis_client.delete(*keys)


class TestResponseCacheIntegration(_IntegrationBase):
    def test_set_then_get_roundtrip(self, cache, fp):
        response = CachedResponse(
            answer="Reset via the portal.",
            sources=[
                SourceRef(chunk_id="doc-1:1:0", doc_id="doc-1", title="Password Reset"),
                SourceRef(chunk_id="doc-2:1:0", doc_id="doc-2", title="Billing"),
            ],
        )
        cache.set(fp, response, source_doc_ids=["doc-1", "doc-2"], ttl=3600)

        assert cache.get(fp) == response

    def test_set_creates_doc_tag_sets_for_eviction(self, cache, fp, redis_client):
        # #20 evicts by deleting the cache keys listed in doc_tag:{doc_id}.
        response = CachedResponse(
            answer="answer",
            sources=[SourceRef(chunk_id="doc-1:1:0", doc_id="doc-1", title="T")],
        )
        cache.set(fp, response, source_doc_ids=["doc-1", "doc-2"], ttl=3600)

        assert redis_client.sismember("doc_tag:doc-1", f"cache:{fp}")
        assert redis_client.sismember("doc_tag:doc-2", f"cache:{fp}")

    def test_get_missing_key_returns_none(self, cache, fp):
        assert cache.get(fp) is None

    def test_set_applies_ttl(self, cache, fp, redis_client):
        cache.set(fp, _response(), source_doc_ids=["doc-1"], ttl=1234)
        ttl = redis_client.ttl(f"cache:{fp}")
        assert 0 < ttl <= 1234

    def test_expired_entry_is_gone(self, cache, fp, redis_client):
        cache.set(fp, _response(), source_doc_ids=["doc-1"], ttl=3600)
        # Force expiry in the near future rather than sleeping a real TTL,
        # then poll until Redis reaps the key: a genuine expiry, not a delete.
        redis_client.pexpire(f"cache:{fp}", 20)
        deadline = time.monotonic() + 5
        while cache.get(fp) is not None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert cache.get(fp) is None
