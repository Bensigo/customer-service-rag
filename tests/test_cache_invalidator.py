"""Specs for the Redis cache invalidator (issue #20).

Two layers, mirroring test_cache.py / test_sessions.py:

- **Unit / fail-open** — a fake redis client (no network) proves that
  ``invalidate_document`` deletes exactly the tagged cache keys plus the
  tag set, is a clean no-op for an unknown doc, and — critically — that
  ANY redis error is swallowed (warn + return), never raised: a cache
  invalidation failure must NEVER fail an otherwise-successful ingest.
  These run everywhere, unmarked, and are the CI-gated units.

- **Integration** — against a live Redis (mirroring test_cache.py): a
  real ``SADD``-tagged entry is evicted, an unknown doc is a no-op, and
  another document's entries are left intact. Marked ``integration``;
  skipped when Redis is unreachable, unless REDIS_REQUIRED=1 (CI) makes
  that a hard failure.

The invalidator touches keys holding customer answers, so — like the
response cache and session store — nothing here is ever logged: warnings
carry the operation and doc id only, never keys, fingerprints, or
content.
"""

import logging
import os
import uuid
from urllib.parse import urlsplit

import pytest
import redis

from app.stores.cache import RedisCacheInvalidator

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


# --- unit tests over a fake redis client (no network) ---


class _FakePipeline:
    """Records DELs and replays them against the fake client on execute."""

    def __init__(self, client):
        self._client = client
        self._deletes: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def delete(self, key):
        self._deletes.append(key)
        return self

    def execute(self):
        for key in self._deletes:
            self._client.store.pop(key, None)
        return [1 for _ in self._deletes]


class _FakeRedis:
    """A minimal in-memory redis: sets are Python sets under a key."""

    def __init__(self, store=None):
        # store maps key -> set (for doc_tag) or object (for cache entries)
        self.store: dict[str, object] = store or {}
        self.deleted: list[str] = []

    def smembers(self, key):
        value = self.store.get(key)
        return set(value) if value else set()

    def pipeline(self):
        pipe = _FakePipeline(self)
        # record every delete against the client so tests can assert order
        original_delete = pipe.delete

        def delete(k):
            self.deleted.append(k)
            return original_delete(k)

        pipe.delete = delete
        return pipe

    def close(self):
        pass


class _RaisingRedis:
    """Every command raises: proves invalidate_document fails open."""

    def smembers(self, key):
        raise redis.ConnectionError("redis down")

    def pipeline(self):
        raise redis.ConnectionError("redis down")

    def close(self):
        pass


def _invalidator_with_client(client) -> RedisCacheInvalidator:
    inv = RedisCacheInvalidator.__new__(RedisCacheInvalidator)
    inv._client = client
    return inv


def test_invalidate_deletes_tagged_keys_and_the_tag_set():
    # doc-1 tags two cached answers; both plus the tag set must be deleted.
    client = _FakeRedis(
        {
            "doc_tag:doc-1": {"cache:aaa", "cache:bbb"},
            "cache:aaa": "answer-a",
            "cache:bbb": "answer-b",
        }
    )
    inv = _invalidator_with_client(client)

    inv.invalidate_document("doc-1")

    assert "cache:aaa" not in client.store
    assert "cache:bbb" not in client.store
    assert "doc_tag:doc-1" not in client.store
    # exactly the two members and the tag set were deleted, nothing else
    assert set(client.deleted) == {"cache:aaa", "cache:bbb", "doc_tag:doc-1"}


def test_invalidate_unknown_doc_is_a_clean_noop():
    client = _FakeRedis({})  # no tag set for this doc
    inv = _invalidator_with_client(client)

    # Must not raise and must not delete anything (an empty tag set means
    # nothing to evict — and no dangling tag set to remove).
    inv.invalidate_document("never-seen")

    assert client.deleted == []


def test_invalidate_leaves_other_docs_entries_intact():
    client = _FakeRedis(
        {
            "doc_tag:doc-1": {"cache:aaa"},
            "cache:aaa": "answer-a",
            "doc_tag:doc-2": {"cache:zzz"},
            "cache:zzz": "answer-z",
        }
    )
    inv = _invalidator_with_client(client)

    inv.invalidate_document("doc-1")

    # doc-2's entry and tag set are untouched.
    assert client.store["cache:zzz"] == "answer-z"
    assert client.store["doc_tag:doc-2"] == {"cache:zzz"}


def test_invalidate_fails_open_when_redis_raises(caplog):
    inv = _invalidator_with_client(_RaisingRedis())
    with caplog.at_level(logging.WARNING):
        # Must NOT raise even though every command raises: a cache
        # invalidation failure must never fail the ingest.
        inv.invalidate_document("doc-1")
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_invalidate_fails_open_does_not_log_keys_or_content(caplog):
    client = _FakeRedis(
        {"doc_tag:secretdoc": {"cache:secretfp"}, "cache:secretfp": "secret answer"}
    )

    class _RaiseOnPipeline(_FakeRedis):
        def pipeline(self):
            raise redis.ConnectionError("redis down")

    inv = _invalidator_with_client(_RaiseOnPipeline(client.store))
    with caplog.at_level(logging.WARNING):
        inv.invalidate_document("secretdoc")
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "cache:secretfp" not in logged
    assert "secret answer" not in logged


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
def invalidator(redis_client):
    inv = RedisCacheInvalidator(_REDIS_URL)
    yield inv
    inv.close()


@pytest.fixture
def keyspace(redis_client):
    """A unique doc-id namespace, cleaned up after the test."""
    token = uuid.uuid4().hex[:12]
    docs = [f"test20-a-{token}", f"test20-b-{token}"]
    fps = [f"cache:test20-{token}-{i}" for i in range(3)]
    yield {"docs": docs, "fps": fps}
    to_delete = list(fps) + [f"doc_tag:{d}" for d in docs]
    redis_client.delete(*to_delete)


class TestRedisCacheInvalidatorIntegration(_IntegrationBase):
    def test_evicts_tagged_entries_and_tag_set(self, invalidator, keyspace, redis_client):
        doc = keyspace["docs"][0]
        fp0, fp1, _ = keyspace["fps"]
        # Seed two cache entries tagged to the doc, exactly as ResponseCache
        # would: SET the entry, SADD its key onto doc_tag:{doc}.
        for fp in (fp0, fp1):
            redis_client.set(fp, "cached answer", ex=3600)
            redis_client.sadd(f"doc_tag:{doc}", fp)

        invalidator.invalidate_document(doc)

        assert redis_client.exists(fp0) == 0
        assert redis_client.exists(fp1) == 0
        assert redis_client.exists(f"doc_tag:{doc}") == 0

    def test_unknown_doc_is_a_noop(self, invalidator, keyspace):
        # No tag set exists for this doc; must not raise.
        invalidator.invalidate_document(keyspace["docs"][0])

    def test_other_docs_entries_survive(self, invalidator, keyspace, redis_client):
        doc_a, doc_b = keyspace["docs"]
        fp_a, fp_b, _ = keyspace["fps"]
        redis_client.set(fp_a, "a", ex=3600)
        redis_client.sadd(f"doc_tag:{doc_a}", fp_a)
        redis_client.set(fp_b, "b", ex=3600)
        redis_client.sadd(f"doc_tag:{doc_b}", fp_b)

        invalidator.invalidate_document(doc_a)

        # doc_b's entry and tag set are untouched.
        assert redis_client.get(fp_b) == "b"
        assert redis_client.sismember(f"doc_tag:{doc_b}", fp_b)

    def test_prunes_dangling_member_whose_entry_already_expired(
        self, invalidator, keyspace, redis_client
    ):
        # #19 follow-up: an expired cache:{fp} leaves a dangling member in
        # the tag set. Invalidation must still remove it (DEL of a missing
        # key is a harmless no-op) and drop the tag set itself.
        doc = keyspace["docs"][0]
        fp_live, fp_gone, _ = keyspace["fps"]
        redis_client.set(fp_live, "live", ex=3600)
        redis_client.sadd(f"doc_tag:{doc}", fp_live, fp_gone)  # fp_gone never SET

        invalidator.invalidate_document(doc)

        assert redis_client.exists(fp_live) == 0
        assert redis_client.exists(f"doc_tag:{doc}") == 0
