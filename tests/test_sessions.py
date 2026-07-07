"""Specs for the Redis conversation session store (issue #16).

Sessions are ephemeral by design: one capped, TTL-scoped Redis list per
session under key "session:{session_id}". The store's semantics are
Redis semantics (RPUSH/LTRIM/EXPIRE done atomically), so the real
backend is the spec: these tests run against a live Redis, marked
``integration``, instead of adding a fakeredis dev dependency that
would only re-implement what is being tested.

Reachability is probed inside the module-scoped fixture — never at
import — so unit-only runs touch no network. Locally the integration
tests skip when REDIS_URL is unset or Redis is unreachable; CI provides
a service container and sets REDIS_REQUIRED=1, which turns an
unreachable Redis into a hard failure so a misconfigured pipeline
cannot silently skip every integration test and still pass.

Session-id validation raises before any command is issued and the
client connects lazily, so those specs run everywhere, unmarked.

Each test writes under a session id containing a fresh UUID and deletes
its key in teardown, so aborted, repeated, or concurrent runs never
pollute each other.
"""

import os
import uuid
from urllib.parse import urlsplit

import pytest
import redis

from app.models import Turn
from app.stores.sessions import SessionStore

# Read at collection time: the autouse hermetic_settings fixture scrubs
# REDIS_URL (a Settings field) from the environment before each test runs.
_REDIS_URL = os.environ.get("REDIS_URL", "")
_REDIS_REQUIRED = os.environ.get("REDIS_REQUIRED") == "1"

_DAY = 24 * 60 * 60


def _key(session_id: str) -> str:
    return f"session:{session_id}"


def _safe_url(url: str) -> str:
    """Return scheme://host:port only, dropping any userinfo so credentials
    embedded in REDIS_URL never reach a skip/fail message or CI log."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{host}{port}" if host else "(redacted)"


def _unavailable(reason: str) -> None:
    """Skip when Redis is unavailable, unless REDIS_REQUIRED=1 (set in CI),
    where that becomes a hard failure — a misconfigured pipeline must not
    pass by silently skipping integration."""
    if _REDIS_REQUIRED:
        pytest.fail(f"REDIS_REQUIRED=1 but integration tests cannot run: {reason}", pytrace=False)
    pytest.skip(reason)


@pytest.fixture(scope="module")
def redis_client():
    # Probed here, not at import, so unit-only runs never touch the network.
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
def store(redis_client):
    store = SessionStore(_REDIS_URL)
    yield store
    store.close()


@pytest.fixture
def session_id(redis_client):
    sid = f"test16_{uuid.uuid4().hex}"
    yield sid
    redis_client.delete(_key(sid))


def _turns(n: int) -> list[Turn]:
    return [
        Turn(role="user" if i % 2 == 0 else "assistant", content=f"message {i}") for i in range(n)
    ]


class TestSessionStore:
    pytestmark = pytest.mark.integration

    def test_append_then_get_history_roundtrip_chronological(self, store, session_id):
        turns = [
            Turn(role="user", content="How do I reset my password?"),
            Turn(role="assistant", content="Click 'Forgot password' on the login page."),
            Turn(role="user", content="That worked, thanks!"),
        ]

        for turn in turns:
            store.append_turn(session_id, turn)

        assert store.get_history(session_id) == turns

    def test_history_capped_at_20_turns(self, store, session_id):
        for turn in _turns(25):
            store.append_turn(session_id, turn)

        history = store.get_history(session_id, limit=25)

        assert len(history) == 20
        assert history[0].content == "message 5"  # the oldest 5 are gone
        assert history[-1].content == "message 24"

    def test_get_history_limit_returns_most_recent(self, store, session_id):
        for turn in _turns(12):
            store.append_turn(session_id, turn)

        recent = store.get_history(session_id, limit=3)
        assert [turn.content for turn in recent] == ["message 9", "message 10", "message 11"]

        by_default = store.get_history(session_id)  # default limit is 10
        assert [turn.content for turn in by_default] == [f"message {i}" for i in range(2, 12)]

    def test_get_history_nonpositive_limit_returns_empty(self, store, session_id):
        # LRANGE index arithmetic makes limit=0 the dangerous case: a naive
        # (-limit, -1) range would return the *whole* list, since -0 == 0.
        store.append_turn(session_id, Turn(role="user", content="hello"))

        assert store.get_history(session_id, limit=0) == []
        assert store.get_history(session_id, limit=-3) == []

    def test_ttl_set_and_refreshed_on_append(self, store, redis_client, session_id):
        store.append_turn(session_id, Turn(role="user", content="hello"))

        initial = redis_client.ttl(_key(session_id))
        assert 0 < initial <= _DAY
        assert initial > _DAY - 60  # freshly stamped to ~24h, not left unbounded

        redis_client.expire(_key(session_id), 60)  # an aging session, near expiry
        store.append_turn(session_id, Turn(role="assistant", content="hi"))

        refreshed = redis_client.ttl(_key(session_id))
        assert refreshed > _DAY - 60  # the append restarted the 24h clock

    def test_unknown_session_returns_empty_history(self, store):
        assert store.get_history(f"never_{uuid.uuid4().hex}") == []
        assert store.get_history("a" * 8) == []  # min-length boundary id is valid
        assert store.get_history("a" * 64) == []  # max-length boundary id is valid


class TestSessionIdValidation:
    """Session ids are opaque and client-generated: anything outside
    ^[A-Za-z0-9_-]{8,64}$ is rejected before any Redis command is issued,
    so these specs run everywhere, unmarked."""

    INVALID_IDS = [
        pytest.param("", id="empty"),
        pytest.param("a" * 7, id="too-short"),
        pytest.param("a" * 65, id="too-long"),
        pytest.param("has spaces99", id="space"),
        pytest.param("session:12345", id="colon-keyspace-injection"),
        pytest.param("Ünïcode-id-99", id="non-ascii"),
        pytest.param("../../etc/pwd", id="traversal"),
        pytest.param("id\nwith_newline", id="newline"),
    ]

    @pytest.fixture
    def store(self):
        # Never pinged: the lazy client makes no connection when every
        # call is rejected up front.
        store = SessionStore("redis://127.0.0.1:6379/0")
        yield store
        store.close()

    @pytest.mark.parametrize("bad_id", INVALID_IDS)
    def test_append_turn_invalid_session_id_raises_value_error(self, store, bad_id):
        with pytest.raises(ValueError, match="session id"):
            store.append_turn(bad_id, Turn(role="user", content="hi"))

    @pytest.mark.parametrize("bad_id", INVALID_IDS)
    def test_get_history_invalid_session_id_raises_value_error(self, store, bad_id):
        with pytest.raises(ValueError, match="session id"):
            store.get_history(bad_id)

    @pytest.mark.parametrize("bad_id", INVALID_IDS)
    def test_invalid_session_id_rejected_even_with_nonpositive_limit(self, store, bad_id):
        # validation must come before the limit<=0 early-out
        with pytest.raises(ValueError, match="session id"):
            store.get_history(bad_id, limit=0)


def test_safe_url_strips_credentials():
    # Redis URLs commonly embed credentials; a skip/fail message built from
    # the raw URL would leak them into CI logs.
    from tests.test_sessions import _safe_url

    safe = _safe_url("redis://user:secretpass@redis.example:6379/0")

    assert "secretpass" not in safe
    assert "user" not in safe
    assert "redis.example:6379" in safe
