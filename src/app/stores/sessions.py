"""Redis-backed conversation session store.

One Redis list per session under key "session:{session_id}", holding
the most recent turns as JSON. Every append caps the list to the newest
MAX_TURNS and restarts a TTL_SECONDS expiry clock, all inside one
MULTI/EXEC pipeline, so a session can never grow unbounded or outlive
its TTL because a step was lost between commands.

PII stance (decision log #22): customer messages are stored verbatim —
scrubbing a live conversation would destroy the very context the
assistant needs — but every session is TTL-bounded (24h) and capped
(20 turns), session ids are opaque client-generated tokens carrying no
user identity, and nothing from sessions is ever logged or indexed.
That is the scrubbing boundary CLAUDE.md requires.
"""

import json
import re

import redis

from app.models import Turn

MAX_TURNS = 20
TTL_SECONDS = 24 * 60 * 60

_SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{8,64}")


def _session_key(session_id: str) -> str:
    """The Redis key for a session, rejecting malformed ids.

    The deliberately narrow id alphabet keeps every id a single opaque
    token: no separators that could smuggle another keyspace ("session:"
    vs the response cache), no whitespace, nothing to interpret. The
    raised message never echoes the rejected value.
    """
    if not _SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError("invalid session id: must match ^[A-Za-z0-9_-]{8,64}$")
    return f"session:{session_id}"


class SessionStore:
    """Capped, TTL-scoped conversation history, one Redis list per session."""

    def __init__(self, redis_url: str) -> None:
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def close(self) -> None:
        self._client.close()

    def append_turn(self, session_id: str, turn: Turn) -> None:
        """Append one turn, atomically re-capping the history to the newest
        MAX_TURNS and restarting the TTL clock: a session expires
        TTL_SECONDS after its *last* append, not its first."""
        key = _session_key(session_id)
        payload = json.dumps({"role": turn.role, "content": turn.content})
        with self._client.pipeline() as pipe:  # MULTI/EXEC: all three or nothing
            pipe.rpush(key, payload)
            pipe.ltrim(key, -MAX_TURNS, -1)
            pipe.expire(key, TTL_SECONDS)
            pipe.execute()

    def get_history(self, session_id: str, limit: int = 10) -> list[Turn]:
        """The most recent ``limit`` turns, oldest first.

        Unknown sessions — never written or already expired — yield [],
        as does limit <= 0 (guarded explicitly: LRANGE(-0, -1) would
        return the whole list, since -0 == 0).
        """
        key = _session_key(session_id)
        if limit <= 0:
            return []
        entries = self._client.lrange(key, -limit, -1)
        return [Turn(**json.loads(entry)) for entry in entries]
