"""Redis-backed first-turn response cache (issue #19).

A cached answer lives under key ``cache:{fingerprint}`` as a JSON blob
(the answer plus its source citations) with a TTL, so identical
first-turn questions skip retrieval, rerank, and the LLM entirely.

Keyspace — the eviction contract #20 consumes:

    cache:{fingerprint}      -> JSON {"answer": str, "sources": [SourceRef...]}
                                (SET ... EX ttl)
    doc_tag:{doc_id}         -> SET of "cache:{fingerprint}" keys whose
                                cached answer cited that document

On every ``set``, the entry is written and each of its source documents
gets ``SADD doc_tag:{doc_id} cache:{fingerprint}`` — all in one pipeline,
so the tags and the entry land together. #20's invalidator, given a
changed ``doc_id``, reads ``doc_tag:{doc_id}`` and deletes every listed
``cache:{...}`` key (then the tag set), so a stale answer is dropped the
moment its source document changes. **Get the key strings exactly right:
#20 deletes through these literal names.**

``doc_id`` is not re-validated here: it is held to ``[a-z0-9][a-z0-9-]{0,63}``
at the document-upload API (its single ingest choke point), an alphabet
with no ``:`` or whitespace, so a ``doc_tag:{doc_id}`` key can never
collide with the ``cache:`` or ``session:`` keyspaces.

Fail-open (non-negotiable): ANY Redis error on ``get`` is treated as a
miss (return None, warn); ANY error on ``set`` skips the write (warn). A
cache outage must never take chat down — the caller falls through to the
live pipeline. Warnings carry the operation name only: the cache holds
customer answers, so no key or content is ever logged (CLAUDE.md).
"""

import json
import logging
from dataclasses import dataclass

import redis

from app.models import SourceRef

logger = logging.getLogger("app.stores.cache")


def _cache_key(fingerprint: str) -> str:
    return f"cache:{fingerprint}"


def _doc_tag_key(doc_id: str) -> str:
    return f"doc_tag:{doc_id}"


@dataclass(frozen=True, slots=True)
class CachedResponse:
    """A cached chat answer with its source citations, JSON-serializable
    for storage under ``cache:{fingerprint}``."""

    answer: str
    sources: list[SourceRef]

    def to_json(self) -> str:
        return json.dumps(
            {
                "answer": self.answer,
                "sources": [
                    {"chunk_id": s.chunk_id, "doc_id": s.doc_id, "title": s.title}
                    for s in self.sources
                ],
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> "CachedResponse":
        data = json.loads(raw)
        sources = [
            SourceRef(chunk_id=s["chunk_id"], doc_id=s["doc_id"], title=s["title"])
            for s in data["sources"]
        ]
        return cls(answer=data["answer"], sources=sources)


class ResponseCache:
    """Read-through response cache over Redis, fail-open on every error."""

    def __init__(self, redis_url: str) -> None:
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def close(self) -> None:
        self._client.close()

    def get(self, fp: str) -> CachedResponse | None:
        """The cached response for ``fp``, or None on miss OR any Redis
        error (fail-open: a cache outage is a miss, never an exception)."""
        try:
            raw = self._client.get(_cache_key(fp))
        except redis.RedisError:
            # Fail open: a cache-read outage must degrade to a live answer,
            # never surface. Log the operation only — never the key/content.
            logger.warning("response cache get failed; treating as miss", exc_info=False)
            return None
        if raw is None:
            return None
        try:
            return CachedResponse.from_json(raw)
        except (ValueError, KeyError, TypeError):
            # A corrupt/foreign/wrong-shaped value at this key is also a miss,
            # not a crash (TypeError guards e.g. a non-list "sources").
            logger.warning("response cache entry unparseable; treating as miss")
            return None

    def set(self, fp: str, response: CachedResponse, source_doc_ids: list[str], ttl: int) -> None:
        """Store ``response`` under ``cache:{fp}`` with ``ttl`` seconds and
        tag it under each ``doc_tag:{doc_id}`` so #20 can evict by document.

        The SET and every SADD run in one pipeline so the entry and its
        eviction tags land together. Any Redis error skips the write
        (fail-open): a cache that cannot store must not break the answer.
        """
        key = _cache_key(fp)
        try:
            with self._client.pipeline() as pipe:
                pipe.set(key, response.to_json(), ex=ttl)
                for doc_id in source_doc_ids:
                    pipe.sadd(_doc_tag_key(doc_id), key)
                pipe.execute()
        except redis.RedisError:
            # Fail open: skip the write. The caller already has a live answer;
            # a failed cache write must not surface. Operation name only.
            logger.warning("response cache set failed; skipping write", exc_info=False)


class RedisCacheInvalidator:
    """Evicts a document's cached answers, implementing the ``CacheInvalidator``
    Protocol the ingestion pipeline (#11) calls after a successful ingest.

    ``invalidate_document(doc_id)`` reads the eviction tag set
    ``doc_tag:{doc_id}`` — whose members are ``cache:{fingerprint}`` key
    strings ResponseCache tagged (#19) — and, in one pipeline, ``DEL``s
    every listed cache key plus the tag set itself. A missing/empty tag
    set is a clean no-op. Because the members are stored as their full
    ``cache:`` key strings, no fingerprint is ever reconstructed here — the
    key strings must match ResponseCache's exactly, which they do via the
    shared ``_cache_key`` / ``_doc_tag_key`` helpers.

    Deleting the tag set on every invalidation also prunes the dangling
    members #19 flagged: once a ``cache:{fp}`` has expired under its TTL,
    its entry in ``doc_tag:{doc_id}`` lingers (SADD sets no TTL), and a
    ``DEL`` of an already-gone key is a harmless no-op — so an updated
    document's tag set is reclaimed cleanly. Residual growth remains only
    for a document that is *never* re-ingested: its tag set is never read,
    so expired members accumulate slowly there until the doc is updated.

    Fail-open (non-negotiable): the pipeline calls this after the writes
    and supersede sweep already succeeded — the new version is live. A
    Redis error here must therefore NEVER raise: it would turn a good
    ingest into a failure over nothing worse than a too-stale cache served
    until its TTL. Any Redis error is logged (operation + doc id only,
    never keys or content) and swallowed, so the pipeline's own
    invalidator-error wrap never triggers.
    """

    def __init__(self, redis_url: str) -> None:
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def close(self) -> None:
        self._client.close()

    def invalidate_document(self, doc_id: str) -> None:
        """Drop every cached answer tagged to ``doc_id`` and its tag set.

        Reads ``doc_tag:{doc_id}`` then pipelines a ``DEL`` of every member
        cache key and the tag set. Fail-open on ANY Redis error: warn and
        return, never raise (a failed eviction must not fail the ingest).
        """
        tag_key = _doc_tag_key(doc_id)
        try:
            members = self._client.smembers(tag_key)
            if not members:
                # No tag set (unknown doc) or an empty one: nothing cached
                # to evict and no tag set to reclaim — a clean no-op.
                return
            with self._client.pipeline() as pipe:
                for member in members:
                    pipe.delete(member)
                pipe.delete(tag_key)
                pipe.execute()
        except redis.RedisError:
            # Fail open: a cache-invalidation outage must never fail an
            # otherwise-successful ingest — the new version is already live.
            # Worst case a too-stale answer is served until its TTL. Log the
            # operation and doc id only, never keys/fingerprints/content.
            logger.warning(
                "cache invalidation failed for doc_id=%s; cached answers linger until TTL",
                doc_id,
                exc_info=False,
            )
