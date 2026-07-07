"""Specs for wiring the real cache invalidator into the ingest path (issue #20).

Since #11 the IngestionService has called ``invalidator.invalidate_document``
after every successful ingest, but ``create_app`` handed it a
``NoopCacheInvalidator`` (a stand-in that does nothing). This issue swaps
in the real ``RedisCacheInvalidator`` so a document update actually
evicts that document's cached answers.

These prove ``_build_ingestion_service``:

- constructs the ``IngestionService`` with a ``RedisCacheInvalidator``
  (NOT the noop), pointed at the same Redis as the response cache, and
- registers a closer for the invalidator's client so shutdown releases it.

All constructors are monkeypatched (no live Redis/Ollama/Qdrant/SQLite).
"""

import app.main as main
from app.config import Settings


class _FakeConn:
    def __init__(self, db_path):
        self.db_path = db_path

    def close(self):
        pass


class _FakeChunkStore:
    def __init__(self, db_path):
        self.db_path = db_path

    def close(self):
        pass


class _FakeBm25:
    def __init__(self, conn):
        self.conn = conn


class _FakeEmbedder:
    def dim(self):
        return 8

    def close(self):
        pass


class _FakeVectorStore:
    def __init__(self, *args, **kwargs):
        pass

    def ensure_collection(self):
        pass

    def close(self):
        pass


class _FakeInvalidator:
    """Records its Redis URL and whether it was closed."""

    def __init__(self, redis_url):
        self.redis_url = redis_url
        self.closed = False

    def invalidate_document(self, doc_id):
        pass

    def close(self):
        self.closed = True


def _patch(monkeypatch):
    monkeypatch.setattr(main, "ChunkStore", _FakeChunkStore)
    monkeypatch.setattr(main, "Bm25Index", _FakeBm25)
    monkeypatch.setattr(main, "VectorStore", _FakeVectorStore)
    monkeypatch.setattr(main, "RedisCacheInvalidator", _FakeInvalidator)
    monkeypatch.setattr(main, "create_embedder", lambda settings: _FakeEmbedder())
    monkeypatch.setattr(main.sqlite3, "connect", lambda db_path, **kw: _FakeConn(db_path))


def test_ingestion_service_uses_redis_cache_invalidator(monkeypatch):
    _patch(monkeypatch)
    settings = Settings(db_path="/tmp/rag-test.sqlite3", redis_url="redis://localhost:6379/2")

    service, _closers = main._build_ingestion_service(settings)

    invalidator = service._invalidator
    assert isinstance(invalidator, _FakeInvalidator)
    # points at the same Redis the response cache uses
    assert invalidator.redis_url == settings.redis_url


def test_ingestion_invalidator_client_is_closed_on_shutdown(monkeypatch):
    _patch(monkeypatch)
    settings = Settings(db_path="/tmp/rag-test.sqlite3")

    service, closers = main._build_ingestion_service(settings)
    invalidator = service._invalidator

    names = {name for name, _ in closers}
    assert "cache invalidator" in names
    for _, close in closers:
        close()
    assert invalidator.closed
