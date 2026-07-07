"""Specs for wiring the chat/retrieval stack into create_app (issue #18).

The threading hazard from #12's review: the chat path serves retrieval
reads concurrently on the threadpool, so it must NOT share the ingest
path's single SQLite connections. These tests monkeypatch the store and
client constructors (no live services) to prove that
``_build_chat_stack`` gives retrieval its OWN chunk-store/BM25
connections, wires every collaborator, and registers closers for all of
them.
"""

import app.main as main
from app.config import Settings


class _FakeConn:
    def __init__(self, db_path):
        self.db_path = db_path
        self.closed = False

    def close(self):
        self.closed = True


class _FakeChunkStore:
    instances = []

    def __init__(self, db_path):
        self.db_path = db_path
        self.closed = False
        _FakeChunkStore.instances.append(self)

    def get_chunks_by_ids(self, ids):
        return []

    def close(self):
        self.closed = True


class _FakeBm25:
    def __init__(self, conn):
        self.conn = conn

    def search(self, query, k):
        return []


class _FakeEmbedder:
    def __init__(self):
        self.closed = False

    def dim(self):
        return 8

    def embed_query(self, text):
        return [0.0] * 8

    def close(self):
        self.closed = True


class _FakeVectorStore:
    def __init__(self, *args, **kwargs):
        self.closed = False

    def ensure_collection(self):
        pass

    def search(self, vector, k):
        return []

    def close(self):
        self.closed = True


class _FakeSessionStore:
    def __init__(self, redis_url):
        self.redis_url = redis_url
        self.closed = False

    def close(self):
        self.closed = True


class _FakeReranker:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeLLM:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _patch(monkeypatch):
    _FakeChunkStore.instances = []
    monkeypatch.setattr(main, "ChunkStore", _FakeChunkStore)
    monkeypatch.setattr(main, "Bm25Index", _FakeBm25)
    monkeypatch.setattr(main, "VectorStore", _FakeVectorStore)
    monkeypatch.setattr(main, "SessionStore", _FakeSessionStore)
    monkeypatch.setattr(main, "create_embedder", lambda settings: _FakeEmbedder())
    monkeypatch.setattr(main, "create_reranker", lambda settings: _FakeReranker())
    monkeypatch.setattr(main, "create_llm_client", lambda settings: _FakeLLM())
    monkeypatch.setattr(main.sqlite3, "connect", lambda db_path, **kw: _FakeConn(db_path))


def test_build_chat_stack_wires_all_collaborators(monkeypatch):
    _patch(monkeypatch)
    settings = Settings(db_path="/tmp/rag-test.sqlite3")

    state, closers = main._build_chat_stack(settings)

    assert isinstance(state["session_store"], _FakeSessionStore)
    assert state["session_store"].redis_url == settings.redis_url
    assert isinstance(state["reranker"], _FakeReranker)
    assert isinstance(state["llm_client"], _FakeLLM)
    # the retriever seam exposes .retrieve (the pool)
    assert hasattr(state["retriever"], "retrieve")
    # every collaborator has a registered closer
    names = {name for name, _ in closers}
    assert {"session store", "reranker", "retriever pool", "llm client"} <= names
    # close them all
    for _, close in closers:
        close()
    assert state["session_store"].closed
    assert state["reranker"].closed


def test_retrieval_uses_its_own_chunk_store_connections(monkeypatch):
    _patch(monkeypatch)
    settings = Settings(db_path="/tmp/rag-test.sqlite3")

    state, closers = main._build_chat_stack(settings)

    # Exercise the pool so at least one retriever (and its chunk store) is
    # built — a chunk store dedicated to the read path, not the ingest one.
    state["retriever"].retrieve("hi")

    assert _FakeChunkStore.instances, "no read-path chunk store was built"
    for store in _FakeChunkStore.instances:
        assert store.db_path == settings.db_path
    # closing the pool closes the read connections it built
    for name, close in closers:
        if name == "retriever pool":
            close()
    assert all(s.closed for s in _FakeChunkStore.instances)
