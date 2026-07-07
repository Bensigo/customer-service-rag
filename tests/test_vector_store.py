"""Specs for the Qdrant vector store (issue #10).

Every behavior runs against two backends via the parametrized ``store``
fixture: qdrant-client's in-process local mode (":memory:", always runs)
and a real Qdrant server (marked ``integration``, skipped unless
QDRANT_URL is set and reachable; CI provides a service container).

Server-backed runs each use a collection named after the test and delete
it in teardown, so parallel or repeated runs never pollute each other.
The dimension-mismatch spec is server-only: it needs two stores sharing
one backend, and every ":memory:" client is its own isolated world.
"""

import os
import urllib.request

import pytest
from qdrant_client import QdrantClient, models

from app.models import Chunk
from app.stores.vector_store import VectorStore

# Read at import (collection) time: the autouse hermetic_settings fixture
# scrubs QDRANT_URL from the environment before each test runs.
_QDRANT_URL = os.environ.get("QDRANT_URL", "").rstrip("/")

DIM = 8


def _server_reachable() -> bool:
    if not _QDRANT_URL:
        return False
    try:
        with urllib.request.urlopen(f"{_QDRANT_URL}/readyz", timeout=2):
            return True
    except OSError:
        return False


_SERVER_UP = _server_reachable()
_SERVER_MARKS = [
    pytest.mark.integration,
    pytest.mark.skipif(not _SERVER_UP, reason="QDRANT_URL unset or Qdrant unreachable"),
]
_BACKENDS = [
    pytest.param("memory", id="memory"),
    pytest.param("server", id="server", marks=_SERVER_MARKS),
]


def _chunk(doc_id: str, version: int, seq: int) -> Chunk:
    return Chunk(
        id=f"{doc_id}:{version}:{seq}",
        doc_id=doc_id,
        version=version,
        seq=seq,
        text=f"chunk {seq} of {doc_id} v{version}",
        title=doc_id,
    )


def _axis(i: int) -> list[float]:
    """Unit vector along axis i: orthogonal to every other axis, so under
    cosine distance the matching chunk scores 1.0 and all others 0.0."""
    vector = [0.0] * DIM
    vector[i] = 1.0
    return vector


def _ids(results: list[tuple[str, float]]) -> list[str]:
    return [chunk_id for chunk_id, _ in results]


def _delete_collection(name: str) -> None:
    client = QdrantClient(url=_QDRANT_URL)
    try:
        client.delete_collection(name)
    finally:
        client.close()


@pytest.fixture(params=_BACKENDS)
def store(request):
    if request.param == "memory":
        store = VectorStore(":memory:", vector_size=DIM)
        store.ensure_collection()
        yield store
        store.close()
        return
    collection = f"test10_{request.node.originalname}"
    store = VectorStore(_QDRANT_URL, vector_size=DIM, collection=collection)
    store.ensure_collection()
    yield store
    store.close()
    _delete_collection(collection)


class TestEnsureCollection:
    def test_ensure_collection_is_idempotent(self, store):
        store.upsert([_chunk("faq", 1, 0)], [_axis(0)])

        store.ensure_collection()  # repeat call: no error...
        store.ensure_collection()

        # ...and no silent recreate: the point indexed before survived
        [(chunk_id, score)] = store.search(_axis(0), k=1)
        assert chunk_id == "faq:1:0"
        assert score == pytest.approx(1.0)

    def test_ensure_collection_survives_losing_a_create_race(self, store, monkeypatch):
        # two services can bootstrap concurrently: the other one creates the
        # collection between our exists-check and our create. Simulate losing
        # that race by forcing the exists-probe to report "missing" while the
        # collection (created by the fixture) is already there.
        store.upsert([_chunk("faq", 1, 0)], [_axis(0)])
        monkeypatch.setattr(QdrantClient, "collection_exists", lambda self, *a, **kw: False)

        store.ensure_collection()  # idempotent no-op, not a 409/already-exists error

        # ...and it validated the existing collection instead of recreating it
        assert _ids(store.search(_axis(0), k=1)) == ["faq:1:0"]

    @pytest.mark.integration
    @pytest.mark.skipif(not _SERVER_UP, reason="QDRANT_URL unset or Qdrant unreachable")
    def test_ensure_collection_dim_mismatch_fails_loudly(self):
        collection = "test10_dim_mismatch"
        original = VectorStore(_QDRANT_URL, vector_size=DIM, collection=collection)
        try:
            original.ensure_collection()
            original.upsert([_chunk("faq", 1, 0)], [_axis(0)])

            wrong = VectorStore(_QDRANT_URL, vector_size=DIM * 2, collection=collection)
            try:
                with pytest.raises(ValueError) as excinfo:
                    wrong.ensure_collection()
            finally:
                wrong.close()

            # the error names the collection and both dimensions...
            message = str(excinfo.value)
            assert collection in message
            assert str(DIM) in message
            assert str(DIM * 2) in message
            # ...and nothing was recreated or dropped along the way
            assert _ids(original.search(_axis(0), k=1)) == ["faq:1:0"]
        finally:
            original.close()
            _delete_collection(collection)

    @pytest.mark.integration
    @pytest.mark.skipif(not _SERVER_UP, reason="QDRANT_URL unset or Qdrant unreachable")
    def test_ensure_collection_foreign_vector_config_fails_loudly(self):
        # a same-named collection created elsewhere with *named* vectors has
        # no single dimension to compare against; ensure_collection must
        # still raise the clear ValueError, not an AttributeError from
        # poking .size on a dict
        collection = "test10_foreign_vector_config"
        client = QdrantClient(url=_QDRANT_URL)
        try:
            client.create_collection(
                collection_name=collection,
                vectors_config={
                    "text": models.VectorParams(size=DIM, distance=models.Distance.COSINE)
                },
            )
            store = VectorStore(_QDRANT_URL, vector_size=DIM, collection=collection)
            try:
                with pytest.raises(ValueError, match=collection):
                    store.ensure_collection()
            finally:
                store.close()
        finally:
            client.delete_collection(collection)
            client.close()


class TestUpsertAndSearch:
    def test_upsert_then_search_returns_chunk_id_with_score(self, store):
        store.upsert([_chunk("faq", 1, seq) for seq in range(3)], [_axis(seq) for seq in range(3)])

        results = store.search(_axis(1), k=3)

        assert _ids(results)[0] == "faq:1:1"
        scores = [score for _, score in results]
        assert scores[0] == pytest.approx(1.0)  # cosine similarity of the exact vector
        assert scores == sorted(scores, reverse=True)  # higher-is-better ordering

    def test_upsert_same_chunks_twice_does_not_duplicate_points(self, store):
        chunks = [_chunk("faq", 1, seq) for seq in range(2)]
        vectors = [_axis(seq) for seq in range(2)]

        store.upsert(chunks, vectors)
        store.upsert(chunks, vectors)

        # random (non-deterministic) point ids would leave 4 points behind
        results = store.search(_axis(0), k=10)
        assert sorted(_ids(results)) == ["faq:1:0", "faq:1:1"]

    def test_upsert_rejects_mismatched_chunks_and_vectors(self, store):
        with pytest.raises(ValueError):
            store.upsert([_chunk("faq", 1, 0)], [])

    def test_upsert_with_no_chunks_is_a_no_op(self, store):
        store.upsert([], [])

        assert store.search(_axis(0), k=1) == []

    def test_search_k_limits_results(self, store):
        store.upsert([_chunk("faq", 1, seq) for seq in range(5)], [_axis(seq) for seq in range(5)])

        assert len(store.search(_axis(0), k=3)) == 3
        assert len(store.search(_axis(0), k=10)) == 5

    def test_search_nonpositive_k_returns_empty_list(self, store):
        # mirrors Bm25Index, so the fusion layer (#13) treats both alike
        store.upsert([_chunk("faq", 1, 0)], [_axis(0)])

        assert store.search(_axis(0), k=0) == []
        assert store.search(_axis(0), k=-1) == []


class TestDeleteDocumentVersion:
    def test_delete_document_version_removes_only_that_version(self, store):
        store.upsert(
            [_chunk("faq", 1, 0), _chunk("faq", 1, 1), _chunk("faq", 2, 0), _chunk("other", 1, 0)],
            [_axis(0), _axis(1), _axis(2), _axis(3)],
        )

        store.delete_document_version("faq", 1)

        remaining = store.search(_axis(0), k=10)
        assert sorted(_ids(remaining)) == ["faq:2:0", "other:1:0"]

    def test_delete_document_version_twice_is_a_no_op(self, store):
        store.upsert([_chunk("faq", 1, 0), _chunk("other", 1, 0)], [_axis(0), _axis(1)])

        store.delete_document_version("faq", 1)
        store.delete_document_version("faq", 1)  # already gone: no error

        assert _ids(store.search(_axis(1), k=10)) == ["other:1:0"]
