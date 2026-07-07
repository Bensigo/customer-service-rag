"""Specs for the ingestion service (issue #11).

Every spec runs against in-memory fakes implementing the collaborator
interfaces the service depends on (chunk store, BM25 index, embedder,
vector store, cache invalidator), so the orchestration and failure
semantics are tested without Docker, Ollama, or Qdrant. The single
end-to-end test against the real backends lives in
test_pipeline_integration.py.

The fakes mirror the real stores' contracts where the service relies on
them: the chunk store allocates latest+1 versions and builds
"{doc_id}:{version}:{seq}" chunk ids, and both indexes delete by
(doc_id, version) equality.
"""

import re
from types import SimpleNamespace

import pytest

from app.ingestion.pipeline import (
    IngestError,
    IngestionService,
    IngestResult,
    NoopCacheInvalidator,
)
from app.models import Chunk, DocumentVersion

DOC = "faq-password"
TITLE = "Password FAQ"
V1_TEXT = "To reset your password, open the login page. Click the forgot-password link."
V2_TEXT = "Refunds for damaged items arrive within five days. Contact billing for invoices."


class FakeChunkStore:
    """In-memory ChunkStore: latest+1 versioning, '{doc}:{version}:{seq}' ids."""

    def __init__(self):
        self.versions: dict[tuple[str, int], list[Chunk]] = {}

    def upsert_document(self, doc_id, title, drafts):
        version = (self.latest_version(doc_id) or 0) + 1
        self.versions[(doc_id, version)] = [
            Chunk(
                id=f"{doc_id}:{version}:{draft.seq}",
                doc_id=doc_id,
                version=version,
                seq=draft.seq,
                text=draft.text,
                title=title,
            )
            for draft in drafts
        ]
        return DocumentVersion(doc_id=doc_id, version=version, chunk_count=len(drafts))

    def get_chunks(self, doc_id, version):
        return list(self.versions.get((doc_id, version), []))

    def latest_version(self, doc_id):
        versions = [version for stored_doc, version in self.versions if stored_doc == doc_id]
        return max(versions) if versions else None

    def delete_version(self, doc_id, version):
        self.versions.pop((doc_id, version), None)


class FakeBm25Index:
    """In-memory Bm25Index: word-overlap scoring is enough to prove
    presence/absence of a version's postings."""

    def __init__(self):
        self.rows: dict[str, tuple[str, int, str]] = {}  # chunk_id -> (doc_id, version, text)

    def index_chunks(self, chunks):
        for chunk in chunks:
            self.rows[chunk.id] = (chunk.doc_id, chunk.version, chunk.text)

    def remove_document_version(self, doc_id, version):
        self.rows = {
            chunk_id: row
            for chunk_id, row in self.rows.items()
            if (row[0], row[1]) != (doc_id, version)
        }

    def search(self, query, k):
        terms = set(re.findall(r"\w+", query.lower()))
        hits = [
            (chunk_id, float(overlap))
            for chunk_id, (_, _, text) in self.rows.items()
            if (overlap := len(terms & set(re.findall(r"\w+", text.lower()))))
        ]
        hits.sort(key=lambda hit: hit[1], reverse=True)
        return hits[:k]


class FlakyBm25Index(FakeBm25Index):
    """Fails the next remove_document_version targeting one specific version."""

    def __init__(self):
        super().__init__()
        self.fail_removal_of_version: int | None = None

    def remove_document_version(self, doc_id, version):
        if version == self.fail_removal_of_version:
            self.fail_removal_of_version = None
            raise ConnectionError("sqlite went away")
        super().remove_document_version(doc_id, version)


class FakeEmbedder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_passages(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]


class FailingEmbedder:
    def embed_passages(self, texts):
        raise ConnectionError("ollama unreachable")


class FakeVectorStore:
    """In-memory VectorStore: search returns every stored point, which is
    all the presence/absence assertions here need."""

    def __init__(self):
        self.points: dict[str, tuple[str, int, list[float]]] = {}

    def upsert(self, chunks, vectors):
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.points[chunk.id] = (chunk.doc_id, chunk.version, vector)

    def delete_document_version(self, doc_id, version):
        self.points = {
            chunk_id: point
            for chunk_id, point in self.points.items()
            if (point[0], point[1]) != (doc_id, version)
        }

    def search(self, vector, k):
        if k <= 0:
            return []
        return [(chunk_id, 1.0) for chunk_id in self.points][:k]


class FlakyVectorStore(FakeVectorStore):
    """Fails the next `failures` upserts, then behaves normally."""

    def __init__(self, failures=0):
        super().__init__()
        self.remaining_failures = failures

    def upsert(self, chunks, vectors):
        if self.remaining_failures:
            self.remaining_failures -= 1
            raise ConnectionError("qdrant unreachable")
        super().upsert(chunks, vectors)


class RecordingInvalidator:
    """Spy that snapshots both indexes at invalidation time, proving the
    invalidation ran only after every write landed."""

    def __init__(self, bm25, vector_store):
        self._bm25 = bm25
        self._vector_store = vector_store
        self.calls: list[tuple[str, frozenset[str], frozenset[str]]] = []

    def invalidate_document(self, doc_id):
        self.calls.append(
            (doc_id, frozenset(self._bm25.rows), frozenset(self._vector_store.points))
        )


def _rig(*, bm25=None, embedder=None, vector_store=None):
    bm25 = bm25 if bm25 is not None else FakeBm25Index()
    vector_store = vector_store if vector_store is not None else FakeVectorStore()
    chunk_store = FakeChunkStore()
    invalidator = RecordingInvalidator(bm25, vector_store)
    service = IngestionService(
        chunk_store,
        bm25,
        embedder if embedder is not None else FakeEmbedder(),
        vector_store,
        invalidator,
    )
    return SimpleNamespace(
        service=service,
        chunk_store=chunk_store,
        bm25=bm25,
        vector_store=vector_store,
        invalidator=invalidator,
    )


def _bm25_ids(bm25, query):
    return {chunk_id for chunk_id, _ in bm25.search(query, 50)}


def _vector_ids(vector_store):
    return {chunk_id for chunk_id, _ in vector_store.search([0.0, 0.0], 50)}


class TestSuccessfulIngest:
    def test_ingest_new_document_writes_chunks_fts_and_vectors(self):
        rig = _rig()

        result = rig.service.ingest(DOC, TITLE, V1_TEXT)

        chunks = rig.chunk_store.get_chunks(DOC, 1)
        assert chunks, "chunks were not stored"
        assert result == IngestResult(doc_id=DOC, version=1, chunk_count=len(chunks))
        ids = {chunk.id for chunk in chunks}
        assert f"{DOC}:1:0" in ids
        assert _bm25_ids(rig.bm25, "reset password") == ids
        assert _vector_ids(rig.vector_store) == ids

    def test_reingesting_identical_content_creates_next_version(self):
        rig = _rig()

        first = rig.service.ingest(DOC, TITLE, V1_TEXT)
        second = rig.service.ingest(DOC, TITLE, V1_TEXT)

        assert (first.version, second.version) == (1, 2)
        ids = {chunk.id for chunk in rig.chunk_store.get_chunks(DOC, 2)}
        assert _bm25_ids(rig.bm25, "reset password") == ids
        assert _vector_ids(rig.vector_store) == ids


class TestSupersedeCleanup:
    def test_ingest_update_supersedes_old_version_in_both_indexes(self):
        rig = _rig()
        rig.service.ingest(DOC, TITLE, V1_TEXT)
        old_ids = {chunk.id for chunk in rig.chunk_store.get_chunks(DOC, 1)}

        result = rig.service.ingest(DOC, TITLE, V2_TEXT)

        assert result.version == 2
        new_ids = {chunk.id for chunk in rig.chunk_store.get_chunks(DOC, 2)}
        assert not old_ids & _bm25_ids(rig.bm25, "reset password refunds billing")
        assert not old_ids & _vector_ids(rig.vector_store)
        assert _bm25_ids(rig.bm25, "refunds billing") == new_ids
        assert _vector_ids(rig.vector_store) == new_ids
        # the pipeline also reclaims the superseded version's chunk rows
        assert rig.chunk_store.get_chunks(DOC, 1) == []

    def test_failed_supersede_cleanup_is_healed_by_the_next_ingest(self):
        bm25 = FlakyBm25Index()
        rig = _rig(bm25=bm25)
        rig.service.ingest(DOC, TITLE, V1_TEXT)
        old_ids = {chunk.id for chunk in rig.chunk_store.get_chunks(DOC, 1)}
        bm25.fail_removal_of_version = 1

        with pytest.raises(IngestError):
            rig.service.ingest(DOC, TITLE, V2_TEXT)

        # the new version is live, but the superseded one leaked...
        assert old_ids <= _bm25_ids(rig.bm25, "reset password")
        assert {chunk.id for chunk in rig.chunk_store.get_chunks(DOC, 2)} <= _vector_ids(
            rig.vector_store
        )

        # ...until the next ingest sweeps every still-present older version
        result = rig.service.ingest(DOC, TITLE, V2_TEXT)

        assert result.version == 3
        final_ids = {chunk.id for chunk in rig.chunk_store.get_chunks(DOC, 3)}
        assert _bm25_ids(rig.bm25, "reset password refunds billing") == final_ids
        assert _vector_ids(rig.vector_store) == final_ids
        assert rig.chunk_store.get_chunks(DOC, 1) == []
        assert rig.chunk_store.get_chunks(DOC, 2) == []


class TestFailureAtomicity:
    def test_vector_write_failure_rolls_back_version_and_raises_ingest_error(self):
        rig = _rig(vector_store=FlakyVectorStore(failures=1))

        with pytest.raises(IngestError) as excinfo:
            rig.service.ingest(DOC, TITLE, V1_TEXT)

        assert isinstance(excinfo.value.__cause__, ConnectionError)
        assert rig.chunk_store.get_chunks(DOC, 1) == []
        assert rig.chunk_store.latest_version(DOC) is None
        assert _bm25_ids(rig.bm25, "reset password") == set()
        assert _vector_ids(rig.vector_store) == set()

    def test_embed_failure_aborts_before_any_write(self):
        rig = _rig(embedder=FailingEmbedder())

        with pytest.raises(IngestError):
            rig.service.ingest(DOC, TITLE, V1_TEXT)

        assert rig.chunk_store.latest_version(DOC) is None
        assert rig.bm25.rows == {}
        assert rig.vector_store.points == {}

    def test_unchunkable_text_is_rejected_before_any_write(self):
        embedder = FakeEmbedder()
        rig = _rig(embedder=embedder)

        with pytest.raises(IngestError):
            rig.service.ingest(DOC, TITLE, "   \n\n  ")

        assert embedder.calls == []
        assert rig.chunk_store.latest_version(DOC) is None

    def test_failed_ingest_can_be_retried_successfully(self):
        vector_store = FlakyVectorStore(failures=0)
        rig = _rig(vector_store=vector_store)
        rig.service.ingest(DOC, TITLE, V1_TEXT)
        v1_ids = {chunk.id for chunk in rig.chunk_store.get_chunks(DOC, 1)}
        vector_store.remaining_failures = 1

        with pytest.raises(IngestError):
            rig.service.ingest(DOC, TITLE, V2_TEXT)

        # the failed update never touched the prior version: v1 still serves
        assert _bm25_ids(rig.bm25, "reset password") == v1_ids
        assert _vector_ids(rig.vector_store) == v1_ids

        result = rig.service.ingest(DOC, TITLE, V2_TEXT)

        # rollback freed version 2, so the retry re-allocates it
        assert result.version == 2
        new_ids = {chunk.id for chunk in rig.chunk_store.get_chunks(DOC, 2)}
        assert _bm25_ids(rig.bm25, "refunds billing") == new_ids
        assert _vector_ids(rig.vector_store) == new_ids

    def test_stale_leftovers_under_a_reused_version_are_cleared_before_indexing(self):
        # a partially failed rollback can leave index entries behind for a
        # version number that a later ingest re-allocates; the service must
        # clear them rather than rely on same-id overwrites, or leftover
        # higher-seq entries would survive a re-chunk into fewer chunks
        rig = _rig()
        stale = [
            Chunk(id=f"{DOC}:1:{seq}", doc_id=DOC, version=1, seq=seq, text="zombie", title=TITLE)
            for seq in range(3)
        ]
        rig.bm25.index_chunks(stale)
        rig.vector_store.upsert(stale, [[0.0, 0.0]] * 3)

        result = rig.service.ingest(DOC, TITLE, V1_TEXT)

        assert result.version == 1
        ids = {chunk.id for chunk in rig.chunk_store.get_chunks(DOC, 1)}
        assert _bm25_ids(rig.bm25, "zombie reset password") == ids
        assert _vector_ids(rig.vector_store) == ids


class TestCacheInvalidation:
    def test_invalidator_called_exactly_once_after_all_writes(self):
        rig = _rig()

        rig.service.ingest(DOC, TITLE, V1_TEXT)

        ids = {chunk.id for chunk in rig.chunk_store.get_chunks(DOC, 1)}
        [(doc_id, bm25_snapshot, vector_snapshot)] = rig.invalidator.calls
        assert doc_id == DOC
        assert ids <= bm25_snapshot, "invalidated before the BM25 write landed"
        assert ids <= vector_snapshot, "invalidated before the vector write landed"

    def test_invalidator_not_called_when_ingest_fails(self):
        rig = _rig(vector_store=FlakyVectorStore(failures=1))

        with pytest.raises(IngestError):
            rig.service.ingest(DOC, TITLE, V1_TEXT)

        assert rig.invalidator.calls == []

    def test_invalidator_not_called_when_supersede_cleanup_fails(self):
        bm25 = FlakyBm25Index()
        rig = _rig(bm25=bm25)
        rig.service.ingest(DOC, TITLE, V1_TEXT)
        bm25.fail_removal_of_version = 1

        with pytest.raises(IngestError):
            rig.service.ingest(DOC, TITLE, V2_TEXT)

        assert [call[0] for call in rig.invalidator.calls] == [DOC]

    def test_noop_invalidator_satisfies_the_protocol(self):
        assert NoopCacheInvalidator().invalidate_document(DOC) is None
