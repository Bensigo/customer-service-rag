import sqlite3
from contextlib import closing
from datetime import datetime

import pytest

from app.models import Chunk, ChunkDraft, DocumentVersion
from app.stores.chunk_store import ChunkStore


def _drafts(count: int) -> list[ChunkDraft]:
    return [ChunkDraft(seq=seq, text=f"chunk text {seq}", token_estimate=4) for seq in range(count)]


def _rows(db_path: str, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    """Query on an independent connection, so only committed state is seen."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql, params).fetchall()


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "chunks.db")


@pytest.fixture
def store(db_path):
    store = ChunkStore(db_path)
    yield store
    store.close()


class TestStoreSetup:
    def test_creates_parent_directory_of_db_path(self, tmp_path):
        db_path = str(tmp_path / "data" / "stores" / "chunks.db")

        store = ChunkStore(db_path)
        try:
            assert store.upsert_document("doc", "Doc", _drafts(1)).version == 1
        finally:
            store.close()

    def test_database_is_in_wal_mode(self, store, db_path):
        # WAL mode is persistent in the database file, so an independent
        # connection observes it
        assert _rows(db_path, "PRAGMA journal_mode")[0][0] == "wal"


class TestUpsertDocument:
    def test_upsert_new_document_is_version_1(self, store, db_path):
        result = store.upsert_document("refund-policy", "Refund policy", _drafts(2))

        assert result == DocumentVersion(doc_id="refund-policy", version=1, chunk_count=2)

        row = _rows(db_path, "SELECT * FROM documents WHERE doc_id = 'refund-policy'")[0]
        assert (row["version"], row["title"], row["active"]) == (1, "Refund policy", 1)
        datetime.fromisoformat(row["created_at"])  # timestamp is recorded, sortable

    def test_chunks_are_stored_with_ids_and_returned_in_seq_order(self, store):
        # insertion order must not matter — seq defines document order
        store.upsert_document("faq", "FAQ", list(reversed(_drafts(2))))

        assert store.get_chunks("faq", 1) == [
            Chunk(id="faq:1:0", doc_id="faq", version=1, seq=0, text="chunk text 0", title="FAQ"),
            Chunk(id="faq:1:1", doc_id="faq", version=1, seq=1, text="chunk text 1", title="FAQ"),
        ]

    def test_upsert_same_doc_id_increments_version_and_deactivates_old(self, store, db_path):
        store.upsert_document("faq", "FAQ", _drafts(2))
        result = store.upsert_document("faq", "FAQ v2", _drafts(3))

        assert result == DocumentVersion(doc_id="faq", version=2, chunk_count=3)
        assert store.latest_version("faq") == 2
        # exactly one active version, and it is the new one
        active_sql = "SELECT version FROM documents WHERE doc_id = 'faq' AND active = 1"
        assert [row["version"] for row in _rows(db_path, active_sql)] == [2]
        # the old version's chunks stay readable until cleanup (#11) deletes them
        assert len(store.get_chunks("faq", 1)) == 2

    def test_versions_are_tracked_per_document(self, store, db_path):
        store.upsert_document("doc-a", "A", _drafts(1))
        store.upsert_document("doc-b", "B", _drafts(1))

        result = store.upsert_document("doc-a", "A v2", _drafts(1))

        assert result.version == 2
        assert store.latest_version("doc-b") == 1
        active_b = _rows(db_path, "SELECT active FROM documents WHERE doc_id = 'doc-b'")
        assert [row["active"] for row in active_b] == [1]

    def test_upsert_is_transactional(self, store, db_path):
        store.upsert_document("faq", "FAQ", _drafts(2))
        # two drafts with the same seq collide on chunk id "faq:2:0",
        # forcing a UNIQUE violation on the second chunk insert
        colliding = [
            ChunkDraft(seq=0, text="first", token_estimate=1),
            ChunkDraft(seq=0, text="duplicate", token_estimate=1),
        ]

        with pytest.raises(sqlite3.IntegrityError):
            store.upsert_document("faq", "FAQ v2", colliding)

        # nothing of version 2 persisted, and version 1 is still active
        assert store.latest_version("faq") == 1
        assert store.get_chunks("faq", 2) == []
        rows = _rows(db_path, "SELECT version, active FROM documents WHERE doc_id = 'faq'")
        assert [(row["version"], row["active"]) for row in rows] == [(1, 1)]


class TestGetChunks:
    def test_unknown_document_or_version_returns_empty_list(self, store):
        store.upsert_document("faq", "FAQ", _drafts(1))

        assert store.get_chunks("faq", 99) == []
        assert store.get_chunks("ghost", 1) == []


class TestGetChunksByIds:
    def test_get_chunks_by_ids_preserves_order_and_skips_missing(self, store):
        store.upsert_document("faq", "FAQ", _drafts(3))

        chunks = store.get_chunks_by_ids(["faq:1:2", "ghost:1:0", "faq:1:0"])

        assert [chunk.id for chunk in chunks] == ["faq:1:2", "faq:1:0"]

    def test_no_ids_returns_empty_list(self, store):
        assert store.get_chunks_by_ids([]) == []


class TestLatestVersion:
    def test_unknown_document_is_none(self, store):
        assert store.latest_version("ghost") is None


class TestDeleteVersion:
    def test_delete_version_removes_chunks_and_document_row(self, store, db_path):
        store.upsert_document("faq", "FAQ", _drafts(2))
        store.upsert_document("faq", "FAQ v2", _drafts(3))

        store.delete_version("faq", 1)

        assert store.get_chunks("faq", 1) == []
        assert _rows(db_path, "SELECT * FROM documents WHERE doc_id = 'faq' AND version = 1") == []
        # the other version is untouched
        assert store.latest_version("faq") == 2
        assert len(store.get_chunks("faq", 2)) == 3

    def test_delete_active_version_reactivates_the_highest_remaining(self, db_path, store):
        # rollback deletes the just-created (active) version; the previously
        # active version must become active again so the document is never
        # left with zero active versions.
        store.upsert_document("faq", "FAQ", _drafts(2))  # v1 active
        store.upsert_document("faq", "FAQ v2", _drafts(3))  # v2 active, v1 inactive

        store.delete_version("faq", 2)

        active = _rows(db_path, "SELECT version FROM documents WHERE doc_id = 'faq' AND active = 1")
        assert [row["version"] for row in active] == [1]

    def test_delete_only_version_leaves_no_active_row(self, db_path, store):
        store.upsert_document("faq", "FAQ", _drafts(2))

        store.delete_version("faq", 1)

        assert _rows(db_path, "SELECT version FROM documents WHERE doc_id = 'faq'") == []
