"""Versioned chunk storage on SQLite.

Every upsert of a document allocates a fresh version (latest + 1) and
atomically becomes the single active version; prior versions stay
readable — their chunk ids may still be referenced by the search
indexes — until pipeline cleanup (#11) calls delete_version. WAL mode
keeps readers unblocked during writes.

One connection per store instance, used from one thread; concurrent
processes are serialized by SQLite's single-writer locking.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from app.models import Chunk, ChunkDraft, DocumentVersion

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    active INTEGER NOT NULL,
    PRIMARY KEY (doc_id, version)
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    text TEXT NOT NULL,
    title TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_version ON chunks (doc_id, version);
"""

_CHUNK_COLUMNS = "id, doc_id, version, seq, text, title"


def _chunk_id(doc_id: str, version: int, seq: int) -> str:
    return f"{doc_id}:{version}:{seq}"


def _to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        doc_id=row["doc_id"],
        version=row["version"],
        seq=row["seq"],
        text=row["text"],
        title=row["title"],
    )


class ChunkStore:
    """SQLite-backed store of document chunks, versioned per document."""

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def upsert_document(self, doc_id: str, title: str, drafts: list[ChunkDraft]) -> DocumentVersion:
        """Store a new version of the document and make it the only active one.

        Runs as one transaction: on any failure nothing of the new version
        is persisted and the previously active version stays active.
        """
        created_at = datetime.now(UTC).isoformat()
        with self._conn:  # commits on success, rolls back on exception
            version = (self.latest_version(doc_id) or 0) + 1
            self._conn.execute("UPDATE documents SET active = 0 WHERE doc_id = ?", (doc_id,))
            self._conn.execute(
                "INSERT INTO documents (doc_id, version, title, created_at, active)"
                " VALUES (?, ?, ?, ?, 1)",
                (doc_id, version, title, created_at),
            )
            self._conn.executemany(
                f"INSERT INTO chunks ({_CHUNK_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        _chunk_id(doc_id, version, draft.seq),
                        doc_id,
                        version,
                        draft.seq,
                        draft.text,
                        title,
                    )
                    for draft in drafts
                ],
            )
        return DocumentVersion(doc_id=doc_id, version=version, chunk_count=len(drafts))

    def get_chunks(self, doc_id: str, version: int) -> list[Chunk]:
        """All chunks of one document version, in seq (document) order."""
        rows = self._conn.execute(
            f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE doc_id = ? AND version = ? ORDER BY seq",
            (doc_id, version),
        ).fetchall()
        return [_to_chunk(row) for row in rows]

    def get_chunks_by_ids(self, ids: list[str]) -> list[Chunk]:
        """Chunks for the given ids, in input order; unknown ids are skipped.

        This is the retrieval hydration path: search indexes return ranked
        ids, and the ranking must survive the round-trip.
        """
        if not ids:
            return []
        # only the placeholder count is interpolated; ids are bound parameters
        placeholders = ", ".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE id IN ({placeholders})", ids
        ).fetchall()
        by_id = {row["id"]: _to_chunk(row) for row in rows}
        return [by_id[chunk_id] for chunk_id in ids if chunk_id in by_id]

    def latest_version(self, doc_id: str) -> int | None:
        """Highest stored version of the document, or None if unknown."""
        row = self._conn.execute(
            "SELECT MAX(version) FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row[0]

    def delete_version(self, doc_id: str, version: int) -> None:
        """Remove one version's chunks and document row (rollback/supersede cleanup)."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM chunks WHERE doc_id = ? AND version = ?", (doc_id, version)
            )
            self._conn.execute(
                "DELETE FROM documents WHERE doc_id = ? AND version = ?", (doc_id, version)
            )
