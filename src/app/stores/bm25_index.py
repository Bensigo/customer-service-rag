"""BM25 full-text index of chunk text on SQLite FTS5.

Table mode: chunks_fts is a *contentless-delete* FTS5 table
(content='' with contentless_delete=1, plus contentless_unindexed=1 so
the chunk_id key survives a round-trip) rather than external-content.
Chunk text already lives authoritatively in the chunk store's "chunks"
table (#6) and hits are hydrated via ChunkStore.get_chunks_by_ids, so
this table keeps only the inverted index and the chunk id - never a
second copy of the text. External-content mode would instead couple this
index to the chunks table's schema and rowids and silently return
garbage whenever the two drifted apart; this class only receives a
connection and cannot guarantee that table even exists. The chosen
options require SQLite >= 3.47 (2024-10), which the interpreter's
bundled SQLite satisfies.

Score normalization: FTS5's bm25() - aliased by "rank" - returns
*negative* values where lower means a better match, so results are
ordered by raw rank ascending (best first) and the score is returned
negated. Callers always see positive, higher-is-better floats whose
magnitudes are relative to the current corpus and query - they are not
comparable across queries.
"""

import re
import sqlite3

from app.models import Chunk

_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    doc_id UNINDEXED,
    version UNINDEXED,
    text,
    content='',
    contentless_delete=1,
    contentless_unindexed=1
)
"""

_MIN_SQLITE = (3, 47, 0)

_TERM_RE = re.compile(r"\w+")


def _to_match_expression(query: str) -> str:
    """Reduce a raw user query to safe FTS5 MATCH syntax.

    Keeps only word-character runs, double-quotes each term, and joins
    them with OR, so quotes, operators (AND, NOT, NEAR, *, -) and
    column filters in user input are matched as text instead of being
    parsed as query syntax. Returns "" when no terms remain.
    """
    terms = _TERM_RE.findall(query)
    return " OR ".join(f'"{term}"' for term in terms)


class Bm25Index:
    """BM25 postings over chunk text, keyed by chunk id, on a caller-owned connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        if sqlite3.sqlite_version_info < _MIN_SQLITE:
            raise RuntimeError(
                "Bm25Index requires SQLite >= 3.47 for contentless-delete FTS5; "
                f"this interpreter bundles {sqlite3.sqlite_version}"
            )
        self._conn = conn
        conn.execute(_SCHEMA)

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index the chunks' text; a repeated chunk id (across or within calls)
        replaces the previous row, never duplicates."""
        deduped = list({chunk.id: chunk for chunk in chunks}.values())
        with self._conn:  # commits on success, rolls back on exception
            self._conn.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id = ?",
                [(chunk.id,) for chunk in deduped],
            )
            self._conn.executemany(
                "INSERT INTO chunks_fts (chunk_id, doc_id, version, text) VALUES (?, ?, ?, ?)",
                [(chunk.id, chunk.doc_id, chunk.version, chunk.text) for chunk in deduped],
            )

    def remove_document_version(self, doc_id: str, version: int) -> None:
        """Drop the postings of every chunk of (doc_id, version); missing ones are a no-op.

        Deletes by equality on the stored doc_id and version columns —
        never by parsing or prefix-matching the composite chunk id, which
        would conflate doc ids containing ":" (e.g. doc "a" version 1
        vs doc "a:1").
        """
        with self._conn:
            self._conn.execute(
                "DELETE FROM chunks_fts WHERE doc_id = ? AND version = ?",
                (doc_id, version),
            )

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k (chunk_id, score) for the query, scores positive and higher-is-better.

        A query with no indexable terms (empty, only symbols or bare
        FTS5 operators) returns [] rather than raising. k <= 0 also
        returns [] - notably, a negative value must never reach LIMIT,
        where SQLite would read it as "no limit".
        """
        if k <= 0:
            return []
        match = _to_match_expression(query)
        if not match:
            return []
        rows = self._conn.execute(
            "SELECT chunk_id, -rank FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (match, k),
        ).fetchall()
        return [(chunk_id, score) for chunk_id, score in rows]
