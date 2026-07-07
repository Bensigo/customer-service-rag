"""Specs for the BM25 index on SQLite FTS5 (issue #7).

The two FTS5 gotchas are encoded as regression tests, not left to be
rediscovered while debugging:

1. bm25() returns negative scores where lower is better, so a naive
   ORDER BY rank DESC returns the worst matches first
   (test_search_ranks_exact_term_match_first) and raw scores leak the
   wrong sign (test_search_scores_are_positive_and_sorted_descending).
2. Raw user queries can be invalid FTS5 query syntax - quotes,
   operators, column filters - and must be sanitized, never raised
   (test_search_with_quotes_and_operators_does_not_raise,
   test_search_empty_or_symbol_only_query_returns_empty_list).
"""

import sqlite3

import pytest

from app.models import Chunk
from app.stores.bm25_index import Bm25Index


def _chunk(doc_id: str, version: int, seq: int, text: str) -> Chunk:
    return Chunk(
        id=f"{doc_id}:{version}:{seq}",
        doc_id=doc_id,
        version=version,
        seq=seq,
        text=text,
        title=doc_id,
    )


# Five chunks with "refund" in only two of them, so the term keeps a
# positive BM25 idf. The heavy chunk repeats the term and must outrank
# the light one; the fillers must not match at all.
_REFUND_HEAVY = _chunk(
    "returns", 1, 0, "To request a refund, fill the refund form; refund approvals take two days."
)
_REFUND_LIGHT = _chunk(
    "shipping", 1, 0, "Shipping delays sometimes qualify for a refund under the delivery promise."
)
_CORPUS = [
    _REFUND_HEAVY,
    _REFUND_LIGHT,
    _chunk("password", 1, 0, "Reset your password from the account settings page."),
    _chunk("billing", 1, 0, "Invoices are emailed on the first day of each month."),
    _chunk("contact", 1, 0, "Contact support via chat between 9am and 5pm on weekdays."),
]


def _ids(results: list[tuple[str, float]]) -> list[str]:
    return [chunk_id for chunk_id, _ in results]


@pytest.fixture
def conn(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "index.db"))
    yield conn
    conn.close()


@pytest.fixture
def index(conn) -> Bm25Index:
    return Bm25Index(conn)


class TestSetup:
    def test_init_reuses_existing_table_and_indexed_chunks_persist(self, tmp_path):
        db_path = str(tmp_path / "index.db")
        writer = sqlite3.connect(db_path)
        try:
            Bm25Index(writer).index_chunks([_REFUND_HEAVY])
        finally:
            writer.close()

        reader = sqlite3.connect(db_path)
        try:
            # second init must not fail on, or wipe, the existing table
            assert _ids(Bm25Index(reader).search("refund", k=5)) == [_REFUND_HEAVY.id]
        finally:
            reader.close()


class TestSearch:
    def test_search_ranks_exact_term_match_first(self, index):
        index.index_chunks(_CORPUS)

        results = index.search("refund", k=5)

        # bm25() is negative and lower-is-better: with ORDER BY rank DESC
        # the light match would come first and this assertion fails
        assert _ids(results) == [_REFUND_HEAVY.id, _REFUND_LIGHT.id]

    def test_search_scores_are_positive_and_sorted_descending(self, index):
        index.index_chunks(_CORPUS)

        scores = [score for _, score in index.search("refund", k=5)]

        assert len(scores) == 2
        assert all(score > 0 for score in scores)
        assert scores == sorted(scores, reverse=True)

    def test_search_returns_at_most_k(self, index):
        index.index_chunks(
            [_chunk("faq", 1, seq, f"billing question number {seq}") for seq in range(5)]
        )

        assert len(index.search("billing", k=3)) == 3
        assert len(index.search("billing", k=10)) == 5

    def test_search_with_quotes_and_operators_does_not_raise(self, index):
        index.index_chunks(_CORPUS)
        hostile_queries = [
            '"refund',  # unterminated string
            "refund AND",  # dangling operator
            "(refund OR",  # unbalanced parenthesis
            "-refund",  # unary NOT syntax
            "title:refund",  # column filter on a nonexistent column
            'refund NEAR/2 policy"',
            "AND OR NOT",  # bare operators only
        ]

        for query in hostile_queries:
            results = index.search(query, k=5)
            assert isinstance(results, list)
            if "refund" in query:
                # operators are treated as text, the real term still matches
                assert _REFUND_HEAVY.id in _ids(results)

    def test_search_empty_or_symbol_only_query_returns_empty_list(self, index):
        index.index_chunks(_CORPUS)

        for query in ["", "   ", '"', "!!! ??? ---", '""()* -']:
            assert index.search(query, k=5) == []

    def test_search_nonpositive_k_returns_empty_list(self, index):
        # SQLite reads a negative LIMIT as "no limit"; k <= 0 must not do that
        index.index_chunks(_CORPUS)

        assert index.search("refund", k=0) == []
        assert index.search("refund", k=-1) == []

    def test_search_unmatched_term_or_empty_index_returns_empty_list(self, index):
        assert index.search("refund", k=5) == []

        index.index_chunks(_CORPUS)

        assert index.search("zebra", k=5) == []


class TestIndexChunks:
    def test_index_chunks_twice_creates_no_duplicates(self, index, conn):
        index.index_chunks(_CORPUS)
        index.index_chunks(_CORPUS)

        assert sorted(_ids(index.search("refund", k=10))) == sorted(
            [_REFUND_HEAVY.id, _REFUND_LIGHT.id]
        )
        assert conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == len(_CORPUS)


class TestRemoveDocumentVersion:
    def test_remove_document_version_excludes_its_chunks_from_results(self, index):
        index.index_chunks(
            [
                _chunk("faq", 1, 0, "refund policy for orders"),
                _chunk("faq", 1, 1, "refund window is 30 days"),
                _chunk("faq", 2, 0, "refund policy for subscriptions"),
                _REFUND_LIGHT,
            ]
        )

        index.remove_document_version("faq", 1)

        chunk_ids = _ids(index.search("refund", k=10))
        assert sorted(chunk_ids) == sorted(["faq:2:0", _REFUND_LIGHT.id])

    def test_remove_document_version_matches_doc_id_and_version_exactly(self, index):
        index.index_chunks(
            [
                _chunk("faq", 1, 0, "refund base version"),
                _chunk("faq", 11, 0, "refund version eleven"),
                _chunk("faq_v2", 1, 0, "refund underscore doc"),
                _chunk("faqXv2", 1, 0, "refund lookalike doc"),
            ]
        )

        index.remove_document_version("faq", 1)  # must not match version 11
        index.remove_document_version("faq_v2", 1)  # "_" must not act as a wildcard

        assert sorted(_ids(index.search("refund", k=10))) == ["faq:11:0", "faqXv2:1:0"]

    def test_remove_document_version_twice_is_a_no_op(self, index):
        index.index_chunks([_chunk("faq", 1, 0, "refund policy"), _REFUND_LIGHT])

        index.remove_document_version("faq", 1)
        index.remove_document_version("faq", 1)  # already gone: no error

        assert _ids(index.search("refund", k=10)) == [_REFUND_LIGHT.id]


def test_remove_does_not_conflate_colon_bearing_doc_ids(conn):
    # doc "a" v1 has chunk id "a:1:0"; doc "a:1" v2 has chunk id "a:1:2:0".
    # A prefix match on "a:1:" would delete both; removal must be exact.
    index = Bm25Index(conn)
    index.index_chunks([_chunk("a", 1, 0, "alpha refund text")])
    index.index_chunks([_chunk("a:1", 2, 0, "bravo refund text")])

    index.remove_document_version("a", 1)

    remaining = conn.execute("SELECT chunk_id FROM chunks_fts").fetchall()
    assert remaining == [("a:1:2:0",)]


def test_index_chunks_dedupes_duplicate_ids_within_one_call(conn):
    index = Bm25Index(conn)
    chunk = _chunk("dup", 1, 0, "duplicate refund text")

    index.index_chunks([chunk, chunk])

    count = conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
    assert count == 1


def test_schema_uses_no_contentless_options_for_portability(conn):
    # contentless_delete / contentless_unindexed require SQLite >= 3.47,
    # which the CI runner's bundled SQLite (3.45.x) does not have. A plain
    # FTS5 table keeps the index working on any SQLite with FTS5 (>= 3.9).
    Bm25Index(conn)

    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'chunks_fts'"
    ).fetchone()[0]

    assert "contentless" not in sql.lower()
    assert "content=" not in sql.lower()
