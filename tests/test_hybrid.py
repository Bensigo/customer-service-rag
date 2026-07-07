"""Specs for hybrid BM25 + vector retrieval with RRF fusion (issue #13).

All fusion behavior is exercised against in-memory fakes with
hand-planted rankings, so the RRF math is deterministic and exactly
assertable and CI needs neither Ollama nor Qdrant.

Planted fixture: BM25 returns [A, B], the vector index returns [B, C].
With 1-based ranks and RRF constant 60 the expected fused scores are
    A: 1/61          (bm25 rank 1)
    B: 1/61 + 1/62   (vector rank 1 + bm25 rank 2)
    C: 1/62          (vector rank 2)
so the expected order is B, A, C - the chunk both indexes agree on wins.
"""

import logging

import pytest
from app.retrieval.hybrid import HybridRetriever

from app.models import Chunk, RetrievedChunk

QUERY = "how do I reset my password"
QUERY_VECTOR = [0.25, -0.5, 0.75]

CHUNK_A = "faq:1:0"
CHUNK_B = "faq:1:1"
CHUNK_C = "faq:1:2"

BM25_HITS = [(CHUNK_A, 9.0), (CHUNK_B, 5.0)]
VECTOR_HITS = [(CHUNK_B, 0.95), (CHUNK_C, 0.80)]

RRF_A = 1 / 61
RRF_B = 1 / 61 + 1 / 62
RRF_C = 1 / 62


class FakeBm25Index:
    def __init__(self, hits: list[tuple[str, float]]):
        self._hits = hits
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        self.calls.append((query, k))
        return self._hits[:k]


class FakeEmbedder:
    """Records which embedding mode was used; retrieval must embed the
    query with embed_query (model-specific query prompt), never
    embed_passages."""

    def __init__(self):
        self.embed_query_calls: list[str] = []
        self.embed_passages_calls: list[list[str]] = []

    def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls.append(text)
        return list(QUERY_VECTOR)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.embed_passages_calls.append(texts)
        return [[0.0] * len(QUERY_VECTOR) for _ in texts]


class FakeVectorStore:
    def __init__(self, hits: list[tuple[str, float]]):
        self._hits = hits
        self.calls: list[tuple[list[float], int]] = []

    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        self.calls.append((vector, k))
        return self._hits[:k]


class FakeChunkStore:
    def __init__(self, chunks: list[Chunk]):
        self._by_id = {chunk.id: chunk for chunk in chunks}

    def get_chunks_by_ids(self, ids: list[str]) -> list[Chunk]:
        return [self._by_id[chunk_id] for chunk_id in ids if chunk_id in self._by_id]


def _chunk(chunk_id: str, text: str) -> Chunk:
    doc_id, version, seq = chunk_id.rsplit(":", 2)
    return Chunk(
        id=chunk_id, doc_id=doc_id, version=int(version), seq=int(seq), text=text, title=doc_id
    )


CHUNKS = [
    _chunk(CHUNK_A, "reset your password from the account page"),
    _chunk(CHUNK_B, "password reset links expire after one hour"),
    _chunk(CHUNK_C, "contact support if the reset email never arrives"),
]


def _make_retriever(
    bm25_hits: list[tuple[str, float]] = BM25_HITS,
    vector_hits: list[tuple[str, float]] = VECTOR_HITS,
    chunks: list[Chunk] = CHUNKS,
) -> tuple[HybridRetriever, FakeBm25Index, FakeEmbedder, FakeVectorStore]:
    bm25 = FakeBm25Index(bm25_hits)
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore(vector_hits)
    retriever = HybridRetriever(bm25, embedder, vector_store, FakeChunkStore(chunks))
    return retriever, bm25, embedder, vector_store


def _ids(results: list[RetrievedChunk]) -> list[str]:
    return [result.chunk.id for result in results]


class TestRrfFusion:
    def test_rrf_chunk_in_both_lists_outranks_single_list_chunks(self):
        retriever, *_ = _make_retriever()

        results = retriever.retrieve(QUERY)

        assert _ids(results) == [CHUNK_B, CHUNK_A, CHUNK_C]

    def test_rrf_scores_match_hand_computed_values(self):
        retriever, *_ = _make_retriever()

        results = retriever.retrieve(QUERY)

        scores = {result.chunk.id: result.score for result in results}
        assert scores[CHUNK_A] == pytest.approx(RRF_A)
        assert scores[CHUNK_B] == pytest.approx(RRF_B)
        assert scores[CHUNK_C] == pytest.approx(RRF_C)

    def test_dedupe_merges_sources_for_shared_chunk(self):
        retriever, *_ = _make_retriever()

        results = retriever.retrieve(QUERY)

        assert _ids(results).count(CHUNK_B) == 1  # deduped, not listed once per index
        sources = {result.chunk.id: result.sources for result in results}
        assert sources[CHUNK_B] == {"bm25", "vector"}
        assert sources[CHUNK_A] == {"bm25"}
        assert sources[CHUNK_C] == {"vector"}

    def test_results_are_hydrated_chunks(self):
        retriever, *_ = _make_retriever()

        results = retriever.retrieve(QUERY)

        assert results[0].chunk == CHUNKS[1]  # the full stored chunk, text included


class TestEmbedding:
    def test_query_uses_embed_query_not_embed_passages(self):
        retriever, _, embedder, vector_store = _make_retriever()

        retriever.retrieve(QUERY)

        assert embedder.embed_query_calls == [QUERY]
        assert embedder.embed_passages_calls == []
        # and the vector index was searched with exactly that embedding
        assert [vector for vector, _ in vector_store.calls] == [QUERY_VECTOR]

    def test_k_each_reaches_both_indexes(self):
        retriever, bm25, _, vector_store = _make_retriever()

        retriever.retrieve(QUERY, k_each=7)

        assert bm25.calls == [(QUERY, 7)]
        assert [k for _, k in vector_store.calls] == [7]


class TestEdgeCases:
    @pytest.mark.parametrize("query", ["", "   ", "\t\n "])
    def test_empty_or_whitespace_query_returns_empty_list(self, query):
        retriever, bm25, embedder, vector_store = _make_retriever()

        assert retriever.retrieve(query) == []
        # nothing was embedded or searched for a blank query
        assert bm25.calls == []
        assert embedder.embed_query_calls == []
        assert vector_store.calls == []

    def test_empty_bm25_list_still_fuses_vector_results(self):
        retriever, *_ = _make_retriever(bm25_hits=[])

        results = retriever.retrieve(QUERY)

        assert _ids(results) == [CHUNK_B, CHUNK_C]
        assert [result.score for result in results] == pytest.approx([1 / 61, 1 / 62])
        assert all(result.sources == {"vector"} for result in results)

    def test_empty_vector_list_still_fuses_bm25_results(self):
        retriever, *_ = _make_retriever(vector_hits=[])

        results = retriever.retrieve(QUERY)

        assert _ids(results) == [CHUNK_A, CHUNK_B]
        assert [result.score for result in results] == pytest.approx([1 / 61, 1 / 62])
        assert all(result.sources == {"bm25"} for result in results)

    def test_missing_chunk_in_store_is_skipped_not_raised(self, caplog):
        # ingest skew: CHUNK_B is still in both indexes (top-ranked) but
        # gone from the chunk store
        chunks_without_b = [CHUNKS[0], CHUNKS[2]]
        retriever, *_ = _make_retriever(chunks=chunks_without_b)

        with caplog.at_level(logging.WARNING, logger="app.retrieval.hybrid"):
            results = retriever.retrieve(QUERY, top_n=2)

        # skipped, and the freed slot is filled by the next-ranked chunk
        assert _ids(results) == [CHUNK_A, CHUNK_C]

        [record] = caplog.records
        assert record.levelno == logging.WARNING
        assert CHUNK_B in record.getMessage()
        # the warning names the chunk id only - never chunk text (PII)
        for chunk in CHUNKS:
            assert chunk.text not in record.getMessage()

    def test_returns_top_n_only(self):
        retriever, *_ = _make_retriever()

        results = retriever.retrieve(QUERY, top_n=2)

        assert _ids(results) == [CHUNK_B, CHUNK_A]

    def test_nonpositive_top_n_returns_empty_list(self):
        retriever, *_ = _make_retriever()

        assert retriever.retrieve(QUERY, top_n=0) == []
        assert retriever.retrieve(QUERY, top_n=-1) == []
