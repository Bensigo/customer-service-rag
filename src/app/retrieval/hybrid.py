"""Hybrid retrieval: BM25 + vector search fused with Reciprocal Rank Fusion.

Fusion is rank-based: each index contributes ``1 / (RRF_K + rank)`` for
every chunk it returns, where rank is **1-based** (the best hit of each
list has rank 1) and ``RRF_K = 60``, the constant from the original RRF
paper (Cormack, Clarke & Buettcher, SIGIR 2009). The constant damps the
gap between adjacent ranks so a single top position cannot dominate the
sum. Fusing on ranks rather than raw scores sidesteps the incomparable
scales of BM25 (positive, unbounded, corpus-relative) and cosine
similarity (bounded by [-1, 1]).

Contributions for a chunk found by both indexes are summed, so
agreement between the indexes beats any single first place: rank 1 in
one list scores 1/61, while rank 1 plus rank 2 across both scores
1/61 + 1/62. Ties in the fused score break deterministically toward the
BM25 list order (stable sort over insertion order).

Hydration happens before truncation: every fused chunk id is looked up
in the chunk store, and ids the store no longer knows (skew while an
ingest is replacing a document version) are skipped with a warning that
names the chunk id only - never chunk text, which may hold customer
PII - so a stale index entry cannot shrink the returned top_n.
"""

import logging
from typing import Protocol

from app.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

RRF_K = 60


class SupportsTextSearch(Protocol):
    """Bm25Index seam: ranked (chunk_id, score), best first."""

    def search(self, query: str, k: int) -> list[tuple[str, float]]: ...


class SupportsQueryEmbedding(Protocol):
    """Embedder seam: query-mode embedding (model-specific query prompt)."""

    def embed_query(self, text: str) -> list[float]: ...


class SupportsVectorSearch(Protocol):
    """VectorStore seam: ranked (chunk_id, score), best first."""

    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]: ...


class SupportsChunkHydration(Protocol):
    """ChunkStore seam: hydrates ids to chunks, preserving input order."""

    def get_chunks_by_ids(self, ids: list[str]) -> list[Chunk]: ...


class HybridRetriever:
    """Fuses BM25 and vector search results into one ranked chunk list."""

    def __init__(
        self,
        bm25_index: SupportsTextSearch,
        embedder: SupportsQueryEmbedding,
        vector_store: SupportsVectorSearch,
        chunk_store: SupportsChunkHydration,
    ) -> None:
        self._bm25_index = bm25_index
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunk_store = chunk_store

    def retrieve(self, query: str, *, k_each: int = 20, top_n: int = 12) -> list[RetrievedChunk]:
        """Top ``top_n`` chunks for the query, RRF-fused across both indexes.

        Takes ``k_each`` candidates from each index. The query is
        embedded with ``embed_query`` so the model's query prompt is
        applied - never embedded raw. A blank query (empty or
        whitespace-only) returns [] without touching any backend, as
        does a non-positive ``top_n``.
        """
        if top_n <= 0 or not query.strip():
            return []
        bm25_hits = self._bm25_index.search(query, k_each)
        vector_hits = self._vector_store.search(self._embedder.embed_query(query), k_each)

        fused_scores: dict[str, float] = {}
        sources: dict[str, set[str]] = {}
        for source, hits in (("bm25", bm25_hits), ("vector", vector_hits)):
            for rank, (chunk_id, _) in enumerate(hits, start=1):
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1 / (RRF_K + rank)
                sources.setdefault(chunk_id, set()).add(source)

        ranked_ids = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)
        chunks = self._chunk_store.get_chunks_by_ids(ranked_ids)
        for chunk_id in set(ranked_ids) - {chunk.id for chunk in chunks}:
            logger.warning(
                "chunk %s is in a search index but missing from the chunk store; skipping it",
                chunk_id,
            )
        return [
            RetrievedChunk(chunk=chunk, score=fused_scores[chunk.id], sources=sources[chunk.id])
            for chunk in chunks[:top_n]
        ]
