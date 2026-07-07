"""Qdrant-backed vector index of chunk embeddings.

Point ids are UUID5 hashes of the chunk id (Qdrant accepts only unsigned
ints or UUIDs as point ids), so re-ingesting a document version
overwrites its points in place instead of appending duplicates.

The collection's vector size is a constructor parameter: the ingestion
pipeline (#11) passes the embedder's probe-discovered dimension (#9),
keeping this store free of any embedding-model knowledge.
ensure_collection is idempotent but never recreates - if the existing
collection was built for a different dimension it raises, because a
silent recreate would drop every indexed vector.

``url`` is handed to qdrant-client as its ``location``, so it accepts a
server URL (e.g. http://localhost:6333) or ":memory:" for the in-process
local backend used by the unit tests.
"""

import uuid

from qdrant_client import QdrantClient, models

from app.models import Chunk

_POINT_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "customer-service-rag.vector-store")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


class VectorStore:
    """Chunk-embedding index on one Qdrant collection, keyed by chunk id."""

    def __init__(self, url: str, vector_size: int, collection: str = "chunks") -> None:
        self._client = QdrantClient(location=url)
        self._vector_size = vector_size
        self._collection = collection

    def close(self) -> None:
        self._client.close()

    def ensure_collection(self) -> None:
        """Create the collection if missing (cosine distance); repeat calls are no-ops.

        A collection that exists with a different vector size fails loudly:
        recreating it here would silently drop everything already indexed.
        Re-embedding into a fresh collection is a deliberate operation.
        """
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._vector_size, distance=models.Distance.COSINE
                ),
            )
            return
        existing_size = self._client.get_collection(self._collection).config.params.vectors.size
        if existing_size != self._vector_size:
            raise ValueError(
                f"Qdrant collection {self._collection!r} holds {existing_size}-dimensional"
                f" vectors but this store was configured for {self._vector_size} dimensions;"
                " refusing to recreate it (that would drop all indexed vectors) -"
                " re-embed into a fresh collection instead"
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Index one vector per chunk; a repeated chunk id overwrites its point.

        Payload carries {chunk_id, doc_id, version, title} so hits map back
        to chunk-store ids and versions can be deleted by filter.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return  # Qdrant rejects an upsert of zero points
        points = [
            models.PointStruct(
                id=_point_id(chunk.id),
                vector=vector,
                payload={
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "version": chunk.version,
                    "title": chunk.title,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)

    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        """Top-k (chunk_id, score) nearest the vector; cosine similarity, higher is better.

        k <= 0 returns [] (mirrors Bm25Index.search, so the fusion layer
        (#13) can treat both indexes uniformly).
        """
        if k <= 0:
            return []
        response = self._client.query_points(
            collection_name=self._collection, query=vector, limit=k, with_payload=True
        )
        return [(point.payload["chunk_id"], point.score) for point in response.points]

    def delete_document_version(self, doc_id: str, version: int) -> None:
        """Drop every point of (doc_id, version); missing ones are a no-op.

        Filters on the stored payload fields - never by parsing the
        composite chunk id, whose doc ids may themselves contain ":".
        """
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                        models.FieldCondition(
                            key="version", match=models.MatchValue(value=version)
                        ),
                    ]
                )
            ),
            wait=True,
        )
