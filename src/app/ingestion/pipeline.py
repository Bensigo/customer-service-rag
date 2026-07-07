"""Ingestion pipeline: chunk -> embed -> store -> dual index -> invalidate.

Atomicity strategy: Qdrant cannot join a SQLite transaction, so true
two-phase commit across the chunk store, the FTS index, and the vector
index is impossible. Instead the pipeline orders its steps so failures
need as little undo as possible, and compensates when they do:

1. Chunking and embedding run before any write, so an Ollama failure
   (the most likely one - a network call per ingest) aborts with
   nothing to undo.
2. The writes run in order chunk store -> BM25 -> vectors. If any of
   them fails, every store is compensated by deleting the just-created
   version, and IngestError is raised. Rollback only ever touches the
   version this ingest created - the previously active version is never
   deleted on a failure path, so retrieval keeps serving it throughout.
3. Only after the new version is fully written are superseded versions
   swept away, per version in order indexes first, chunk store last, so
   a version whose chunks are gone from the chunk store is guaranteed
   gone from both indexes. The sweep revisits every version older than
   the new one, which is what heals a previously failed cleanup on the
   next ingest of the same document.
4. The cache invalidator runs exactly once, only after the writes and
   the sweep all succeeded - a failed ingest never invalidates. When the
   writes failed, the cache is still right (retrieval kept the prior
   version); when only the sweep failed, cached answers derived from the
   prior version linger until the retry invalidates - see the windows
   below.

Known windows, none of which surface stale text permanently:
- Between writing the new version and finishing the sweep, both
  versions are briefly searchable.
- If compensation itself partially fails, index entries of the failed
  version can linger. When the chunk-store undo succeeded, their ids
  hydrate to nothing (the chunk rows are gone, and
  ChunkStore.get_chunks_by_ids skips unknown ids); when the chunk-store
  undo also failed, the aborted version stays retrievable through the
  surviving index until it is cleaned up. Either way it is removed for
  good when the next ingest reuses that version number: the pipeline
  clears both indexes for a freshly allocated version before indexing
  into it, instead of relying on same-id overwrites, which would leak
  higher-seq entries when a re-chunk yields fewer chunks.
- If the sweep fails, the superseded version stays searchable - and any
  answers cached from it stay unspoiled in the cache - until the next
  ingest of the document retries the sweep and then invalidates; the
  ingest is reported as failed so callers know to retry.

Error messages carry document ids and version numbers only - never
chunk text or titles, which may contain customer content.
"""

from dataclasses import dataclass
from typing import Protocol

from app.ingestion.chunker import chunk_text
from app.models import Chunk, ChunkDraft, DocumentVersion


class IngestError(Exception):
    """An ingest failed. Messages carry ids and versions, never document text."""


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of a successful ingest: the now-active document version."""

    doc_id: str
    version: int
    chunk_count: int


class CacheInvalidator(Protocol):
    """Drops cached answers derived from a document after it changes."""

    def invalidate_document(self, doc_id: str) -> None: ...


class NoopCacheInvalidator:
    """Stand-in until the Redis-backed invalidator (#20) lands."""

    def invalidate_document(self, doc_id: str) -> None:
        """Nothing is cached yet, so there is nothing to invalidate."""


class SupportsChunkStorage(Protocol):
    """The slice of ChunkStore (#6) the pipeline depends on."""

    def upsert_document(
        self, doc_id: str, title: str, drafts: list[ChunkDraft]
    ) -> DocumentVersion: ...

    def get_chunks(self, doc_id: str, version: int) -> list[Chunk]: ...

    def delete_version(self, doc_id: str, version: int) -> None: ...


class SupportsKeywordIndexing(Protocol):
    """The slice of Bm25Index (#7) the pipeline depends on."""

    def index_chunks(self, chunks: list[Chunk]) -> None: ...

    def remove_document_version(self, doc_id: str, version: int) -> None: ...


class SupportsPassageEmbedding(Protocol):
    """The slice of Embedder (#9) the pipeline depends on."""

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...


class SupportsVectorIndexing(Protocol):
    """The slice of VectorStore (#10) the pipeline depends on."""

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def delete_document_version(self, doc_id: str, version: int) -> None: ...


class IngestionService:
    """Orchestrates one document ingest across the chunk store and both indexes.

    Pure orchestration over injected collaborators: the embedder is the
    app-lifetime instance owned by the caller (never created or closed
    here), and the BM25 index is expected to sit on its own sqlite3
    connection to the same WAL database as the chunk store.

    Ingests of the same document must not run concurrently (the app runs
    one ingestion worker; SQLite's single-writer locking backs this up):
    the compensation and sweep guarantees reason about one in-flight
    version per document at a time.

    ``doc_id`` is expected to already match the API's ``^[a-z0-9-]{1,64}$``
    precondition (#12 validates it before calling); the service does not
    re-validate. It stays correct for any colon-free doc id (deletes are
    keyed on (doc_id, version) equality), but callers outside the API
    should enforce the same charset.
    """

    def __init__(
        self,
        chunk_store: SupportsChunkStorage,
        bm25_index: SupportsKeywordIndexing,
        embedder: SupportsPassageEmbedding,
        vector_store: SupportsVectorIndexing,
        invalidator: CacheInvalidator,
    ) -> None:
        self._chunk_store = chunk_store
        self._bm25_index = bm25_index
        self._embedder = embedder
        self._vector_store = vector_store
        self._invalidator = invalidator

    def ingest(self, doc_id: str, title: str, text: str) -> IngestResult:
        """Chunk, embed, store, and index one document as its next version.

        On success the new version is the only searchable one and the
        cache invalidator has run once. On IngestError nothing of the
        new version remains (or, if only the supersede sweep failed, the
        new version is live and the next ingest finishes the sweep).
        """
        drafts = chunk_text(text)
        if not drafts:
            raise IngestError(
                f"document {doc_id!r} contains no chunkable text; nothing was written"
            )
        try:
            vectors = self._embedder.embed_passages([draft.text for draft in drafts])
        except Exception as error:
            raise IngestError(
                f"embedding document {doc_id!r} failed; nothing was written"
            ) from error
        stored = self._write_new_version(doc_id, title, drafts, vectors)
        self._sweep_superseded_versions(doc_id, stored.version)
        try:
            self._invalidator.invalidate_document(doc_id)
        except Exception as error:
            raise IngestError(
                f"document {doc_id!r} version {stored.version} is ingested and live, but"
                " cache invalidation failed; cached answers linger until they expire"
            ) from error
        return IngestResult(doc_id=doc_id, version=stored.version, chunk_count=stored.chunk_count)

    def _write_new_version(
        self,
        doc_id: str,
        title: str,
        drafts: list[ChunkDraft],
        vectors: list[list[float]],
    ) -> DocumentVersion:
        """Write the new version everywhere, compensating on any failure."""
        try:
            stored = self._chunk_store.upsert_document(doc_id, title, drafts)
        except Exception as error:
            # upsert_document is a single SQLite transaction: nothing to undo
            raise IngestError(f"storing document {doc_id!r} failed; nothing was written") from error
        try:
            # seq order from the store matches draft order, which is the
            # order the vectors were embedded in
            chunks = self._chunk_store.get_chunks(doc_id, stored.version)
            # a compensated earlier attempt may have reused this version
            # number and left stale index entries behind - clear both
            # indexes before writing (same-id overwrites would leak
            # higher-seq leftovers when a re-chunk yields fewer chunks)
            self._bm25_index.remove_document_version(doc_id, stored.version)
            self._vector_store.delete_document_version(doc_id, stored.version)
            self._bm25_index.index_chunks(chunks)
            self._vector_store.upsert(chunks, vectors)
        except Exception as error:
            raise self._rollback(doc_id, stored.version) from error
        return stored

    def _rollback(self, doc_id: str, version: int) -> IngestError:
        """Best-effort compensation: delete the just-written version everywhere.

        Every store is attempted even when an earlier one fails, to keep
        the indexes as close to the chunk store as possible; failures are
        reported (by store and error type) in the returned IngestError.
        """
        failures: list[str] = []
        # flakiest store first (vectors reach Qdrant over the network), chunk
        # store last so its chunk-row absence still proves a fully-gone version
        undo_steps = (
            ("vector store", self._vector_store.delete_document_version),
            ("bm25 index", self._bm25_index.remove_document_version),
            ("chunk store", self._chunk_store.delete_version),
        )
        for store_name, undo in undo_steps:
            try:
                undo(doc_id, version)
            except Exception as undo_error:
                failures.append(f"{store_name}: {type(undo_error).__name__}")
        if failures:
            return IngestError(
                f"ingesting document {doc_id!r} version {version} failed and rollback was"
                f" incomplete ({'; '.join(failures)}); retrying the ingest clears the leftovers"
            )
        return IngestError(
            f"ingesting document {doc_id!r} version {version} failed;"
            " all partial writes were rolled back"
        )

    def _sweep_superseded_versions(self, doc_id: str, new_version: int) -> None:
        """Delete every older version still present, from indexes and store.

        Per version the order is indexes first, chunk store last, so
        chunks absent from the chunk store prove the version is fully
        gone. The sweep checks every version older than the new one
        (cheap indexed SQLite lookups), which retries any cleanup a
        previous ingest failed to finish.
        """
        for version in range(new_version - 1, 0, -1):
            if not self._chunk_store.get_chunks(doc_id, version):
                continue
            try:
                # flakiest store first, chunk store last: a mid-sweep failure
                # leaves the version whole in the survivors rather than
                # half-removed, and chunk-row absence still proves it fully gone
                self._vector_store.delete_document_version(doc_id, version)
                self._bm25_index.remove_document_version(doc_id, version)
                self._chunk_store.delete_version(doc_id, version)
            except Exception as error:
                raise IngestError(
                    f"document {doc_id!r} version {new_version} is ingested and live, but"
                    f" cleanup of superseded version {version} failed; the next ingest of"
                    " this document retries the cleanup"
                ) from error
