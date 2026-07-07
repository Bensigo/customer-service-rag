from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Title and plain text pulled from one uploaded file by the
    extractors, before chunking. PDF page breaks survive as form-feed
    characters in ``text``."""

    title: str
    text: str


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """A chunk of document text produced by the chunker, before storage
    assigns it a document id and version."""

    seq: int
    text: str
    token_estimate: int


@dataclass(frozen=True, slots=True)
class Chunk:
    """A stored chunk, bound to one version of a document. Its id
    ("{doc_id}:{version}:{seq}") is the key shared with the search
    indexes, so retrieval hits can be hydrated back to text."""

    id: str
    doc_id: str
    version: int
    seq: int
    text: str
    title: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk surfaced by retrieval: the hydrated chunk, its fused
    relevance score (RRF - comparable only within one result list), and
    which indexes returned it (``sources`` is a subset of
    {"bm25", "vector"}). The shared retrieval-facing type consumed by
    reranking, context assembly, and chat."""

    chunk: Chunk
    score: float
    sources: set[str]


@dataclass(frozen=True, slots=True)
class DocumentVersion:
    """One stored version of a document, as returned by the chunk store
    after an upsert."""

    doc_id: str
    version: int
    chunk_count: int
