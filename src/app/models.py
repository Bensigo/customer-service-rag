from dataclasses import dataclass


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
class DocumentVersion:
    """One stored version of a document, as returned by the chunk store
    after an upsert."""

    doc_id: str
    version: int
    chunk_count: int
