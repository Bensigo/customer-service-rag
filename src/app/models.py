from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """A chunk of document text produced by the chunker, before storage
    assigns it a document id and version."""

    seq: int
    text: str
    token_estimate: int
