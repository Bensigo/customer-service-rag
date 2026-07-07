"""Sentence-aware chunking with overlapping windows.

Pure functions, stdlib only. Sentence splitting is regex-based with a
small abbreviation list — good enough for support documents, and the
documented swap point if a model-based splitter is ever needed.

Known splitter trade-off: an abbreviation from the list at a *genuine*
sentence end (e.g. "Ask for Dr. The clinic...") merges the two sentences.
A word list cannot disambiguate that case, so the list is biased small —
words that commonly end sentences ("no", "etc") are excluded, because a
false merge grows a sentence without bound (and oversized sentences are
never split), while a false split merely produces two short chunks.

Token counts are a conservative heuristic (``ceil(words * 1.4)``) rather
than a real tokenizer, so this module stays dependency-free; the default
``max_tokens=350`` keeps worst-case chunks under the embedding model's
512-token window (#9 adds a cross-check test against the real tokenizer).
"""

import math
import re

from app.models import ChunkDraft

# Sentence boundary: terminal punctuation, optional closing quotes or
# brackets (kept with the sentence), then whitespace.
_BOUNDARY = re.compile(r"([.!?][\"')\]]*)\s+")

_TRAILING_PUNCTUATION = ".!?\"')]"

# Trailing-word abbreviations that must not end a sentence. Compared
# lowercase, with trailing punctuation stripped. Biased small — see the
# module docstring for why sentence-final words don't belong here.
_ABBREVIATIONS = {
    "dr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "e.g",
    "i.e",
    "a.m",
    "p.m",
    "inc",
    "ltd",
    "fig",
}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences; newlines always end a sentence."""
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            sentences.extend(_split_line(line))
    return sentences


def estimate_tokens(text: str) -> int:
    """Conservative token estimate: ceil(words * 1.4)."""
    return math.ceil(len(text.split()) * 1.4)


def chunk_text(text: str, *, max_tokens: int = 350, overlap_sentences: int = 2) -> list[ChunkDraft]:
    """Pack sentences into overlapping windows bounded by max_tokens.

    Each window starts with the last ``overlap_sentences`` sentences of
    the previous one. A single sentence over the budget becomes its own
    chunk — sentences are never split or dropped.
    """
    sentences = split_sentences(text)
    drafts: list[ChunkDraft] = []
    start = 0
    while start < len(sentences):
        tokens = 0
        end = start
        while end < len(sentences):
            sentence_tokens = estimate_tokens(sentences[end])
            if end > start and tokens + sentence_tokens > max_tokens:
                break
            tokens += sentence_tokens
            end += 1
        chunk = " ".join(sentences[start:end])
        drafts.append(
            ChunkDraft(seq=len(drafts), text=chunk, token_estimate=estimate_tokens(chunk))
        )
        if end >= len(sentences):
            break
        # overlap backward, but always advance to guarantee termination
        start = max(end - overlap_sentences, start + 1)
    return drafts


def _split_line(line: str) -> list[str]:
    parts: list[str] = []
    start = 0
    prev_end = 0
    for match in _BOUNDARY.finditer(line):
        sentence_end = match.end(1)
        # examine only the text since the previous boundary, so a long run
        # of abbreviation hits stays O(n) instead of re-scanning the whole
        # accumulated candidate each time
        segment = line[prev_end:sentence_end]
        prev_end = match.end()
        if _ends_with_abbreviation(segment):
            continue
        parts.append(line[start:sentence_end].strip())
        start = match.end()
    tail = line[start:].strip()
    if tail:
        parts.append(tail)
    return [part for part in parts if part]


def _ends_with_abbreviation(segment: str) -> bool:
    words = segment.rsplit(None, 1)
    if not words:
        return True  # nothing before the boundary — never a real sentence end
    last = words[-1].rstrip(_TRAILING_PUNCTUATION).lower()
    return last in _ABBREVIATIONS
