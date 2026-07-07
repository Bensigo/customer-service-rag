"""Specs for the first-turn cache fingerprint (issue #19).

The fingerprint keys the response cache. Two properties matter:

- Normalization: superficial differences in a question (case, spacing,
  a trailing "?") must collapse to the same key, so "Reset my password?"
  and "reset  my password" share a cache entry.
- First-turn-only: a non-empty history yields ``None`` (uncacheable) by
  deliberate policy — a fingerprint including a conversation digest would
  almost never repeat, so caching later turns would be decorative. The
  cache targets first-turn FAQs, where support traffic concentrates.

Pure and offline: no Redis, no network — runs everywhere, unmarked.
"""

from app.chat.fingerprint import fingerprint
from app.models import Turn


def test_normalization_collapses_case_whitespace_and_trailing_punctuation():
    # Different case, internal spacing, and a trailing "?" all normalize
    # to the same key: one cache entry serves both phrasings.
    a = fingerprint("Reset my password?", [])
    b = fingerprint("reset  my password", [])
    assert a is not None
    assert a == b


def test_surrounding_whitespace_is_stripped():
    assert fingerprint("  reset my password  ", []) == fingerprint("reset my password", [])


def test_distinct_questions_have_distinct_fingerprints():
    assert fingerprint("reset my password", []) != fingerprint("cancel my subscription", [])


def test_is_deterministic_across_calls():
    assert fingerprint("how do I reset my password", []) == fingerprint(
        "how do I reset my password", []
    )


def test_returns_sha256_hex_digest():
    fp = fingerprint("reset my password", [])
    assert fp is not None
    assert len(fp) == 64
    assert all(c in "0123456789abcdef" for c in fp)


def test_returns_none_when_history_is_nonempty():
    # Deliberate policy: only first-turn messages are cacheable.
    history = [Turn(role="user", content="earlier"), Turn(role="assistant", content="reply")]
    assert fingerprint("reset my password", history) is None


def test_returns_fingerprint_when_history_is_empty():
    assert fingerprint("reset my password", []) is not None
