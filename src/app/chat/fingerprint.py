"""First-turn cache fingerprint (issue #19).

``fingerprint`` maps a chat message to the stable key under which its
answer is cached. Two design decisions are baked in:

**Normalization.** The message is casefolded, its internal whitespace
collapsed to single spaces, and surrounding whitespace plus trailing
punctuation stripped, before hashing. So "Reset my password?" and
"reset  my password" produce the same key and share one cache entry —
support questions differ far more in surface form than in intent.

**First-turn only (deliberate policy).** ``fingerprint`` returns ``None``
(uncacheable) whenever ``history`` is non-empty. A fingerprint that
folded in a conversation digest would almost never repeat across users,
making the cache decorative. This is an explicit *first-turn* cache:
first-turn FAQs ("how do I reset my password") dominate support traffic,
so caching only the first turn captures nearly all the reuse while
keeping keys shared across users. Later turns run the full pipeline.
See the decision log (#22); hit-rate becomes measurable via #21.

Pure and dependency-free: no Redis, no network.
"""

import hashlib
import re

from app.models import Turn

# Runs of whitespace (spaces, tabs, newlines) collapse to a single space.
_WHITESPACE = re.compile(r"\s+")
# Trailing punctuation that carries no intent for an FAQ lookup, so
# "reset my password?" and "reset my password." key the same entry.
_TRAILING_PUNCTUATION = ".?!,;:"


def _normalize(message: str) -> str:
    collapsed = _WHITESPACE.sub(" ", message).strip()
    return collapsed.casefold().rstrip(_TRAILING_PUNCTUATION)


def fingerprint(message: str, history: list[Turn]) -> str | None:
    """The cache key for ``message``, or ``None`` if uncacheable.

    Returns ``None`` when ``history`` is non-empty (first-turn-only
    policy). Otherwise returns the sha256 hex digest of the normalized
    message — deterministic and stable across processes.
    """
    if history:
        return None
    normalized = _normalize(message)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
