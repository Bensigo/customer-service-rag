"""A small pool of retrievers, one per read connection (issue #18).

The chat endpoint offloads ``retrieve`` onto the event-loop threadpool,
so several retrievals can run on different threads at once. A single
SQLite connection must not be touched by two threads concurrently — and
it must not be the ingest path's connection either (#12 opens those with
``check_same_thread=False`` and serializes writes through a
single-permit limiter; concurrent chat reads live outside that limiter).

``RetrieverPool`` solves both: it lends out ``HybridRetriever`` instances
each bound to their OWN connections (built lazily by ``factory``, up to
``size``), and guarantees an instance is checked out to exactly one
thread at a time. WAL mode lets these independent read connections run
concurrently and stay isolated from the writer. ``retrieve`` blocks when
all instances are busy, capping concurrent SQLite readers at ``size``.

The pool exposes ``retrieve`` with the same signature as
``HybridRetriever``, so the chat route uses it as a drop-in retriever.
"""

import logging
import threading
from collections.abc import Callable

from app.models import RetrievedChunk

logger = logging.getLogger(__name__)


class _Retriever:
    """Structural type: what the pool needs from a HybridRetriever."""

    def retrieve(
        self, query: str, *, k_each: int = 20, top_n: int = 12
    ) -> list[RetrievedChunk]: ...


# factory() builds a fresh retriever plus its own connection-closing callback.
RetrieverFactory = Callable[[], tuple[_Retriever, Callable[[], None]]]


class RetrieverPool:
    """Thread-safe pool of independent retrievers, one per read connection."""

    def __init__(self, factory: RetrieverFactory, *, size: int = 4) -> None:
        if size < 1:
            raise ValueError("pool size must be >= 1")
        self._factory = factory
        self._size = size
        # A bounded semaphore caps outstanding checkouts at `size`, so at
        # most `size` retrievers (and connections) ever exist concurrently.
        self._slots = threading.BoundedSemaphore(size)
        self._lock = threading.Lock()
        self._idle: list[tuple[_Retriever, Callable[[], None]]] = []
        self._all: list[tuple[_Retriever, Callable[[], None]]] = []
        self._closed = False

    def _acquire(self) -> tuple[_Retriever, Callable[[], None]]:
        self._slots.acquire()
        with self._lock:
            if self._closed:
                self._slots.release()
                raise RuntimeError("retriever pool is closed")
            if self._idle:
                return self._idle.pop()
        # No idle instance but a slot is free: build a new one (its own conns).
        # If the factory fails (e.g. the DB can't be opened), release the slot
        # so a build failure can't permanently shrink the pool toward deadlock.
        try:
            entry = self._factory()
        except Exception:
            self._slots.release()
            raise
        with self._lock:
            self._all.append(entry)
        return entry

    def _release(self, entry: tuple[_Retriever, Callable[[], None]]) -> None:
        with self._lock:
            if not self._closed:
                self._idle.append(entry)
        self._slots.release()

    def retrieve(self, query: str, *, k_each: int = 20, top_n: int = 12) -> list[RetrievedChunk]:
        """Borrow a retriever, run the query, and return it to the pool.

        Blocks until a retriever is free when all ``size`` are busy.
        """
        retriever, _ = entry = self._acquire()
        try:
            return retriever.retrieve(query, k_each=k_each, top_n=top_n)
        finally:
            self._release(entry)

    def close(self) -> None:
        """Close every retriever's connections. Idempotent; further
        ``retrieve`` calls raise. Each close is attempted even if one
        fails, so a single failure cannot leak the rest."""
        with self._lock:
            self._closed = True
            entries = list(self._all)
            self._all.clear()
            self._idle.clear()
        for _, close in entries:
            try:
                close()
            except Exception:
                logger.exception("failed to close a pooled retriever connection")
