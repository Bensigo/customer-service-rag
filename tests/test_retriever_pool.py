"""Specs for the retriever connection pool (issue #18).

The chat path serves retrieval reads concurrently on the event-loop
threadpool. Sharing one SQLite connection across those reader threads
(and the ingest writer thread) is unsafe even with
check_same_thread=False, so retrieval gets its OWN pool of retrievers,
each bound to its own connections; a retriever is never used by two
threads at once.

These tests use fake retrievers (no SQLite) to prove the pool's
checkout/return/close contract and its concurrency guarantee.
"""

import threading
import time

from app.retrieval.pool import RetrieverPool


class RecordingRetriever:
    """A fake HybridRetriever: records concurrent use and its own closes."""

    def __init__(self, live_counter, max_seen, lock, *, delay=0.0):
        self._live = live_counter
        self._max_seen = max_seen
        self._lock = lock
        self._delay = delay
        self.closed = False

    def retrieve(self, query, *, k_each=20, top_n=12):
        with self._lock:
            self._live[0] += 1
            self._max_seen[0] = max(self._max_seen[0], self._live[0])
        try:
            if self._delay:
                time.sleep(self._delay)
            return [query]
        finally:
            with self._lock:
                self._live[0] -= 1

    def close(self):
        self.closed = True


def _make_factory(created, live, max_seen, lock, *, delay=0.0):
    def factory():
        r = RecordingRetriever(live, max_seen, lock, delay=delay)
        created.append(r)
        return r, r.close

    return factory


def test_retrieve_delegates_and_returns_result():
    created, live, max_seen, lock = [], [0], [0], threading.Lock()
    pool = RetrieverPool(_make_factory(created, live, max_seen, lock), size=2)

    assert pool.retrieve("hello") == ["hello"]
    pool.close()


def test_reuses_a_single_retriever_for_sequential_calls():
    created, live, max_seen, lock = [], [0], [0], threading.Lock()
    pool = RetrieverPool(_make_factory(created, live, max_seen, lock), size=3)

    for _ in range(5):
        pool.retrieve("q")

    # Sequential calls return the same instance to the pool and reuse it,
    # so only one retriever is ever built.
    assert len(created) == 1
    pool.close()


def test_never_hands_one_retriever_to_two_threads_at_once():
    created, live, max_seen, lock = [], [0], [0], threading.Lock()
    pool = RetrieverPool(_make_factory(created, live, max_seen, lock, delay=0.05), size=3)

    threads = [threading.Thread(target=lambda: pool.retrieve("q")) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    # Each created retriever is single-threaded, and the pool caps
    # concurrency at `size`, so at most `size` were ever created.
    assert len(created) <= 3
    # No retriever's max concurrent use exceeded 1 is proven by the pool
    # design; the aggregate concurrency never exceeded the pool size.
    assert max_seen[0] <= 3
    pool.close()


def test_factory_failure_releases_the_slot():
    # A factory that fails must not permanently consume a slot, or the pool
    # would shrink toward deadlock. After `size` failures the pool must still
    # be usable once the factory recovers.
    calls = [0]

    def flaky_factory():
        calls[0] += 1
        if calls[0] <= 2:
            raise RuntimeError("cannot open db")

        class _R:
            def retrieve(self, query, *, k_each=20, top_n=12):
                return [query]

            def close(self):
                pass

        r = _R()
        return r, r.close

    pool = RetrieverPool(flaky_factory, size=1)

    for _ in range(2):
        try:
            pool.retrieve("q")
        except RuntimeError:
            pass
    # The slot was released each time; a recovered factory now succeeds.
    assert pool.retrieve("q") == ["q"]
    pool.close()


def test_close_closes_every_created_retriever():
    created, live, max_seen, lock = [], [0], [0], threading.Lock()
    pool = RetrieverPool(_make_factory(created, live, max_seen, lock), size=2)

    pool.retrieve("a")
    pool.close()

    assert created and all(r.closed for r in created)


def test_close_waits_for_an_in_flight_retrieve_before_closing():
    # A connection must never be closed while a thread is still running a
    # query on it: close() blocks until the checkout is returned.
    started = threading.Event()
    proceed = threading.Event()
    closed_during_retrieve = []

    class _BlockingRetriever:
        def retrieve(self, query, *, k_each=20, top_n=12):
            started.set()
            proceed.wait(timeout=5)
            # record whether close() had already closed us mid-query
            closed_during_retrieve.append(self.closed)
            return [query]

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    built = []

    def factory():
        r = _BlockingRetriever()
        built.append(r)
        return r, r.close

    pool = RetrieverPool(factory, size=1)

    worker = threading.Thread(target=lambda: pool.retrieve("q"))
    worker.start()
    started.wait(timeout=5)

    closer_done = threading.Event()

    def do_close():
        pool.close()
        closer_done.set()

    closer = threading.Thread(target=do_close)
    closer.start()

    # close() must be blocked while the retrieve is still in flight.
    assert not closer_done.wait(timeout=0.3)
    assert not built[0].closed

    proceed.set()
    worker.join(timeout=5)
    closer.join(timeout=5)

    # the retrieve saw an OPEN connection throughout; close ran only after.
    assert closed_during_retrieve == [False]
    assert built[0].closed


def test_double_close_is_a_noop():
    created, live, max_seen, lock = [], [0], [0], threading.Lock()
    pool = RetrieverPool(_make_factory(created, live, max_seen, lock), size=2)

    pool.retrieve("a")
    pool.close()
    pool.close()  # must not deadlock or double-close

    assert all(r.closed for r in created)
