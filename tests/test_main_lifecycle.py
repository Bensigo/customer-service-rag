"""Specs for create_app()'s shutdown resource cleanup (issue #12).

The lifespan builds several stateful resources (SQLite connections, a
pooled HTTP embedder client, a Qdrant client). Shutdown must attempt to
close every one even if an earlier close raises, so a single failing
close cannot leak the others — asserted here without any live services.
"""

from app.main import _close_all


def test_close_all_runs_every_closer_even_when_one_raises():
    calls = []

    def ok(name):
        return lambda: calls.append(name)

    def boom():
        calls.append("boom")
        raise RuntimeError("close failed")

    closers = [
        ("first", ok("first")),
        ("second", boom),
        ("third", ok("third")),
    ]

    # must not propagate; every closer must have been attempted, in order
    _close_all(closers)

    assert calls == ["first", "boom", "third"]
