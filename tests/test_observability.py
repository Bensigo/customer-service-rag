"""Unit specs for the observability helpers (issue #21).

These run everywhere (no live services): the request-id validator, the
stage-timing accumulator, and the readiness aggregation are pure logic,
exercised directly without an app or backing store.
"""

import time

import pytest

from app.observability import (
    StageTimings,
    normalize_request_id,
    readiness_report,
)


class TestNormalizeRequestId:
    def test_generates_hex_when_absent(self):
        rid = normalize_request_id(None)
        assert isinstance(rid, str)
        # uuid4 hex: 32 lowercase hex chars, no dashes
        assert len(rid) == 32
        assert all(c in "0123456789abcdef" for c in rid)

    def test_generates_fresh_ids(self):
        assert normalize_request_id(None) != normalize_request_id(None)

    def test_accepts_a_sane_inbound_token(self):
        assert normalize_request_id("abc-123_DEF") == "abc-123_DEF"

    def test_rejects_overlong_token_and_generates_instead(self):
        long = "a" * 200
        rid = normalize_request_id(long)
        assert rid != long
        assert len(rid) == 32

    def test_rejects_token_with_unsafe_characters(self):
        # spaces, control chars, header-injection newlines, etc. are rejected
        for bad in ["has space", "line\nbreak", "semi;colon", "", "  "]:
            rid = normalize_request_id(bad)
            assert rid != bad
            assert len(rid) == 32


class TestStageTimings:
    def test_records_named_durations_in_ms(self):
        timings = StageTimings()
        with timings.stage("retrieval"):
            time.sleep(0.01)
        recorded = timings.as_dict()
        assert "retrieval_ms" in recorded
        assert isinstance(recorded["retrieval_ms"], (int, float))
        assert recorded["retrieval_ms"] >= 5  # at least ~10ms slept

    def test_unrun_stages_absent(self):
        timings = StageTimings()
        with timings.stage("retrieval"):
            pass
        recorded = timings.as_dict()
        assert "retrieval_ms" in recorded
        # rerank/llm never ran => not present
        assert "rerank_ms" not in recorded
        assert "llm_ms" not in recorded

    def test_total_ms_is_available(self):
        timings = StageTimings()
        with timings.stage("retrieval"):
            time.sleep(0.005)
        assert timings.total_ms() >= 0


class TestReadinessReport:
    def test_all_up_is_ready(self):
        checks = {"sqlite": lambda: None, "qdrant": lambda: None, "redis": lambda: None}
        ready, statuses = readiness_report(checks)
        assert ready is True
        assert statuses == {"sqlite": "ok", "qdrant": "ok", "redis": "ok"}

    def test_one_down_names_the_failing_dependency(self):
        def boom():
            raise RuntimeError("connection refused to redis://secret:pw@host:6379")

        checks = {"sqlite": lambda: None, "qdrant": lambda: None, "redis": boom}
        ready, statuses = readiness_report(checks)
        assert ready is False
        assert statuses["sqlite"] == "ok"
        assert statuses["qdrant"] == "ok"
        assert statuses["redis"] == "down"
        # the failing status must NOT leak the connection string / credentials
        assert "secret" not in str(statuses)
        assert "pw" not in str(statuses["redis"])
        assert "6379" not in str(statuses)


@pytest.mark.parametrize("missing", ["sqlite", "qdrant", "redis"])
def test_readiness_report_requires_all_three(missing):
    checks = {"sqlite": lambda: None, "qdrant": lambda: None, "redis": lambda: None}
    del checks[missing]
    with pytest.raises(ValueError):
        readiness_report(checks)
