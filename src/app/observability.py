"""Observability primitives (issue #21): structured logs, per-stage
timings, request-id propagation, and the readiness aggregation.

Design constraints, all enforced by tests:

- **Content is never logged.** Every log line carries ids, counts,
  durations, and flags only — never a customer message, chunk text,
  generated answer, or document body (CLAUDE.md PII rule). This module
  emits *no* content; callers pass ids/counts, and the summary line's
  schema is fixed to metadata.
- **One instrumentation seam.** ``StageTimings`` accumulates monotonic
  durations into a request-scoped object so the chat handler times
  retrieval / rerank / llm without editing every module; a single
  ``request_summary`` line is emitted per chat request.
- **Idempotent setup.** ``configure_logging`` binds structlog to render
  JSON to stdout with a timestamp + level, and is safe to call many
  times (create_app may run per-test) without leaking processor state or
  breaking other tests' logging.

The readiness probe (:func:`readiness_report`) pings each dependency by
running a tiny supplied callable; it names a failing dependency by its
key and never surfaces the underlying exception text, so a connection
string or credential embedded in an error can never leak.
"""

import re
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Inbound X-Request-ID is untrusted: accept only a short, opaque token
# (the same alphabet ids elsewhere use) so a header can never smuggle
# whitespace, control characters, or a log/header-injection payload into
# the bound log context. Anything else is replaced with a generated id.
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

REQUEST_ID_HEADER = "X-Request-ID"

# The three dependencies the readiness probe must report on.
_REQUIRED_DEPENDENCIES = ("sqlite", "qdrant", "redis")

_configured = False


def configure_logging() -> None:
    """Configure structlog to render JSON to stdout, idempotently.

    Adds an ISO timestamp and level to every event, merges any bound
    contextvars (so the request id rides along), and renders JSON.
    Guarded by a module flag so repeated calls (create_app runs once per
    test) never stack processors or disturb the stdlib logging other
    tests rely on.
    """
    global _configured
    if _configured:
        return
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        # A plain logger to stdout; no stdlib-logging integration, so this
        # cannot change the root logger's level or handlers under other tests.
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def normalize_request_id(raw: str | None) -> str:
    """Return a safe request id: the inbound token if it is a sane short
    token, otherwise a fresh uuid4 hex.

    A valid inbound id is echoed so a caller can correlate its request;
    anything absent, empty, over-long, or containing unsafe characters is
    replaced with a generated id rather than trusted.
    """
    if raw is not None and _REQUEST_ID_RE.fullmatch(raw):
        return raw
    return uuid.uuid4().hex


class StageTimings:
    """Request-scoped accumulator of per-stage monotonic durations (ms).

    ``stage(name)`` is a context manager timing one pipeline stage; a
    stage that never runs is simply absent from :meth:`as_dict`, so a
    cache hit (no retrieval/rerank/llm) or a grounded refusal (no
    rerank/llm) reports those stages as omitted. This is the single
    timing seam the chat handler wires in — no per-module edits.
    """

    def __init__(self) -> None:
        self._durations_ms: dict[str, float] = {}
        self._start = time.monotonic()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.monotonic()
        try:
            yield
        finally:
            elapsed_ms = (time.monotonic() - started) * 1000
            # accumulate, so a stage entered twice sums rather than clobbers
            self._durations_ms[name] = self._durations_ms.get(name, 0.0) + elapsed_ms

    def total_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000

    def as_dict(self) -> dict[str, float]:
        """The recorded stage durations, keyed ``{name}_ms`` and rounded.

        Only stages that actually ran are present; ``total_ms`` is added
        by the summary emitter, not here.
        """
        return {f"{name}_ms": round(ms, 1) for name, ms in self._durations_ms.items()}


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Bind a request id into the structlog contextvars for the request.

    Reads an inbound ``X-Request-ID`` (validated/replaced by
    :func:`normalize_request_id`), binds it so every log line in the
    request carries ``request_id``, echoes it in the response header, and
    clears the contextvars afterward so ids never bleed across requests.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], object]) -> Response:
        request_id = normalize_request_id(request.headers.get(REQUEST_ID_HEADER))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        finally:
            # Clear before returning so no id leaks into the next request
            # handled on this thread/task.
            structlog.contextvars.clear_contextvars()
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


def emit_request_summary(
    *,
    request_id: str,
    route: str,
    status: int,
    cache_hit: bool,
    timings: StageTimings,
) -> None:
    """Emit exactly one structured ``request_summary`` line for a request.

    Carries ids, the route, status, the ``cache_hit`` flag, the per-stage
    durations that ran (omitted stages absent), and ``total_ms`` — never
    any customer content.
    """
    logger = structlog.get_logger("app.request")
    logger.info(
        "request_summary",
        request_id=request_id,
        route=route,
        status=status,
        cache_hit=cache_hit,
        total_ms=round(timings.total_ms(), 1),
        **timings.as_dict(),
    )


def ready_checks_from_handles(
    *, sqlite: object, qdrant: object, redis: object
) -> dict[str, Callable[[], object]]:
    """Build the three readiness check callables from dependency handles.

    Each handle is a small, dedicated client that :func:`readiness_report`
    exercises with a trivial round-trip:

    - SQLite: ``SELECT 1`` on a read connection.
    - Qdrant: ``get_collections()`` (a lightweight metadata call).
    - Redis: ``ping()``.

    The handles are supplied by the app lifespan so the probe never
    disturbs the ingest/chat connections' single-thread contract.
    """
    return {
        "sqlite": lambda: sqlite.execute("SELECT 1"),
        "qdrant": qdrant.get_collections,
        "redis": redis.ping,
    }


def readiness_report(
    checks: dict[str, Callable[[], object]],
) -> tuple[bool, dict[str, str]]:
    """Run each dependency check and aggregate an overall readiness verdict.

    ``checks`` must cover exactly the required dependencies (sqlite,
    qdrant, redis). Each value is a zero-arg callable that raises on
    failure. Returns ``(ready, {dependency: "ok"|"down"})`` — the failing
    dependency is named by its key, and the underlying exception (which
    may embed a connection string or credential) is deliberately swallowed
    so nothing sensitive leaks.
    """
    missing = set(_REQUIRED_DEPENDENCIES) - set(checks)
    if missing:
        raise ValueError(f"readiness checks missing dependencies: {sorted(missing)}")
    statuses: dict[str, str] = {}
    ready = True
    for name, check in checks.items():
        try:
            check()
            statuses[name] = "ok"
        except Exception:
            # Name the dependency only; never surface the exception text,
            # which can carry a URL/credential.
            statuses[name] = "down"
            ready = False
    return ready, statuses
