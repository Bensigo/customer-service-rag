"""FastAPI application factory and resource lifecycle.

create_app() wires the app-lifetime resources the upload API needs in a
lifespan: the ingestion service's collaborators are expensive and
stateful (the Embedder holds a pooled HTTP client and dim() is a live
Ollama round-trip; ChunkStore owns a SQLite connection; VectorStore
holds a Qdrant client), so they are built once at startup, shared across
requests via app.state, and closed on shutdown — never per-request.

Closes run in reverse construction order and every close is attempted
even if an earlier one raises, so one failing close can't leak the rest.
"""

import logging
import sqlite3
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import anyio
import anyio.to_thread
import redis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient

from app.api import chat, documents
from app.chat.llm import create_llm_client
from app.config import Settings, get_settings
from app.ingestion.pipeline import IngestionService
from app.observability import (
    RequestIdMiddleware,
    configure_logging,
    readiness_report,
    ready_checks_from_handles,
)
from app.retrieval.embedder import create_embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pool import RetrieverPool
from app.retrieval.reranker import create_reranker
from app.stores.bm25_index import Bm25Index
from app.stores.cache import RedisCacheInvalidator, ResponseCache
from app.stores.chunk_store import ChunkStore
from app.stores.sessions import SessionStore
from app.stores.vector_store import VectorStore

logger = logging.getLogger("app.main")

# Short per-dependency timeout for the /ready probe: a readiness check must
# fail fast rather than hang an orchestrator's health poll behind a slow dep.
_READY_TIMEOUT_SECONDS = 2.0


def _close_all(closers: list[tuple[str, Callable[[], None]]]) -> None:
    """Run every close in order, logging and swallowing individual failures.

    Shutdown must release as many resources as possible: one connection
    that fails to close cannot be allowed to skip the others. Failures are
    logged (by resource name only) and the rest still run.
    """
    for name, close in closers:
        try:
            close()
        except Exception:
            logger.exception("failed to close %s during shutdown", name)


def _build_ingestion_service(
    settings: Settings,
) -> tuple[IngestionService, list[tuple[str, Callable[[], None]]]]:
    """Construct the ingestion service and its ordered list of closers.

    Closers are appended as each resource opens and returned in reverse
    (LIFO) order, so a failure partway through startup still tears down
    everything already built.
    """
    closers: list[tuple[str, Callable[[], None]]] = []
    try:
        chunk_store = ChunkStore(settings.db_path)
        closers.append(("chunk store", chunk_store.close))
        # second connection to the SAME WAL database for the BM25 index,
        # as the pipeline expects (chunk store and index co-located).
        # check_same_thread=False mirrors ChunkStore: created here on the
        # startup thread, used from the single ingest worker thread.
        bm25_conn = sqlite3.connect(settings.db_path, check_same_thread=False)
        closers.append(("bm25 connection", bm25_conn.close))
        bm25_index = Bm25Index(bm25_conn)

        embedder = create_embedder(settings)
        closers.append(("embedder", embedder.close))
        vector_size = embedder.dim()  # live Ollama round-trip on first call

        vector_store = VectorStore(
            settings.qdrant_url,
            vector_size=vector_size,
            collection=settings.qdrant_collection,
        )
        closers.append(("vector store", vector_store.close))
        vector_store.ensure_collection()

        # Real cache invalidator (#20): a successful ingest now evicts the
        # document's cached first-turn answers from the same Redis the
        # response cache writes. It owns its own redis-py client, closed on
        # shutdown. Fail-open internally, so a cache outage never fails ingest.
        invalidator = RedisCacheInvalidator(settings.redis_url)
        closers.append(("cache invalidator", invalidator.close))
    except Exception:
        _close_all(list(reversed(closers)))
        raise

    service = IngestionService(chunk_store, bm25_index, embedder, vector_store, invalidator)
    return service, list(reversed(closers))


# How many concurrent read connections the chat path may hold open.
_RETRIEVER_POOL_SIZE = 4


def _build_chat_stack(
    settings: Settings,
) -> tuple[dict[str, object], list[tuple[str, Callable[[], None]]]]:
    """Construct the chat/retrieval collaborators and their closers.

    Ownership is deliberately separate from the ingest path. The embedder
    and vector store are read-only network clients built fresh here (their
    own pooled HTTP / Qdrant clients); the retriever POOL gives each of
    its slots its OWN SQLite chunk-store and BM25 read connections, so no
    reader thread ever shares a connection with another reader or with the
    ingest writer (the #12 threading hazard). WAL mode keeps these
    independent read connections concurrent and isolated from writes.

    Returns the collaborators to place on app.state and an ordered list of
    closers (reverse construction order), so a failure partway through
    startup still tears down everything already built.
    """
    closers: list[tuple[str, Callable[[], None]]] = []
    try:
        embedder = create_embedder(settings)
        closers.append(("chat embedder", embedder.close))
        vector_size = embedder.dim()  # live Ollama round-trip on first call
        vector_store = VectorStore(
            settings.qdrant_url,
            vector_size=vector_size,
            collection=settings.qdrant_collection,
        )
        closers.append(("chat vector store", vector_store.close))
        vector_store.ensure_collection()

        def _make_retriever() -> tuple[HybridRetriever, Callable[[], None]]:
            # One dedicated read connection pair per pool slot, to the same
            # WAL database the ingest path writes — never a shared connection.
            chunk_store = ChunkStore(settings.db_path)
            bm25_conn = sqlite3.connect(settings.db_path, check_same_thread=False)
            bm25_index = Bm25Index(bm25_conn)

            def _close() -> None:
                chunk_store.close()
                bm25_conn.close()

            retriever = HybridRetriever(bm25_index, embedder, vector_store, chunk_store)
            return retriever, _close

        retriever_pool = RetrieverPool(_make_retriever, size=_RETRIEVER_POOL_SIZE)
        closers.append(("retriever pool", retriever_pool.close))

        reranker = create_reranker(settings)
        closers.append(("reranker", reranker.close))

        session_store = SessionStore(settings.redis_url)
        closers.append(("session store", session_store.close))

        # First-turn response cache: same Redis as sessions, its own client.
        response_cache = ResponseCache(settings.redis_url)
        closers.append(("response cache", response_cache.close))

        llm_client = create_llm_client(settings)
        closers.append(("llm client", llm_client.close))
    except Exception:
        _close_all(list(reversed(closers)))
        raise

    state: dict[str, object] = {
        "retriever": retriever_pool,
        "reranker": reranker,
        "session_store": session_store,
        "response_cache": response_cache,
        "llm_client": llm_client,
    }
    return state, list(reversed(closers))


def _build_ready_handles(
    settings: Settings,
) -> tuple[dict[str, object], list[tuple[str, Callable[[], None]]]]:
    """Build dedicated, short-timeout handles for the /ready probe.

    These are separate from the ingest/chat connections on purpose: the
    readiness probe runs on the request event loop's threadpool and must
    never share a connection with the single-thread ingest writer or a
    pooled reader (the #12 threading contract). Each is a tiny client that
    does one trivial round-trip and fails fast on a short timeout.

    Returns the handles to place on app.state and their closers (reverse
    construction order), so a failure partway through startup still tears
    down everything already built.
    """
    closers: list[tuple[str, Callable[[], None]]] = []
    try:
        # A read-only SQLite connection for `SELECT 1`; its own connection so
        # a probe never touches the ingest writer or a pooled reader.
        sqlite_conn = sqlite3.connect(
            settings.db_path, check_same_thread=False, timeout=_READY_TIMEOUT_SECONDS
        )
        closers.append(("ready sqlite", sqlite_conn.close))
        qdrant_client = QdrantClient(
            location=settings.qdrant_url, timeout=int(_READY_TIMEOUT_SECONDS)
        )
        closers.append(("ready qdrant", qdrant_client.close))
        redis_client = redis.Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=_READY_TIMEOUT_SECONDS,
            socket_timeout=_READY_TIMEOUT_SECONDS,
        )
        closers.append(("ready redis", redis_client.close))
    except Exception:
        _close_all(list(reversed(closers)))
        raise

    handles: dict[str, object] = {
        "ready_sqlite": sqlite_conn,
        "ready_qdrant": qdrant_client,
        "ready_redis": redis_client,
    }
    return handles, list(reversed(closers))


def create_app() -> FastAPI:
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Test seam: when a caller has pre-stashed an ingestion_service
        # (unit tests injecting a fake), skip building the live resources
        # so no Ollama/Qdrant/SQLite is touched and nothing needs closing.
        preinjected = getattr(app.state, "ingestion_service", None) is not None
        need_max_bytes = getattr(app.state, "max_upload_bytes", None) is None
        need_cache_ttl = getattr(app.state, "cache_ttl_seconds", None) is None
        need_min_rerank = getattr(app.state, "min_rerank_score", None) is None
        closers: list[tuple[str, Callable[[], None]]] = []
        # Only load Settings when something actually needs it, so unit tests
        # that inject a fake and preset the cap never require live config.
        if not preinjected or need_max_bytes or need_cache_ttl or need_min_rerank:
            settings = get_settings()
            if not preinjected:
                service, closers = _build_ingestion_service(settings)
                app.state.ingestion_service = service
                # The chat/retrieval stack shares the same "skip when a fake
                # is preinjected" seam so the unit suite stays offline.
                try:
                    chat_state, chat_closers = _build_chat_stack(settings)
                except Exception:
                    _close_all(closers)
                    raise
                for key, value in chat_state.items():
                    setattr(app.state, key, value)
                # closers run in reverse order; chat resources built after
                # ingest must be torn down before it.
                closers = chat_closers + closers
                # Dedicated /ready probe handles (#21): their own tiny,
                # short-timeout clients, never the ingest/chat connections.
                try:
                    ready_handles, ready_closers = _build_ready_handles(settings)
                except Exception:
                    _close_all(closers)
                    raise
                for key, value in ready_handles.items():
                    setattr(app.state, key, value)
                closers = ready_closers + closers
            if need_max_bytes:
                app.state.max_upload_bytes = settings.max_upload_bytes
            # The chat cache read-through resolves its TTL from app.state, so
            # unit tests that override the cache dep still get a real value.
            if need_cache_ttl:
                app.state.cache_ttl_seconds = settings.cache_ttl_seconds
            # The chat relevance gate (#50) resolves its floor from app.state
            # the same way, so overriding the reranker dep still gets a value.
            if need_min_rerank:
                app.state.min_rerank_score = settings.min_rerank_score
        # single-permit limiter: extraction + ingestion run one at a time
        # off the event loop (see app.api.documents for why one worker)
        app.state.ingest_limiter = anyio.CapacityLimiter(1)
        try:
            yield
        finally:
            _close_all(closers)

    app = FastAPI(title="customer-service-rag", lifespan=lifespan)
    # Request-id middleware wraps every request: bind the id into the log
    # context and echo it in the response header (#21).
    app.add_middleware(RequestIdMiddleware)
    app.include_router(documents.router)
    app.include_router(chat.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe: process is up. No dependency checks (readiness is #21)."""
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        """Readiness probe (#21): ping SQLite, Qdrant, and Redis with short
        timeouts. 200 when all up; 503 naming the down dependency otherwise.

        Never leaks a connection string or credential: a failed check is
        reported by dependency name only (see readiness_report). Unlike
        /health, this deliberately touches the (dedicated) dependency
        handles — an orchestrator uses it to decide whether to route
        traffic. The blocking pings run off the event loop so a slow or
        hung dependency never freezes concurrent requests.
        """
        checks = ready_checks_from_handles(
            sqlite=request.app.state.ready_sqlite,
            qdrant=request.app.state.ready_qdrant,
            redis=request.app.state.ready_redis,
        )
        is_ready, statuses = await anyio.to_thread.run_sync(readiness_report, checks)
        status_code = 200 if is_ready else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ready" if is_ready else "not_ready",
                "dependencies": statuses,
            },
        )

    return app
