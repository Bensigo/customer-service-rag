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
from fastapi import FastAPI

from app.api import documents
from app.config import Settings, get_settings
from app.ingestion.pipeline import IngestionService, NoopCacheInvalidator
from app.retrieval.embedder import create_embedder
from app.stores.bm25_index import Bm25Index
from app.stores.chunk_store import ChunkStore
from app.stores.vector_store import VectorStore

logger = logging.getLogger("app.main")


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
    except Exception:
        _close_all(list(reversed(closers)))
        raise

    service = IngestionService(
        chunk_store, bm25_index, embedder, vector_store, NoopCacheInvalidator()
    )
    return service, list(reversed(closers))


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Test seam: when a caller has pre-stashed an ingestion_service
        # (unit tests injecting a fake), skip building the live resources
        # so no Ollama/Qdrant/SQLite is touched and nothing needs closing.
        preinjected = getattr(app.state, "ingestion_service", None) is not None
        need_max_bytes = getattr(app.state, "max_upload_bytes", None) is None
        closers: list[tuple[str, Callable[[], None]]] = []
        # Only load Settings when something actually needs it, so unit tests
        # that inject a fake and preset the cap never require live config.
        if not preinjected or need_max_bytes:
            settings = get_settings()
            if not preinjected:
                service, closers = _build_ingestion_service(settings)
                app.state.ingestion_service = service
            if need_max_bytes:
                app.state.max_upload_bytes = settings.max_upload_bytes
        # single-permit limiter: extraction + ingestion run one at a time
        # off the event loop (see app.api.documents for why one worker)
        app.state.ingest_limiter = anyio.CapacityLimiter(1)
        try:
            yield
        finally:
            _close_all(closers)

    app = FastAPI(title="customer-service-rag", lifespan=lifespan)
    app.include_router(documents.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe: process is up. No dependency checks (readiness is #21)."""
        return {"status": "ok"}

    return app
