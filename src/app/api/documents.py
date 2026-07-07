"""Document upload API: POST /documents and PUT /documents/{doc_id}.

The HTTP boundary around #8's ``extract`` and #11's ``IngestionService``.
Its job is the enforcement the ingestion service deliberately does not
repeat, so nothing untrusted reaches the pipeline:

- doc_id is validated against ``_DOC_ID_RE`` here, before ingest is
  called (the service does not re-validate — see its docstring). Both
  the PUT path parameter and the POST-derived/provided id go through the
  same check, because the id lands in cache tags and Qdrant filters.
- The body is read incrementally and rejected at ``max_upload_bytes``
  without ever buffering an unbounded amount — the DoS cap in front of
  pypdf, which parses attacker-controlled bytes in-process (#8 note).
- Uploaded filenames are reduced to their basename (POSIX *and* Windows)
  before a slug or title is derived, so ``C:\\path\\a.txt`` cannot leak
  a path prefix into the doc_id or title (#8 note).

Extraction and ingestion are synchronous blocking work (pypdf parsing,
Ollama embedding, SQLite + Qdrant writes) and run by design in-request,
but off the event loop via a single-permit ``CapacityLimiter``: one
worker thread at a time, so a slow ingest never freezes concurrent
requests and ``extract``'s process-global ``warnings`` filter can never
overlap another extraction's scope (#8 note).

Error responses and log lines carry only the doc_id, filename, byte
size, and reason — never file content, which may hold customer PII.
"""

import logging
import re
import time
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Annotated

import anyio
import anyio.to_thread
from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile

from app.ingestion.extractors import (
    EmptyDocumentError,
    ExtractionError,
    UnsupportedFileType,
    extract,
)
from app.ingestion.pipeline import IngestError, IngestResult

logger = logging.getLogger("app.api.documents")

router = APIRouter(tags=["documents"])

# doc_id lands in cache tags and Qdrant payload filters, so it is held to
# a strict charset: lowercase alphanumerics and hyphens, 1..64 chars,
# never leading with a hyphen. Kept in lockstep with IngestionService's
# documented precondition.
# \Z (not $) so a trailing newline can't slip through; matched with fullmatch.
_DOC_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")

# 64 KiB read granularity: large enough to be cheap, small enough that
# the over-limit read never buffers much past the cap.
_READ_CHUNK_BYTES = 64 * 1024


class IngestionServiceProtocol:
    """Structural stand-in for typing only (see app.ingestion.pipeline)."""

    def ingest(self, doc_id: str, title: str, text: str) -> IngestResult: ...


def get_ingestion_service(request: Request) -> IngestionServiceProtocol:
    """Resolve the app-lifetime IngestionService stashed on app.state.

    A plain function so tests can swap it via dependency_overrides; the
    real instance is built once in create_app()'s lifespan.
    """
    return request.app.state.ingestion_service


def _max_upload_bytes(request: Request) -> int:
    return request.app.state.max_upload_bytes


def _offload_limiter(request: Request) -> anyio.CapacityLimiter:
    return request.app.state.ingest_limiter


# Annotated dependency aliases (the FastAPI-idiomatic form that keeps the
# Depends call out of a default argument, so ruff's B008 stays quiet).
IngestionServiceDep = Annotated[IngestionServiceProtocol, Depends(get_ingestion_service)]
MaxUploadBytesDep = Annotated[int, Depends(_max_upload_bytes)]
OffloadLimiterDep = Annotated[anyio.CapacityLimiter, Depends(_offload_limiter)]
DocIdForm = Annotated[str | None, Form()]


def _basename(filename: str) -> str:
    """Strip any directory prefix, POSIX or Windows.

    ``extract`` treats filenames lexically, so a Windows path like
    ``C:\\docs\\a.txt`` would otherwise keep its prefix in a stem-derived
    title. PureWindowsPath splits on both separators and drops drive
    letters, covering POSIX names too.
    """
    return PureWindowsPath(filename).name


def _slug_from_filename(filename: str) -> str:
    """Derive a doc_id slug from a filename's basename stem.

    Lowercases, collapses every run of non ``[a-z0-9]`` into a single
    hyphen, and trims leading/trailing hyphens. May return "" (e.g. a
    stem of only punctuation); the caller validates the result against
    _DOC_ID_RE and rejects an unusable slug rather than inventing an id.
    """
    stem = PureWindowsPath(_basename(filename)).stem
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return slug[:64]


@dataclass(frozen=True, slots=True)
class _UploadOutcome:
    doc_id: str
    version: int
    chunk_count: int


async def _read_capped(upload: UploadFile, max_bytes: int) -> bytes:
    """Read the upload body, rejecting once it would exceed max_bytes.

    Reads in bounded chunks and stops the moment the running total passes
    the cap, so an oversized (or lying-Content-Length) upload never
    buffers more than one chunk past the limit. Returns the full body
    when it fits.
    """
    buffer = bytearray()
    while True:
        chunk = await upload.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise HTTPException(
                status_code=413,  # Content Too Large
                detail=f"upload exceeds the maximum of {max_bytes} bytes",
            )
    return bytes(buffer)


def _ingest_blocking(
    service: IngestionServiceProtocol,
    doc_id: str,
    filename: str,
    content: bytes,
) -> IngestResult:
    """Extract then ingest — the blocking work run on the offload thread.

    Kept together so a single worker thread does extraction (which mutates
    the process-global warnings filter) and ingestion back to back, never
    overlapping another request's extraction. Exceptions propagate to the
    caller, which maps them to HTTP status codes.
    """
    extracted = extract(filename, content)
    return service.ingest(doc_id, extracted.title, extracted.text)


async def _handle_upload(
    *,
    upload: UploadFile,
    doc_id: str | None,
    service: IngestionServiceProtocol,
    max_bytes: int,
    limiter: anyio.CapacityLimiter,
) -> _UploadOutcome:
    filename = _basename(upload.filename or "")
    resolved_id = doc_id if doc_id is not None else _slug_from_filename(filename)
    if not _DOC_ID_RE.fullmatch(resolved_id):
        # 422: the id is unusable (bad explicit id, or a filename with no
        # sluggable characters). Message names the offending id only.
        raise HTTPException(
            status_code=422,  # Unprocessable Content
            detail=(
                f"invalid document id {resolved_id!r}; must match {_DOC_ID_RE.pattern} "
                "(lowercase letters, digits, and hyphens; 1-64 chars; no leading hyphen)"
            ),
        )

    content = await _read_capped(upload, max_bytes)
    size = len(content)
    started = time.monotonic()
    try:
        result = await anyio.to_thread.run_sync(
            _ingest_blocking, service, resolved_id, filename, content, limiter=limiter
        )
    except UnsupportedFileType as error:
        # extractors document these messages as safe (filename + byte size only)
        raise HTTPException(status_code=415, detail=str(error)) from error
    except (EmptyDocumentError, ExtractionError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except IngestError as error:
        # Do NOT relay str(error): IngestError messages are meant to carry
        # only ids/versions, but the API must not depend on that — a safe
        # message is synthesized here from the validated id and filename so
        # no document text can leak into the response even if the pipeline
        # ever embeds it. The full error is logged server-side (below).
        logger.warning(
            "ingest failed doc_id=%s filename=%s error_type=%s",
            resolved_id,
            filename,
            type(error).__name__,
        )
        raise HTTPException(
            status_code=422,
            detail=f"could not ingest document {resolved_id!r} from {filename!r}",
        ) from error
    duration = time.monotonic() - started
    logger.info(
        "ingested document doc_id=%s filename=%s size=%d chunk_count=%d duration_ms=%.1f",
        resolved_id,
        filename,
        size,
        result.chunk_count,
        duration * 1000,
    )
    return _UploadOutcome(
        doc_id=result.doc_id, version=result.version, chunk_count=result.chunk_count
    )


@router.post("/documents", status_code=201)
async def create_document(
    file: UploadFile,
    service: IngestionServiceDep,
    max_bytes: MaxUploadBytesDep,
    limiter: OffloadLimiterDep,
    doc_id: DocIdForm = None,
) -> dict[str, object]:
    """Upload a new document; doc_id is the form field or a filename slug."""
    outcome = await _handle_upload(
        upload=file,
        doc_id=doc_id,
        service=service,
        max_bytes=max_bytes,
        limiter=limiter,
    )
    return {
        "doc_id": outcome.doc_id,
        "version": outcome.version,
        "chunk_count": outcome.chunk_count,
    }


@router.put("/documents/{doc_id}")
async def update_document(
    doc_id: str,
    file: UploadFile,
    service: IngestionServiceDep,
    max_bytes: MaxUploadBytesDep,
    limiter: OffloadLimiterDep,
) -> dict[str, object]:
    """Re-upload/update a document under an explicit doc_id (200)."""
    outcome = await _handle_upload(
        upload=file,
        doc_id=doc_id,
        service=service,
        max_bytes=max_bytes,
        limiter=limiter,
    )
    return {
        "doc_id": outcome.doc_id,
        "version": outcome.version,
        "chunk_count": outcome.chunk_count,
    }
