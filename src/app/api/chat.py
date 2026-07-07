"""Grounded chat API: POST /chat (issue #18).

The HTTP boundary that strings the query pipeline together:

    get_history -> retrieve -> rerank -> assemble_context -> generate
    -> append user+assistant turns -> respond with citations

Design rules baked in here:

- **Grounded refusal.** If retrieval returns no chunks, the LLM is
  skipped entirely and a fixed escalation answer is returned with empty
  sources. A support bot must refuse rather than hallucinate — this path
  is deterministic and never reaches the model.
- **Generation failure.** A timeout/error from the LLM (after its own
  bounded retries) becomes ``503 {"error": "generation_unavailable"}``,
  and *no* turns are appended: a failed request leaves the session
  exactly as it was (neither the user nor the assistant turn is stored),
  so a retry replays cleanly.
- **Off the event loop.** retrieve, rerank, and generate are synchronous
  blocking work; each runs via ``anyio.to_thread.run_sync`` on the
  default worker threadpool (not the single-permit ingest limiter — chat
  reads are served concurrently), so a slow generation never freezes
  concurrent requests such as /health.

Validation: session_id must match the session store's id grammar and
the message must be 1..2000 non-blank characters — both enforced by the
request model (422 on violation) before any backend is touched.

Logging carries ids and counts only — never the message, chunk text, or
the generated answer, all of which are customer content (CLAUDE.md).
"""

import logging
from typing import Annotated, Protocol

import anyio.to_thread
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.chat.context import assemble_context
from app.chat.fingerprint import fingerprint
from app.chat.llm import LLMClient, LLMError
from app.models import Message, RetrievedChunk, SourceRef, Turn
from app.observability import StageTimings, emit_request_summary
from app.stores.cache import CachedResponse

logger = logging.getLogger("app.api.chat")

router = APIRouter(tags=["chat"])

# Returned verbatim when retrieval finds nothing: refuse, don't hallucinate.
ESCALATION_ANSWER = (
    "I couldn't find anything about that in our help center, so I don't want "
    "to guess. I'll escalate this to a human support agent who can help."
)

# How many turns of prior history to replay into the prompt.
_HISTORY_LIMIT = 10
# Rerank keeps the strongest few chunks that actually reach the model.
_RERANK_TOP_N = 5


# --- collaborator seams (resolved from app.state, overridable in tests) ---


class SupportsSessionStore(Protocol):
    def get_history(self, session_id: str, limit: int = 10) -> list[Turn]: ...

    def append_turn(self, session_id: str, turn: Turn) -> None: ...


class SupportsRetriever(Protocol):
    def retrieve(
        self, query: str, *, k_each: int = 20, top_n: int = 12
    ) -> list[RetrievedChunk]: ...


class SupportsReranker(Protocol):
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int = 5
    ) -> list[RetrievedChunk]: ...


class SupportsResponseCache(Protocol):
    def get(self, fp: str) -> CachedResponse | None: ...

    def set(
        self, fp: str, response: CachedResponse, source_doc_ids: list[str], ttl: int
    ) -> None: ...


def get_session_store(request: Request) -> SupportsSessionStore:
    return request.app.state.session_store


def get_retriever(request: Request) -> SupportsRetriever:
    return request.app.state.retriever


def get_reranker(request: Request) -> SupportsReranker:
    return request.app.state.reranker


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


def get_response_cache(request: Request) -> SupportsResponseCache:
    return request.app.state.response_cache


def get_cache_ttl_seconds(request: Request) -> int:
    return request.app.state.cache_ttl_seconds


SessionStoreDep = Annotated[SupportsSessionStore, Depends(get_session_store)]
RetrieverDep = Annotated[SupportsRetriever, Depends(get_retriever)]
RerankerDep = Annotated[SupportsReranker, Depends(get_reranker)]
LLMClientDep = Annotated[LLMClient, Depends(get_llm_client)]
ResponseCacheDep = Annotated[SupportsResponseCache, Depends(get_response_cache)]
CacheTtlDep = Annotated[int, Depends(get_cache_ttl_seconds)]


class ChatRequest(BaseModel):
    # Same grammar the session store enforces (defense in depth): reject a
    # malformed id at the edge with a 422 rather than a 500 from the store.
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{8,64}$")
    # 1..2000 chars, non-blank once stripped — enforced during validation so
    # an empty or whitespace-only question is a uniform 422, never reaching
    # retrieval. The stripped value is what downstream code uses.
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def _strip_and_require_content(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped


class SourceModel(BaseModel):
    chunk_id: str
    doc_id: str
    title: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceModel]
    # True only when served from the response cache (first-turn hit); the
    # live pipeline always returns False.
    cached: bool = False


def _sources(refs: list[SourceRef]) -> list[SourceModel]:
    return [SourceModel(chunk_id=ref.chunk_id, doc_id=ref.doc_id, title=ref.title) for ref in refs]


def _generate(llm: LLMClient, messages: list[Message]) -> str:
    return llm.complete(messages)


async def _append_turns(
    sessions: SupportsSessionStore, session_id: str, question: str, answer: str
) -> None:
    """Append the user then assistant turn, each offloaded (blocking
    redis-py), so the conversation continues after a hit or a miss."""
    await anyio.to_thread.run_sync(
        lambda: sessions.append_turn(session_id, Turn(role="user", content=question))
    )
    await anyio.to_thread.run_sync(
        lambda: sessions.append_turn(session_id, Turn(role="assistant", content=answer))
    )


async def _cache_get(cache: SupportsResponseCache, fp: str) -> CachedResponse | None:
    """Offloaded cache read that fails open on ANY error: a cache outage —
    or even a bug in the cache — must degrade to a miss, never take chat
    down. ResponseCache already guards Redis errors; this is the outer
    belt-and-suspenders. Logs the operation only, never the key/content."""
    try:
        return await anyio.to_thread.run_sync(cache.get, fp)
    except Exception:
        logger.warning("chat cache read failed; treating as miss")
        return None


async def _cache_set(
    cache: SupportsResponseCache,
    fp: str,
    response: CachedResponse,
    source_doc_ids: list[str],
    ttl: int,
) -> None:
    """Offloaded cache write that fails open on ANY error: a failed write
    must never surface, the caller already has a live answer."""
    try:
        await anyio.to_thread.run_sync(lambda: cache.set(fp, response, source_doc_ids, ttl))
    except Exception:
        logger.warning("chat cache write failed; skipping")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    sessions: SessionStoreDep,
    retriever: RetrieverDep,
    reranker: RerankerDep,
    llm: LLMClientDep,
    cache: ResponseCacheDep,
    cache_ttl: CacheTtlDep,
) -> ChatResponse | JSONResponse:
    """POST /chat — the single instrumentation seam for #21.

    Owns the request-scoped ``StageTimings`` and emits exactly one
    ``request_summary`` log line (ids/counts/durations/flags only, never
    content) per request, across every return path — cache hit, grounded
    refusal, generation failure, or a live answer.
    """
    timings = StageTimings()
    result = await _run_chat(body, sessions, retriever, reranker, llm, cache, cache_ttl, timings)
    emit_request_summary(
        request_id=getattr(request.state, "request_id", ""),
        route="/chat",
        status=result.status,
        cache_hit=result.cache_hit,
        timings=timings,
    )
    return result.response


class _ChatResult:
    """The chat response plus the summary-line metadata for #21."""

    __slots__ = ("response", "status", "cache_hit")

    def __init__(self, response: ChatResponse | JSONResponse, status: int, cache_hit: bool) -> None:
        self.response = response
        self.status = status
        self.cache_hit = cache_hit


async def _run_chat(
    body: ChatRequest,
    sessions: SupportsSessionStore,
    retriever: SupportsRetriever,
    reranker: SupportsReranker,
    llm: LLMClient,
    cache: SupportsResponseCache,
    cache_ttl: int,
    timings: StageTimings,
) -> _ChatResult:
    session_id = body.session_id
    question = body.message  # validated: stripped and non-blank

    # get_history is blocking redis-py I/O: offload it like every other
    # blocking stage so a slow Redis never freezes the event loop.
    history = await anyio.to_thread.run_sync(
        lambda: sessions.get_history(session_id, limit=_HISTORY_LIMIT)
    )

    # First-turn read-through cache. fingerprint() returns None for any turn
    # with prior history (deliberate first-turn-only policy), so follow-ups
    # bypass the cache entirely and run the full pipeline below.
    fp = fingerprint(question, history)
    if fp is not None:
        cached = await _cache_get(cache, fp)
        if cached is not None:
            # HIT: skip retrieval, rerank, and the LLM entirely (those stage
            # timers never run, so they are omitted from the summary). Still
            # append both turns so the conversation continues from here.
            await _append_turns(sessions, session_id, question, cached.answer)
            logger.info(
                "chat cache hit session=%s source_count=%d",
                session_id,
                len(cached.sources),
            )
            response = ChatResponse(
                answer=cached.answer, sources=_sources(cached.sources), cached=True
            )
            return _ChatResult(response, status=200, cache_hit=True)

    # Blocking retrieval off the event loop (default threadpool: concurrent).
    with timings.stage("retrieval"):
        candidates = await anyio.to_thread.run_sync(retriever.retrieve, question)

    if not candidates:
        # Grounded refusal: no sources => never call the model (rerank/llm
        # timers never run, so they are omitted from the summary).
        logger.info("chat no-context escalation session=%s", session_id)
        return _ChatResult(
            ChatResponse(answer=ESCALATION_ANSWER, sources=[]), status=200, cache_hit=False
        )

    with timings.stage("rerank"):
        reranked = await anyio.to_thread.run_sync(
            lambda: reranker.rerank(question, candidates, top_n=_RERANK_TOP_N)
        )
    assembled = assemble_context(question, reranked, history)

    try:
        with timings.stage("llm"):
            answer = await anyio.to_thread.run_sync(_generate, llm, assembled.messages)
    except LLMError:
        # No turns are appended: the session is untouched so a retry is clean.
        logger.warning("chat generation unavailable session=%s", session_id)
        return _ChatResult(
            JSONResponse(status_code=503, content={"error": "generation_unavailable"}),
            status=503,
            cache_hit=False,
        )

    if not answer.strip():
        # A blank answer (the thinking-model empty-output failure mode) is not
        # a usable reply: treat it as generation-unavailable rather than
        # returning a 200 with an empty answer. No turns are appended.
        logger.warning("chat generation empty session=%s", session_id)
        return _ChatResult(
            JSONResponse(status_code=503, content={"error": "generation_unavailable"}),
            status=503,
            cache_hit=False,
        )

    # Turns are appended only after a successful generation, so a failed
    # turn never lingers for a retry.
    await _append_turns(sessions, session_id, question, answer)

    # Write back into the first-turn cache (fp is not None only for first
    # turns). Tag by the unique source doc ids so #20 can evict by document.
    # Offloaded and fail-open inside the cache: a failed write never surfaces.
    if fp is not None:
        source_doc_ids = list(dict.fromkeys(ref.doc_id for ref in assembled.source_refs))
        cached_response = CachedResponse(answer=answer, sources=list(assembled.source_refs))
        await _cache_set(cache, fp, cached_response, source_doc_ids, cache_ttl)

    logger.info(
        "chat answered session=%s source_count=%d",
        session_id,
        len(assembled.source_refs),
    )
    response = ChatResponse(answer=answer, sources=_sources(assembled.source_refs), cached=False)
    return _ChatResult(response, status=200, cache_hit=False)
