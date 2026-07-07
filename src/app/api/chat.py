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
from app.chat.llm import LLMClient, LLMError
from app.models import Message, RetrievedChunk, SourceRef, Turn

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


def get_session_store(request: Request) -> SupportsSessionStore:
    return request.app.state.session_store


def get_retriever(request: Request) -> SupportsRetriever:
    return request.app.state.retriever


def get_reranker(request: Request) -> SupportsReranker:
    return request.app.state.reranker


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


SessionStoreDep = Annotated[SupportsSessionStore, Depends(get_session_store)]
RetrieverDep = Annotated[SupportsRetriever, Depends(get_retriever)]
RerankerDep = Annotated[SupportsReranker, Depends(get_reranker)]
LLMClientDep = Annotated[LLMClient, Depends(get_llm_client)]


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


def _sources(refs: list[SourceRef]) -> list[SourceModel]:
    return [SourceModel(chunk_id=ref.chunk_id, doc_id=ref.doc_id, title=ref.title) for ref in refs]


def _generate(llm: LLMClient, messages: list[Message]) -> str:
    return llm.complete(messages)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    sessions: SessionStoreDep,
    retriever: RetrieverDep,
    reranker: RerankerDep,
    llm: LLMClientDep,
) -> ChatResponse | JSONResponse:
    session_id = body.session_id
    question = body.message  # validated: stripped and non-blank

    history = sessions.get_history(session_id, limit=_HISTORY_LIMIT)

    # Blocking retrieval off the event loop (default threadpool: concurrent).
    candidates = await anyio.to_thread.run_sync(retriever.retrieve, question)

    if not candidates:
        # Grounded refusal: no sources => never call the model.
        logger.info("chat no-context escalation session=%s", session_id)
        return ChatResponse(answer=ESCALATION_ANSWER, sources=[])

    reranked = await anyio.to_thread.run_sync(
        lambda: reranker.rerank(question, candidates, top_n=_RERANK_TOP_N)
    )
    assembled = assemble_context(question, reranked, history)

    try:
        answer = await anyio.to_thread.run_sync(_generate, llm, assembled.messages)
    except LLMError:
        # No turns are appended: the session is untouched so a retry is clean.
        logger.warning("chat generation unavailable session=%s", session_id)
        return JSONResponse(status_code=503, content={"error": "generation_unavailable"})

    sessions.append_turn(session_id, Turn(role="user", content=question))
    sessions.append_turn(session_id, Turn(role="assistant", content=answer))

    logger.info(
        "chat answered session=%s source_count=%d",
        session_id,
        len(assembled.source_refs),
    )
    return ChatResponse(answer=answer, sources=_sources(assembled.source_refs))
