"""Ollama-backed text embedding with model-aware passage/query prompts.

Implementation choice: a thin direct client for Ollama's ``POST
/api/embed`` endpoint on httpx2, not langchain-ollama's
``OllamaEmbeddings``. One JSON endpoint does not justify pulling in the
langchain-core dependency tree, httpx2 is already vetted and locked in
this repo, and the direct call lets us send ``truncate: false`` so
over-context input fails loudly instead of being silently cut —
langchain-ollama does not expose that flag.

Prompt templates are pinned per model (``_TEMPLATES``): nomic-embed-text
was trained with ``search_document:`` / ``search_query:`` prefixes;
qwen3-embedding takes an instruction-style prompt on the query side only
(per its model card, ``Instruct: {task}\\nQuery:{text}`` with no space
after ``Query:``). Models without an entry fall back to the generic
search_document/search_query prefix pair so a query embedding always
differs from a passage embedding of the same text; add a real entry
before using a new model in production.

Vector dimension is not hardcoded: qwen3-embedding returns 4096 floats,
nomic-embed-text 768. ``Embedder.dim()`` embeds a one-character probe
once and caches the length; Qdrant collection bootstrap (#10) reads it.

Chunker coupling: the chunker's 350-token default budget must fit the
embed model's context window. Ollama reports context_length via
``/api/tags`` — qwen3-embedding 40960, nomic-embed-text 2048 — both far
above the worst-case chunk, and the integration test embeds one with
``truncate: false`` as a running sanity check.

Calls are synchronous blocking I/O by design; callers in async context
must offload them (e.g. ``anyio.to_thread.run_sync``) — enforced at the
call sites in #12/#18.
"""

from typing import NamedTuple, Protocol

import httpx2

from app.config import Settings

_DIM_PROBE = "x"
_REQUEST_TIMEOUT_SECONDS = 120.0


class EmbeddingError(Exception):
    """An embeddings request failed. Messages never include input text."""


class PromptTemplates(NamedTuple):
    """Format strings (``{text}`` placeholder) applied before embedding."""

    passage: str
    query: str


_DEFAULT_TEMPLATES = PromptTemplates(
    passage="search_document: {text}",
    query="search_query: {text}",
)

_TEMPLATES: dict[str, PromptTemplates] = {
    "nomic-embed-text": _DEFAULT_TEMPLATES,
    "qwen3-embedding": PromptTemplates(
        passage="{text}",
        query=(
            "Instruct: Given a web search query, retrieve relevant passages "
            "that answer the query\nQuery:{text}"
        ),
    ),
}


class SupportsEmbed(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def close(self) -> None: ...


class OllamaEmbeddingsClient:
    """Thin client for Ollama's /api/embed endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = _REQUEST_TIMEOUT_SECONDS,
        transport: httpx2.BaseTransport | None = None,
    ):
        self.base_url = base_url
        self.model = model
        self._http = httpx2.Client(base_url=base_url, timeout=timeout, transport=transport)

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts, "truncate": False}
        try:
            response = self._http.post("/api/embed", json=payload)
            response.raise_for_status()
        except httpx2.HTTPStatusError as error:
            raise EmbeddingError(
                f"Ollama embed request for model {self.model!r} failed with "
                f"HTTP {error.response.status_code}: {_error_detail(error.response)}"
            ) from error
        except httpx2.HTTPError as error:
            raise EmbeddingError(
                f"Ollama embed request to {self.base_url} failed: {error}"
            ) from error
        try:
            data = response.json()
        except ValueError as error:
            raise EmbeddingError(
                f"Ollama at {self.base_url} returned a non-JSON response — "
                "is OLLAMA_BASE_URL pointing at an Ollama server?"
            ) from error
        if not isinstance(data, dict):
            raise EmbeddingError(
                f"Ollama at {self.base_url} returned a non-object JSON response — "
                "is OLLAMA_BASE_URL pointing at an Ollama server?"
            )
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            received = len(embeddings) if isinstance(embeddings, list) else "no"
            raise EmbeddingError(f"Ollama returned {received} embeddings for {len(texts)} inputs")
        return embeddings

    def close(self) -> None:
        """Release pooled connections. Further embed() calls raise."""
        self._http.close()


class Embedder:
    """Embeds passages and queries with model-appropriate prompts."""

    def __init__(self, client: SupportsEmbed, model: str):
        self.client = client
        self._templates = _templates_for(model)
        self._cached_dim: int | None = None

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        template = self._templates.passage
        return self.client.embed([template.format(text=text) for text in texts])

    def embed_query(self, text: str) -> list[float]:
        [vector] = self.client.embed([self._templates.query.format(text=text)])
        return vector

    def dim(self) -> int:
        if self._cached_dim is None:
            [vector] = self.client.embed([_DIM_PROBE])
            self._cached_dim = len(vector)
        return self._cached_dim

    def close(self) -> None:
        """Close the underlying embeddings client; further calls raise."""
        self.client.close()


def create_embedder(settings: Settings) -> Embedder:
    """Build the app-lifetime embedder for the configured Ollama model.

    The returned instance owns a pooled HTTP client: construct it once at
    startup (FastAPI lifespan), share it across requests, and call
    ``close()`` on shutdown to release the connections.
    """
    client = OllamaEmbeddingsClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
    )
    return Embedder(client=client, model=settings.ollama_embed_model)


def _templates_for(model: str) -> PromptTemplates:
    base_name = model.split(":", 1)[0]
    return _TEMPLATES.get(base_name, _DEFAULT_TEMPLATES)


def _error_detail(response: httpx2.Response) -> str:
    """Distill an untrusted error response into safe exception text.

    Only Ollama's ``{"error": "..."}`` field is relayed — stripped of
    non-printable characters, then bounded — so a proxy that echoes the
    request body (which can hold chunk PII) or dumps control characters
    never reaches exception messages or logs. Any other body shape is
    summarized by its content type alone.
    """
    try:
        data = response.json()
    except ValueError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("error"), str):
        return _bounded(_printable(data["error"]))
    content_type = _bounded(_printable(response.headers.get("content-type", "unknown")), limit=60)
    return f"[{content_type} body omitted]"


def _printable(text: str) -> str:
    """Drop control and other non-printable characters (keeps spaces)."""
    return "".join(char for char in text if char.isprintable())


def _bounded(body: str, limit: int = 200) -> str:
    """Cap server-controlled text before it lands in an exception message."""
    if len(body) <= limit:
        return body
    return f"{body[:limit]}... [truncated {len(body) - limit} chars]"
