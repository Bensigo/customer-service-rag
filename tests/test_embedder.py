"""Unit tests for the Ollama embedder. No network: the Embedder logic is
exercised against an in-memory fake client, and the HTTP client against
an httpx2 MockTransport.
"""

import hashlib
import json

import httpx2
import pytest

from app.config import Settings
from app.retrieval.embedder import (
    Embedder,
    EmbeddingError,
    OllamaEmbeddingsClient,
    create_embedder,
)

QWEN3_QUERY = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query:reset my password"
)


class FakeEmbeddingsClient:
    """Deterministic stand-in for OllamaEmbeddingsClient.

    Vectors are derived from the input text, so different inputs always
    produce different vectors. Every call is recorded for shape asserts.
    """

    def __init__(self, dim: int = 8):
        self.vector_dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [byte / 255 for byte in digest[: self.vector_dim]]


def make_embedder(
    model: str = "qwen3-embedding", dim: int = 8
) -> tuple[Embedder, FakeEmbeddingsClient]:
    client = FakeEmbeddingsClient(dim=dim)
    return Embedder(client=client, model=model), client


# --- prefixing -------------------------------------------------------------


def test_nomic_passages_get_search_document_prefix():
    embedder, client = make_embedder(model="nomic-embed-text")

    embedder.embed_passages(["reset my password", "shipping costs"])

    assert client.calls == [
        ["search_document: reset my password", "search_document: shipping costs"]
    ]


def test_nomic_query_gets_search_query_prefix():
    embedder, client = make_embedder(model="nomic-embed-text")

    embedder.embed_query("reset my password")

    assert client.calls == [["search_query: reset my password"]]


def test_qwen3_passages_are_sent_verbatim():
    embedder, client = make_embedder(model="qwen3-embedding")

    embedder.embed_passages(["reset my password"])

    assert client.calls == [["reset my password"]]


def test_qwen3_query_gets_instruction_template():
    embedder, client = make_embedder(model="qwen3-embedding")

    embedder.embed_query("reset my password")

    assert client.calls == [[QWEN3_QUERY]]


def test_model_tag_suffix_is_ignored_for_template_lookup():
    embedder, client = make_embedder(model="nomic-embed-text:latest")

    embedder.embed_query("reset my password")

    assert client.calls == [["search_query: reset my password"]]


def test_unknown_model_falls_back_to_default_prefixes():
    embedder, client = make_embedder(model="mystery-embed")

    embedder.embed_passages(["reset my password"])
    embedder.embed_query("reset my password")

    assert client.calls[0] != client.calls[1]


@pytest.mark.parametrize("model", ["qwen3-embedding", "nomic-embed-text", "mystery-embed"])
def test_embed_query_differs_from_passage_embedding_of_same_text(model):
    embedder, _ = make_embedder(model=model)
    text = "reset my password"

    [passage_vector] = embedder.embed_passages([text])
    query_vector = embedder.embed_query(text)

    assert query_vector != passage_vector


# --- batching --------------------------------------------------------------


def test_embed_passages_returns_one_vector_per_text_in_order():
    embedder, client = make_embedder(dim=4)
    texts = ["a", "b", "c"]

    vectors = embedder.embed_passages(texts)

    assert len(vectors) == 3
    assert all(len(vector) == 4 for vector in vectors)
    assert vectors == client.embed(texts)  # same inputs -> same deterministic vectors


def test_embed_passages_uses_a_single_client_call():
    embedder, client = make_embedder()

    embedder.embed_passages(["a", "b", "c"])

    assert len(client.calls) == 1


def test_embed_passages_empty_list_returns_empty_without_client_call():
    embedder, client = make_embedder()

    assert embedder.embed_passages([]) == []
    assert client.calls == []


# --- dim discovery ---------------------------------------------------------


def test_dim_is_probe_vector_length():
    embedder, _ = make_embedder(dim=12)

    assert embedder.dim() == 12


def test_dim_is_cached_after_one_probe_call():
    embedder, client = make_embedder()

    first = embedder.dim()
    second = embedder.dim()

    assert first == second
    assert len(client.calls) == 1


# --- HTTP client -----------------------------------------------------------


def make_http_client(handler) -> OllamaEmbeddingsClient:
    return OllamaEmbeddingsClient(
        base_url="http://ollama.test:11434",
        model="nomic-embed-text",
        transport=httpx2.MockTransport(handler),
    )


def test_http_client_posts_model_input_and_truncate_false():
    seen: dict = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.read())
        return httpx2.Response(200, json={"embeddings": [[0.1], [0.2]]})

    client = make_http_client(handler)

    vectors = client.embed(["first", "second"])

    assert seen["path"] == "/api/embed"
    assert seen["body"] == {
        "model": "nomic-embed-text",
        "input": ["first", "second"],
        "truncate": False,
    }
    assert vectors == [[0.1], [0.2]]


def test_http_client_raises_embedding_error_on_http_error_status():
    def handler(request):
        return httpx2.Response(404, json={"error": "model not found"})

    client = make_http_client(handler)

    with pytest.raises(EmbeddingError, match="model not found"):
        client.embed(["text"])


def test_http_client_raises_embedding_error_when_unreachable():
    def handler(request):
        raise httpx2.ConnectError("connection refused")

    client = make_http_client(handler)

    with pytest.raises(EmbeddingError, match="ollama.test"):
        client.embed(["text"])


def test_http_client_raises_embedding_error_on_count_mismatch():
    def handler(request):
        return httpx2.Response(200, json={"embeddings": [[0.1]]})

    client = make_http_client(handler)

    with pytest.raises(EmbeddingError, match="2"):
        client.embed(["first", "second"])


def test_http_client_raises_embedding_error_on_non_json_success_body():
    """A wrong OLLAMA_BASE_URL can hit a service that answers 200 with HTML."""

    def handler(request):
        return httpx2.Response(200, text="<html>not ollama</html>")

    client = make_http_client(handler)

    with pytest.raises(EmbeddingError, match="non-JSON"):
        client.embed(["text"])


def test_http_client_bounds_error_body_size():
    """Server-controlled error bodies must not flood the exception message."""

    def handler(request):
        return httpx2.Response(502, text="e" * 5000)

    client = make_http_client(handler)

    with pytest.raises(EmbeddingError) as excinfo:
        client.embed(["text"])

    assert len(str(excinfo.value)) < 500


def test_http_client_close_releases_the_connection():
    def handler(request):
        return httpx2.Response(200, json={"embeddings": [[0.1]]})

    client = make_http_client(handler)
    client.embed(["text"])

    client.close()

    with pytest.raises(RuntimeError, match="closed"):
        client.embed(["text"])


def test_http_client_error_messages_never_contain_input_text():
    """Chunk text can hold customer PII — transport errors must not echo it."""

    def handler(request):
        return httpx2.Response(500, json={"error": "boom"})

    client = make_http_client(handler)

    with pytest.raises(EmbeddingError) as excinfo:
        client.embed(["ssn 000-11-2222"])

    assert "000-11-2222" not in str(excinfo.value)


# --- factory ---------------------------------------------------------------


def test_create_embedder_wires_settings():
    settings = Settings(
        anthropic_api_key="test-key",
        ollama_base_url="http://ollama.internal:11434",
        ollama_embed_model="nomic-embed-text",
    )

    embedder = create_embedder(settings)

    assert isinstance(embedder.client, OllamaEmbeddingsClient)
    assert embedder.client.base_url == "http://ollama.internal:11434"
    assert embedder.client.model == "nomic-embed-text"
