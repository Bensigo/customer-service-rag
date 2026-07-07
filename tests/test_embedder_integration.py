"""Integration tests against a live Ollama server.

Skipped entirely when Ollama is unreachable or the test model is not
pulled (CI has no Ollama — these run locally only). Uses the small
nomic-embed-text model to stay fast; qwen3-embedding remains the
documented production default.
"""

import math
import os

import httpx2
import pytest

from app.ingestion.chunker import chunk_text
from app.retrieval.embedder import Embedder, OllamaEmbeddingsClient

# Read at collection time, before the hermetic_settings fixture scrubs env.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = "nomic-embed-text"


pytestmark = pytest.mark.integration


def _model_available() -> bool:
    """Probe for a live Ollama with the test model pulled.

    Any failure at all — unreachable server, a bad OLLAMA_BASE_URL
    (httpx2.InvalidURL is not an HTTPError), an unexpected payload —
    means "unavailable": these tests then skip rather than error.
    """
    try:
        response = httpx2.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        response.raise_for_status()
        names = [entry["name"] for entry in response.json()["models"]]
    except Exception:
        return False
    return any(name == MODEL or name.startswith(f"{MODEL}:") for name in names)


@pytest.fixture(scope="module")
def embedder():
    # Probed here, not at import, so unit-only runs never touch the network.
    if not _model_available():
        pytest.skip(f"Ollama not reachable at {OLLAMA_BASE_URL} or {MODEL} not pulled")
    client = OllamaEmbeddingsClient(base_url=OLLAMA_BASE_URL, model=MODEL)
    yield Embedder(client=client, model=MODEL)
    client.close()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.hypot(*a) * math.hypot(*b))


def test_embed_passages_returns_vectors_matching_dim(embedder):
    vectors = embedder.embed_passages(["How do I reset my password?", "Shipping costs"])

    assert len(vectors) == 2
    assert all(len(vector) == embedder.dim() for vector in vectors)
    assert all(isinstance(value, float) for vector in vectors for value in vector)


def test_dim_is_positive_and_stable(embedder):
    assert embedder.dim() > 0
    assert embedder.dim() == embedder.dim()


def test_similar_texts_closer_than_dissimilar(embedder):
    query = embedder.embed_query("reset my password")
    [on_topic, off_topic] = embedder.embed_passages(
        [
            "Password reset help: click 'Forgot password' on the login page.",
            "Standard shipping costs $4.99 and takes three to five business days.",
        ]
    )

    assert _cosine(query, on_topic) > _cosine(query, off_topic)


def test_worst_case_chunker_output_fits_model_context(embedder):
    """The chunker's 350-token default must fit the embed model's context.

    The client sends truncate=false, so an over-context chunk would fail
    loudly here instead of being silently cut. nomic-embed-text reports a
    2048-token context via /api/tags (qwen3-embedding: 40960) — both leave
    ample headroom over the chunker's worst case.
    """
    text = " ".join(
        f"Support ticket {i} was resolved by checking the billing dashboard, "
        "reviewing recent invoices, and confirming the refund with the customer."
        for i in range(120)
    )
    chunks = chunk_text(text)
    worst = max(chunks, key=lambda chunk: chunk.token_estimate)
    assert worst.token_estimate >= 300  # fixture is a genuine worst case

    [vector] = embedder.embed_passages([worst.text])

    assert len(vector) == embedder.dim()
