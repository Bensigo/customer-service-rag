"""Specs for the retrieval eval harness + golden dataset (issue #14).

The hit-rate math is exercised against a fake retriever with hand-planted
rankings, so the numbers are deterministic and exactly assertable and CI
needs neither Ollama nor Qdrant. One end-to-end test ingests a small
subset of the real sample docs into a throwaway SQLite + a per-run Qdrant
collection and runs the eval through the live embedder; it is marked
``integration`` and skips when Ollama or Qdrant is unreachable.

The golden dataset is validated here too: every ``expected_doc_id`` must
name a real ``data/samples/*.md`` file, so a typo in the dataset is a
failing test rather than a silent always-miss at eval time.
"""

import os
import sqlite3
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import httpx2
import pytest
from qdrant_client import QdrantClient

from app.eval.__main__ import format_report
from app.eval.retrieval_eval import EvalReport, GoldenExample, load_golden, run_retrieval_eval
from app.ingestion.pipeline import IngestionService, NoopCacheInvalidator
from app.models import Chunk, RetrievedChunk
from app.retrieval.embedder import Embedder, OllamaEmbeddingsClient
from app.retrieval.hybrid import HybridRetriever
from app.stores.bm25_index import Bm25Index
from app.stores.chunk_store import ChunkStore
from app.stores.vector_store import VectorStore

_REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = _REPO_ROOT / "data" / "samples"
GOLDEN_PATH = _REPO_ROOT / "data" / "eval" / "golden.jsonl"


def _chunk(doc_id: str) -> Chunk:
    """A minimal chunk belonging to ``doc_id`` (only doc_id is read by the harness)."""
    return Chunk(id=f"{doc_id}:1:0", doc_id=doc_id, version=1, seq=0, text="", title=doc_id)


def _retrieved(doc_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk=_chunk(doc_id), score=score, sources={"bm25"})


class FakeRetriever:
    """Returns a pre-planted ranked list per question, recording the
    top_n/k_each it was asked for so tests can assert the harness requests
    enough candidates to measure the top-k it reports."""

    def __init__(self, results: dict[str, list[RetrievedChunk]]) -> None:
        self._results = results
        self.calls: list[tuple[int, int]] = []  # (k_each, top_n)

    def retrieve(self, query: str, *, k_each: int = 20, top_n: int = 12) -> list[RetrievedChunk]:
        self.calls.append((k_each, top_n))
        return self._results[query]


class TestHitRate:
    def test_hits_two_of_three_questions(self):
        # q1 and q3 have the expected doc in their results; q2 does not.
        dataset = [
            GoldenExample(question="q1", expected_doc_id="alpha"),
            GoldenExample(question="q2", expected_doc_id="beta"),
            GoldenExample(question="q3", expected_doc_id="gamma"),
        ]
        retriever = FakeRetriever(
            {
                "q1": [_retrieved("alpha", 0.9), _retrieved("other", 0.8)],
                "q2": [_retrieved("wrong", 0.9), _retrieved("nope", 0.8)],
                "q3": [_retrieved("gamma", 0.7)],
            }
        )

        report = run_retrieval_eval(retriever, dataset, k=5)

        assert report.hit_rate_at_k == 2 / 3
        assert report.misses == ["q2"]

    def test_k_cutoff_excludes_relevant_chunk_past_rank_k(self):
        # The expected doc sits at rank k+1, so it must NOT count as a hit at k.
        k = 3
        ranked = [_retrieved(f"filler{i}", 1.0 - i * 0.01) for i in range(k)]
        ranked.append(_retrieved("target", 0.1))  # rank k+1
        dataset = [GoldenExample(question="q", expected_doc_id="target")]
        retriever = FakeRetriever({"q": ranked})

        report = run_retrieval_eval(retriever, dataset, k=k)

        assert report.hit_rate_at_k == 0.0
        assert report.misses == ["q"]

    def test_relevant_chunk_at_rank_k_is_a_hit(self):
        # Boundary: the expected doc at exactly rank k still counts.
        k = 3
        ranked = [_retrieved(f"filler{i}", 1.0 - i * 0.01) for i in range(k - 1)]
        ranked.append(_retrieved("target", 0.1))  # rank k
        dataset = [GoldenExample(question="q", expected_doc_id="target")]
        retriever = FakeRetriever({"q": ranked})

        report = run_retrieval_eval(retriever, dataset, k=k)

        assert report.hit_rate_at_k == 1.0
        assert report.misses == []

    def test_all_hits_reports_full_rate_and_no_misses(self):
        dataset = [GoldenExample(question="q1", expected_doc_id="alpha")]
        retriever = FakeRetriever({"q1": [_retrieved("alpha", 0.9)]})

        report = run_retrieval_eval(retriever, dataset, k=5)

        assert report.hit_rate_at_k == 1.0
        assert report.misses == []

    def test_report_is_shaped_for_future_metrics(self):
        # #39 adds precision@k/recall@k/mrr; the report must expose the two
        # baseline fields today without those extras being required.
        report = EvalReport(hit_rate_at_k=0.5, misses=["q2"])

        assert report.hit_rate_at_k == 0.5
        assert report.misses == ["q2"]

    def test_requests_at_least_k_candidates_from_the_retriever(self):
        # The default HybridRetriever returns top_n=12; measuring hit-rate
        # at a k above that would silently truncate. The harness must ask
        # the retriever for at least k candidates so retrieved[:k] is real.
        k = 25
        retriever = FakeRetriever({"q": [_retrieved("alpha", 0.9)]})
        dataset = [GoldenExample(question="q", expected_doc_id="alpha")]

        run_retrieval_eval(retriever, dataset, k=k)

        (k_each, top_n) = retriever.calls[0]
        assert top_n >= k, "retriever asked for fewer than k results"
        assert k_each >= k, "each index asked for fewer than k candidates"


class TestFormatReport:
    """The CLI's table renderer: readable, and never leaks chunk text."""

    def test_shows_hit_rate_k_and_total(self):
        report = EvalReport(hit_rate_at_k=0.75, misses=["q2"])

        rendered = format_report(report, k=5, total=4)

        assert "hit-rate@5" in rendered
        assert "75" in rendered  # 0.75 rendered as a percentage
        assert "3/4" in rendered  # 3 of 4 questions hit

    def test_lists_misses(self):
        report = EvalReport(hit_rate_at_k=0.0, misses=["why won't it work", "help"])

        rendered = format_report(report, k=5, total=2)

        assert "why won't it work" in rendered
        assert "help" in rendered

    def test_no_misses_reads_cleanly(self):
        report = EvalReport(hit_rate_at_k=1.0, misses=[])

        rendered = format_report(report, k=3, total=2)

        assert "hit-rate@3" in rendered
        assert "2/2" in rendered


class TestLoadGolden:
    def test_parses_jsonl_into_golden_examples(self, tmp_path):
        path = tmp_path / "g.jsonl"
        path.write_text(
            '{"question": "how do i reset", "expected_doc_id": "password-reset"}\n'
            '{"question": "where are invoices", "expected_doc_id": "billing"}\n',
            encoding="utf-8",
        )

        dataset = load_golden(path)

        assert dataset == [
            GoldenExample(question="how do i reset", expected_doc_id="password-reset"),
            GoldenExample(question="where are invoices", expected_doc_id="billing"),
        ]

    def test_ignores_blank_lines(self, tmp_path):
        path = tmp_path / "g.jsonl"
        path.write_text(
            '{"question": "q1", "expected_doc_id": "d1"}\n'
            "\n"
            "   \n"
            '{"question": "q2", "expected_doc_id": "d2"}\n',
            encoding="utf-8",
        )

        dataset = load_golden(path)

        assert [ex.question for ex in dataset] == ["q1", "q2"]


class TestGoldenDatasetIntegrity:
    """Schema + referential integrity of the checked-in golden dataset."""

    def test_golden_dataset_loads_and_references_existing_sample_docs(self):
        dataset = load_golden(GOLDEN_PATH)
        sample_stems = {p.stem for p in SAMPLES_DIR.glob("*.md")}

        assert 15 <= len(dataset) <= 20, "golden set should hold 15-20 rows"
        assert sample_stems, "no sample docs found"
        for example in dataset:
            assert example.question.strip(), "every question must be non-empty"
            assert example.expected_doc_id in sample_stems, (
                f"golden expected_doc_id {example.expected_doc_id!r} has no "
                f"matching data/samples/*.md file"
            )

    def test_questions_are_not_verbatim_sample_sentences(self):
        # A trivially-easy eval (questions copied from the docs) is
        # meaningless; guard that no question is a substring of its doc.
        dataset = load_golden(GOLDEN_PATH)
        for example in dataset:
            doc_text = (SAMPLES_DIR / f"{example.expected_doc_id}.md").read_text(encoding="utf-8")
            assert example.question.lower() not in doc_text.lower(), (
                f"question {example.question!r} is copied verbatim from its doc"
            )


# --- Integration: end-to-end eval on a small subset against live services ---

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
MODEL = "nomic-embed-text"

# A 3-doc subset with hand-written questions whose expected doc is obvious.
SUBSET_QUESTIONS = [
    GoldenExample(
        question="i cant log in and forgot my password", expected_doc_id="password-reset"
    ),
    GoldenExample(question="how do i change my billing card", expected_doc_id="billing"),
    GoldenExample(question="when does my package arrive", expected_doc_id="shipping"),
]
SUBSET_DOC_IDS = [ex.expected_doc_id for ex in SUBSET_QUESTIONS]


def _safe_url(url: str) -> str:
    """scheme://host:port only, dropping any userinfo so credentials embedded
    in OLLAMA_BASE_URL/QDRANT_URL never reach a skip message or CI log."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{host}{port}" if host else "(redacted)"


def _ollama_model_available() -> bool:
    try:
        response = httpx2.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        response.raise_for_status()
        names = [entry["name"] for entry in response.json()["models"]]
    except Exception:
        return False
    return any(name == MODEL or name.startswith(f"{MODEL}:") for name in names)


def _qdrant_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{QDRANT_URL}/readyz", timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture
def retriever_rig(tmp_path, request):
    # Probed here, not at import, so unit-only runs never touch the network.
    if not _ollama_model_available():
        pytest.skip(f"Ollama not reachable at {_safe_url(OLLAMA_BASE_URL)} or {MODEL} not pulled")
    if not _qdrant_reachable():
        pytest.skip(f"Qdrant not reachable at {_safe_url(QDRANT_URL)}")

    db_path = str(tmp_path / "eval.sqlite3")
    chunk_store = ChunkStore(db_path)
    request.addfinalizer(chunk_store.close)
    conn = sqlite3.connect(db_path)
    request.addfinalizer(conn.close)
    bm25 = Bm25Index(conn)
    client = OllamaEmbeddingsClient(base_url=OLLAMA_BASE_URL, model=MODEL)
    embedder = Embedder(client=client, model=MODEL)
    request.addfinalizer(embedder.close)
    vector_size = embedder.dim()
    collection = f"test14_eval_{uuid.uuid4().hex[:12]}"
    vector_store = VectorStore(QDRANT_URL, vector_size=vector_size, collection=collection)

    def teardown_vector_store() -> None:
        vector_store.close()
        qdrant = QdrantClient(url=QDRANT_URL)
        try:
            qdrant.delete_collection(collection)
        finally:
            qdrant.close()

    request.addfinalizer(teardown_vector_store)
    vector_store.ensure_collection()

    service = IngestionService(chunk_store, bm25, embedder, vector_store, NoopCacheInvalidator())
    for doc_id in SUBSET_DOC_IDS:
        text = (SAMPLES_DIR / f"{doc_id}.md").read_text(encoding="utf-8")
        service.ingest(doc_id, doc_id, text)

    return HybridRetriever(bm25, embedder, vector_store, chunk_store)


@pytest.mark.integration
def test_eval_runs_end_to_end_on_fixture_corpus(retriever_rig):
    report = run_retrieval_eval(retriever_rig, SUBSET_QUESTIONS, k=3)

    assert isinstance(report, EvalReport)
    assert 0.0 < report.hit_rate_at_k <= 1.0, "hybrid retrieval should hit at least one question"
    assert len(report.misses) < len(SUBSET_QUESTIONS)
