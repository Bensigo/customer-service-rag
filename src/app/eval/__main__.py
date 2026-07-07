"""``python -m app.eval`` — run the retrieval eval against live services.

Wires the real retrieval stack from Settings, ingests the checked-in
sample corpus into a *throwaway* SQLite database and a per-run Qdrant
collection (never the configured production db/collection), runs the
retrieval metrics (hit-rate@k, precision@k, recall@k, MRR) over the
golden dataset at a small k sweep, prints a readable table, and cleans
up both throwaways on the way out.

The eval needs live Ollama (embeddings) and Qdrant, so it is a local
/manual command, not a CI step — the unit tests in
tests/test_retrieval_eval.py cover the metric math with a fake
retriever, and CI runs those.

Nothing here prints chunk text: the table shows the metrics and the
list of missed *questions* (from the golden dataset, operator-authored)
only.
"""

import argparse
import sqlite3
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from qdrant_client import QdrantClient

from app.config import Settings, get_settings
from app.eval.retrieval_eval import (
    _DEFAULT_K_EACH,
    _DEFAULT_TOP_N,
    EvalReport,
    GoldenExample,
    RerankingRetriever,
    load_golden,
    run_retrieval_eval,
)
from app.ingestion.pipeline import IngestionService, NoopCacheInvalidator
from app.retrieval.embedder import create_embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import create_reranker
from app.stores.bm25_index import Bm25Index
from app.stores.chunk_store import ChunkStore
from app.stores.vector_store import VectorStore

_REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLES_DIR = _REPO_ROOT / "data" / "samples"
GOLDEN_PATH = _REPO_ROOT / "data" / "eval" / "golden.jsonl"

# hit-rate@5 is saturated on the 7-doc sample corpus (#14 baseline), so the
# CLI sweeps low k where precision/recall/MRR actually discriminate.
K_SWEEP = (1, 3, 5)


def format_report(report: EvalReport, *, k: int, total: int) -> str:
    """Render an EvalReport as a readable multi-line table.

    Shows all four metrics (hit-rate@k, precision@k, recall@k, MRR) as
    percentages / a ratio, an ``N/total`` hit count, then lists any
    missed questions. Never includes chunk text — only the metrics and
    the operator-authored questions that missed.
    """
    hits = total - len(report.misses)
    lines = [
        f"Retrieval eval @k={k}",
        "=" * 40,
        f"questions    : {total}",
        f"hit-rate@{k}   : {report.hit_rate_at_k:.1%}  ({hits}/{total})",
        f"precision@{k}  : {report.precision_at_k:.1%}",
        f"recall@{k}     : {report.recall_at_k:.1%}",
        f"MRR          : {report.mrr:.3f}",
    ]
    if report.misses:
        lines.append("")
        lines.append(f"misses ({len(report.misses)}):")
        lines.extend(f"  - {question}" for question in report.misses)
    else:
        lines.append("")
        lines.append("no misses — every question hit")
    return "\n".join(lines)


def _build_retriever(
    settings: Settings, db_path: str, collection: str
) -> tuple[HybridRetriever, IngestionService, list[tuple[str, Callable[[], None]]]]:
    """Build the retrieval stack on a throwaway db + collection.

    Returns the retriever, the ingestion service, and an ordered list of
    (name, close) closers to run LIFO on teardown. On any failure during
    construction the already-opened resources are closed before the error
    propagates.
    """
    closers: list[tuple[str, Callable[[], None]]] = []
    try:
        chunk_store = ChunkStore(db_path)
        closers.append(("chunk store", chunk_store.close))
        # second connection to the same WAL database for the BM25 index,
        # as the ingestion pipeline expects (store and index co-located).
        bm25_conn = sqlite3.connect(db_path)
        closers.append(("bm25 connection", bm25_conn.close))
        bm25_index = Bm25Index(bm25_conn)

        embedder = create_embedder(settings)
        closers.append(("embedder", embedder.close))
        vector_size = embedder.dim()  # live Ollama round-trip

        vector_store = VectorStore(
            settings.qdrant_url, vector_size=vector_size, collection=collection
        )
        closers.append(("vector store", vector_store.close))
        vector_store.ensure_collection()
        # Register the throwaway-collection drop the moment it exists, so it
        # is torn down even if a later build step raises (the drop runs
        # before vector_store.close since teardown is LIFO).
        closers.append(
            ("eval collection", lambda: _drop_collection(settings.qdrant_url, collection))
        )
    except Exception:
        _close_all(list(reversed(closers)))
        raise

    service = IngestionService(
        chunk_store, bm25_index, embedder, vector_store, NoopCacheInvalidator()
    )
    retriever = HybridRetriever(bm25_index, embedder, vector_store, chunk_store)
    return retriever, service, list(reversed(closers))


def _close_all(closers: list[tuple[str, Callable[[], None]]]) -> None:
    """Run every close in order; one failure never skips the rest."""
    for name, close in closers:
        try:
            close()
        except Exception as error:  # noqa: BLE001 — teardown is best-effort
            print(f"warning: failed to close {name}: {type(error).__name__}", file=sys.stderr)


def _ingest_samples(service: IngestionService) -> int:
    """Ingest every data/samples/*.md, keyed by filename stem. Returns the count."""
    sample_files = sorted(SAMPLES_DIR.glob("*.md"))
    for path in sample_files:
        doc_id = path.stem
        text = path.read_text(encoding="utf-8")
        # title = doc_id keeps the corpus self-describing without inventing
        # display names; the eval only reads doc_id anyway.
        service.ingest(doc_id, doc_id, text)
    return len(sample_files)


def _mean_query_latency(retriever, dataset: list[GoldenExample]) -> float:
    """Mean wall-clock seconds per ``retrieve`` call over the dataset.

    Used to report the reranker's added per-query latency (with minus
    without). Times only the retrieve call, not metric math. Returns 0.0
    for an empty dataset.
    """
    if not dataset:
        return 0.0
    total = 0.0
    for example in dataset:
        start = time.perf_counter()
        retriever.retrieve(example.question, k_each=_DEFAULT_K_EACH, top_n=_DEFAULT_TOP_N)
        total += time.perf_counter() - start
    return total / len(dataset)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.eval", description=__doc__)
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="also run the LLM reranker (Ollama) and print a with/without comparison "
        "including added per-query latency",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    dataset = load_golden(GOLDEN_PATH)

    # Throwaway db + collection so the eval never touches production data.
    collection = f"eval_run_{uuid.uuid4().hex[:12]}"
    with tempfile.TemporaryDirectory(prefix="rag-eval-") as tmp_dir:
        db_path = str(Path(tmp_dir) / "eval.sqlite3")
        retriever, service, closers = _build_retriever(settings, db_path, collection)
        reranker = None
        try:
            corpus_size = _ingest_samples(service)
            base_reports = {k: run_retrieval_eval(retriever, dataset, k=k) for k in K_SWEEP}
            base_latency = _mean_query_latency(retriever, dataset)

            rerank_reports: dict[int, EvalReport] | None = None
            rerank_latency = 0.0
            if args.rerank:
                reranker = create_reranker(settings)
                reranked = RerankingRetriever(retriever, reranker)
                rerank_reports = {k: run_retrieval_eval(reranked, dataset, k=k) for k in K_SWEEP}
                rerank_latency = _mean_query_latency(reranked, dataset)
        finally:
            if reranker is not None:
                reranker.close()
            # closers already include dropping the throwaway collection (LIFO).
            _close_all(closers)

    print(f"corpus: {corpus_size} docs from {SAMPLES_DIR}")
    if rerank_reports is None:
        for k in K_SWEEP:
            print()
            print(format_report(base_reports[k], k=k, total=len(dataset)))
        return 0

    print(
        f"\nmean retrieve latency: baseline {base_latency:.3f}s/query, "
        f"with rerank {rerank_latency:.3f}s/query "
        f"(+{rerank_latency - base_latency:.3f}s/query)"
    )
    for k in K_SWEEP:
        print()
        print(f"--- k={k}: WITHOUT rerank ---")
        print(format_report(base_reports[k], k=k, total=len(dataset)))
        print(f"\n--- k={k}: WITH rerank ---")
        print(format_report(rerank_reports[k], k=k, total=len(dataset)))
    return 0


def _drop_collection(qdrant_url: str, collection: str) -> None:
    """Delete the throwaway Qdrant collection; best-effort on teardown."""
    client = QdrantClient(location=qdrant_url)
    try:
        client.delete_collection(collection)
    except Exception as error:  # noqa: BLE001 — teardown is best-effort
        print(
            f"warning: failed to drop eval collection {collection}: {type(error).__name__}",
            file=sys.stderr,
        )
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
