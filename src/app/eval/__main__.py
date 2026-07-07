"""``python -m app.eval`` — run the retrieval eval against live services.

Wires the real retrieval stack from Settings, ingests the checked-in
sample corpus into a *throwaway* SQLite database and a per-run Qdrant
collection (never the configured production db/collection), runs
hit-rate@k over the golden dataset, prints a readable table, and cleans
up both throwaways on the way out.

The eval needs live Ollama (embeddings) and Qdrant, so it is a local
/manual command, not a CI step — the unit tests in
tests/test_retrieval_eval.py cover the metric math with a fake
retriever, and CI runs those.

Nothing here prints chunk text: the table shows the hit rate and the
list of missed *questions* (from the golden dataset, operator-authored)
only.
"""

import sqlite3
import sys
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path

from qdrant_client import QdrantClient

from app.config import Settings, get_settings
from app.eval.retrieval_eval import EvalReport, load_golden, run_retrieval_eval
from app.ingestion.pipeline import IngestionService, NoopCacheInvalidator
from app.retrieval.embedder import create_embedder
from app.retrieval.hybrid import HybridRetriever
from app.stores.bm25_index import Bm25Index
from app.stores.chunk_store import ChunkStore
from app.stores.vector_store import VectorStore

_REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLES_DIR = _REPO_ROOT / "data" / "samples"
GOLDEN_PATH = _REPO_ROOT / "data" / "eval" / "golden.jsonl"

DEFAULT_K = 5


def format_report(report: EvalReport, *, k: int, total: int) -> str:
    """Render an EvalReport as a readable multi-line table.

    Shows hit-rate@k as a percentage and an ``N/total`` count, then lists
    any missed questions. Never includes chunk text — only the hit rate
    and the operator-authored questions that missed.
    """
    hits = total - len(report.misses)
    lines = [
        "Retrieval eval — hit-rate@k baseline",
        "=" * 40,
        f"questions   : {total}",
        f"hit-rate@{k}  : {report.hit_rate_at_k:.1%}  ({hits}/{total})",
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


def main() -> int:
    settings = get_settings()
    dataset = load_golden(GOLDEN_PATH)

    # Throwaway db + collection so the eval never touches production data.
    collection = f"eval_run_{uuid.uuid4().hex[:12]}"
    with tempfile.TemporaryDirectory(prefix="rag-eval-") as tmp_dir:
        db_path = str(Path(tmp_dir) / "eval.sqlite3")
        retriever, service, closers = _build_retriever(settings, db_path, collection)
        try:
            corpus_size = _ingest_samples(service)
            report = run_retrieval_eval(retriever, dataset, k=DEFAULT_K)
        finally:
            # closers already include dropping the throwaway collection (LIFO).
            _close_all(closers)

    print(f"corpus: {corpus_size} docs from {SAMPLES_DIR}")
    print(format_report(report, k=DEFAULT_K, total=len(dataset)))
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
