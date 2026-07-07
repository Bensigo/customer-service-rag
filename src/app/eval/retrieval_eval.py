"""Retrieval eval harness: golden dataset + hit-rate@k baseline (issue #14).

Deliberately sequenced before the reranker (#15) so every retrieval
change ships against a recorded baseline. The metric here is
hit-rate@k: a question is a *hit* when any of the retriever's top-k
chunks belongs to the question's ``expected_doc_id``; hit-rate@k is the
fraction of questions that hit.

The harness is intentionally metric-light. ``EvalReport`` carries only
the baseline fields today (``hit_rate_at_k`` and the miss list) but is a
plain dataclass so #39 can add ``precision_at_k``/``recall_at_k``/``mrr``
as new fields without breaking existing callers, which read the current
two by name.

Nothing here logs or stores chunk text: only ``chunk.doc_id`` (an
operator-chosen document identifier, not customer content) is read from
retrieved chunks, and reports carry question text and doc ids only.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.models import RetrievedChunk

# HybridRetriever's own defaults (k_each=20, top_n=12). The eval floors its
# requests at these so a small k behaves exactly like a normal retrieve,
# while a large k still gets enough candidates to measure.
_DEFAULT_K_EACH = 20
_DEFAULT_TOP_N = 12


@dataclass(frozen=True, slots=True)
class GoldenExample:
    """One labeled eval row: a customer-phrased question and the id of the
    document that should answer it."""

    question: str
    expected_doc_id: str


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Outcome of a retrieval eval run.

    ``hit_rate_at_k`` is hits / len(dataset); ``misses`` lists the
    questions whose top-k held no chunk from the expected document. Kept
    a plain dataclass so #39 can add precision@k/recall@k/mrr fields
    without breaking callers that read these two by name.
    """

    hit_rate_at_k: float
    misses: list[str] = field(default_factory=list)


class SupportsRetrieve(Protocol):
    """The retrieval seam the harness evaluates (HybridRetriever, #13)."""

    def retrieve(
        self, query: str, *, k_each: int = ..., top_n: int = ...
    ) -> list[RetrievedChunk]: ...


def load_golden(path: str | Path) -> list[GoldenExample]:
    """Parse a golden-dataset JSONL file into GoldenExamples.

    One JSON object per line, each with ``question`` and
    ``expected_doc_id``. Blank lines are ignored so the file can be
    formatted for readability.
    """
    examples: list[GoldenExample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        examples.append(
            GoldenExample(question=row["question"], expected_doc_id=row["expected_doc_id"])
        )
    return examples


def run_retrieval_eval(
    retriever: SupportsRetrieve, dataset: list[GoldenExample], k: int = 5
) -> EvalReport:
    """Compute hit-rate@k for ``retriever`` over the golden ``dataset``.

    For each example the retriever's top-k chunks are inspected; the
    example is a hit when any of them belongs to ``expected_doc_id``.
    Questions with no hit are collected into ``misses``. An empty dataset
    yields a hit rate of 0.0 (no questions answered) and no misses.
    """
    # Ask the retriever for at least k results (and at least k candidates
    # per index) so retrieved[:k] is the retriever's true top-k rather than
    # a silently-short slice when k exceeds the default top_n (12).
    top_n = max(k, _DEFAULT_TOP_N)
    k_each = max(k, _DEFAULT_K_EACH)
    hits = 0
    misses: list[str] = []
    for example in dataset:
        retrieved = retriever.retrieve(example.question, k_each=k_each, top_n=top_n)
        top_k_doc_ids = {chunk.chunk.doc_id for chunk in retrieved[:k]}
        if example.expected_doc_id in top_k_doc_ids:
            hits += 1
        else:
            misses.append(example.question)
    hit_rate = hits / len(dataset) if dataset else 0.0
    return EvalReport(hit_rate_at_k=hit_rate, misses=misses)
