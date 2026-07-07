"""Retrieval eval harness: golden dataset + retrieval-quality metrics
(issues #14 and #39).

Deliberately sequenced before the reranker (#15) so every retrieval
change ships against a recorded baseline. A retrieved chunk is
*relevant* to a question when its ``chunk.doc_id`` is in the question's
``relevant_doc_ids`` set (doc-level relevance). The harness reports four
metrics per run, all means over the dataset:

- **hit-rate@k** (#14): fraction of questions with any relevant chunk in
  the top-k.
- **precision@k**: relevant chunks in the top-k, divided by ``k``.
- **recall@k**: distinct relevant doc ids present in the top-k, divided
  by the number of relevant docs for the question.
- **MRR**: reciprocal of the 1-based rank of the first relevant chunk.

The metric functions (``precision_at_k``, ``recall_at_k``,
``reciprocal_rank``) are pure and I/O-free so they can be unit-tested
with hand-computed values.

Nothing here logs or stores chunk text: only ``chunk.doc_id`` (an
operator-chosen document identifier, not customer content) is read from
retrieved chunks, and reports carry question text and doc ids only.
"""

import json
from collections.abc import Iterable, Sequence
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
    """One labeled eval row: a customer-phrased question and the ids of the
    documents that genuinely answer it.

    ``relevant_doc_ids`` is a tuple (not a list) so the frozen dataclass
    stays hashable; it must be non-empty (enforced by ``load_golden``).
    A retrieved chunk is relevant to this row when its ``doc_id`` is in
    this set.
    """

    question: str
    relevant_doc_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Outcome of a retrieval eval run, all metrics averaged over the
    dataset.

    ``hit_rate_at_k`` is hits / len(dataset); ``misses`` lists the
    questions whose top-k held no relevant chunk. ``precision_at_k``,
    ``recall_at_k``, and ``mrr`` are the means of the per-question
    metrics. The precision/recall/mrr fields default to 0.0 so callers
    predating #39 that build a report from hit-rate alone still work.
    """

    hit_rate_at_k: float
    misses: list[str] = field(default_factory=list)
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0


class SupportsRetrieve(Protocol):
    """The retrieval seam the harness evaluates (HybridRetriever, #13)."""

    def retrieve(
        self, query: str, *, k_each: int = ..., top_n: int = ...
    ) -> list[RetrievedChunk]: ...


def load_golden(path: str | Path) -> list[GoldenExample]:
    """Parse a golden-dataset JSONL file into GoldenExamples.

    One JSON object per line, each with ``question`` and a non-empty
    ``relevant_doc_ids`` list. Blank lines are ignored so the file can be
    formatted for readability. A row with an empty ``relevant_doc_ids``
    is rejected with ``ValueError`` — a question with no relevant doc
    would make recall undefined and always score zero, which is a dataset
    bug, not a valid label.
    """
    examples: list[GoldenExample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        relevant_doc_ids = tuple(row["relevant_doc_ids"])
        if not relevant_doc_ids:
            raise ValueError(f"golden row has empty relevant_doc_ids: {row['question']!r}")
        examples.append(GoldenExample(question=row["question"], relevant_doc_ids=relevant_doc_ids))
    return examples


def precision_at_k(ranked: Sequence[RetrievedChunk], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-k ranked chunks that are relevant.

    The denominator is ``k`` (not the number of chunks retrieved): a
    query that returns fewer than ``k`` chunks is penalised for the empty
    slots. Returns 0.0 for an empty ranked list.
    """
    if not ranked:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for chunk in ranked[:k] if chunk.chunk.doc_id in relevant_set)
    return hits / k


def recall_at_k(ranked: Sequence[RetrievedChunk], relevant: Iterable[str], k: int) -> float:
    """Fraction of the relevant doc ids that appear in the top-k.

    Counts *distinct* relevant doc ids in the top-k (a doc surfaced by
    several chunks counts once), divided by the number of relevant docs.
    Returns 0.0 for an empty ranked list or an empty relevant set.
    """
    relevant_set = set(relevant)
    if not ranked or not relevant_set:
        return 0.0
    found = {chunk.chunk.doc_id for chunk in ranked[:k]} & relevant_set
    return len(found) / len(relevant_set)


def reciprocal_rank(ranked: Sequence[RetrievedChunk], relevant: Iterable[str]) -> float:
    """Reciprocal of the 1-based rank of the first relevant chunk.

    Returns 0.0 when no chunk in ``ranked`` is relevant (including an
    empty ranked list).
    """
    relevant_set = set(relevant)
    for rank, chunk in enumerate(ranked, start=1):
        if chunk.chunk.doc_id in relevant_set:
            return 1 / rank
    return 0.0


def run_retrieval_eval(
    retriever: SupportsRetrieve, dataset: list[GoldenExample], k: int = 5
) -> EvalReport:
    """Compute hit-rate@k, precision@k, recall@k, and MRR for
    ``retriever`` over the golden ``dataset``.

    For each example the retriever's ranked results are scored against
    the example's ``relevant_doc_ids``: a *hit* is any relevant chunk in
    the top-k; precision/recall/mrr are the pure metric functions above.
    Questions with no hit are collected into ``misses``. The report's
    precision/recall/mrr are the means over the dataset. An empty dataset
    yields all-zero metrics and no misses.
    """
    # Ask the retriever for at least k results (and at least k candidates
    # per index) so retrieved[:k] is the retriever's true top-k rather than
    # a silently-short slice when k exceeds the default top_n (12).
    top_n = max(k, _DEFAULT_TOP_N)
    k_each = max(k, _DEFAULT_K_EACH)
    hits = 0
    misses: list[str] = []
    precision_sum = 0.0
    recall_sum = 0.0
    rr_sum = 0.0
    for example in dataset:
        retrieved = retriever.retrieve(example.question, k_each=k_each, top_n=top_n)
        relevant = example.relevant_doc_ids
        top_k_doc_ids = {chunk.chunk.doc_id for chunk in retrieved[:k]}
        if top_k_doc_ids & set(relevant):
            hits += 1
        else:
            misses.append(example.question)
        precision_sum += precision_at_k(retrieved, relevant, k)
        recall_sum += recall_at_k(retrieved, relevant, k)
        rr_sum += reciprocal_rank(retrieved, relevant)
    n = len(dataset)
    if not n:
        return EvalReport(hit_rate_at_k=0.0, misses=misses)
    return EvalReport(
        hit_rate_at_k=hits / n,
        misses=misses,
        precision_at_k=precision_sum / n,
        recall_at_k=recall_sum / n,
        mrr=rr_sum / n,
    )
