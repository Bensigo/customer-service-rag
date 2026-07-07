# customer-service-rag

## Retrieval eval

A small retrieval eval harness measures hit-rate@k over a golden set of
customer-phrased questions, so every retrieval change (e.g. the reranker
in #15) ships against a recorded baseline.

- `data/samples/*.md` — a fictional SaaS product's support docs (password
  reset, billing, shipping, returns, account deletion, API keys, contact
  support). This is the eval corpus and doubles as a demo corpus.
- `data/eval/golden.jsonl` — labeled `{question, expected_doc_id}` rows
  phrased like real customers (short, paraphrased, some misspelled).

Run it against a live stack (Ollama for embeddings, Qdrant for vectors):

```
make eval
```

`make eval` ingests every sample doc into a throwaway SQLite database and
a per-run Qdrant collection (never your configured production db or
collection), runs the eval, prints hit-rate@k plus any missed questions,
and cleans both throwaways up. The metric math (`run_retrieval_eval`,
`load_golden`) is unit-tested with a fake retriever in CI; the
end-to-end run stays a local command because it needs live services.
