# Decision log

Short, dated rationale for the design choices worth remembering. Each
entry links the issue or PR where the decision was made. Trade-offs are
recorded honestly, including the ones a reviewer disagreed with.

## Orchestration and generation

### LangChain kept as a thin adapter behind `LLMClient`
LangChain is a project requirement, so it stays — but confined to a
one-method seam. `LLMClient.complete(messages) -> str` is the only thing
the chat endpoint knows about generation; `LangChainOllamaClient` and
`LangChainAnthropicClient` are the only classes that import LangChain.
The planning reviewer preferred dropping LangChain for the bare provider
SDKs. Recorded trade-off: we accept LangChain's dependency weight to meet
the requirement, and cap the blast radius by keeping every LangChain call
inside `src/app/chat/llm.py`. Swapping to a bare SDK later means rewriting
one file, not the pipeline. (#18, plan provenance in #1)

### Ollama pivot — a zero-key local demo
Embeddings, generation, and reranking all run on a local Ollama on the
host (`gemma4` generation, `qwen3-embedding` embeddings). The demo needs
**no API keys** and no model weights baked into the app image — a reader
can `docker compose up` and get a working RAG stack. This superseded the
original sentence-transformers / bge-small / `langchain-anthropic`-default
design. (Design pivot 2026-07-07 in #1; carried through #9, #10, #15, #18)

### Cloud LLM is a documented bring-your-own-key path, not the default
Setting `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` swaps generation to
Claude via the same `LLMClient` Protocol (`LangChainAnthropicClient`). It
is opt-in so the default demo stays keyless; the seam makes the swap a
config change, not a code change. (#18, #22)

### No sampling params with `claude-sonnet-5`
The Anthropic adapter deliberately sets no `temperature`/`top_p`/`top_k`.
`claude-sonnet-5` returns HTTP 400 on non-default sampling params, so the
reflexive `temperature=0` RAG habit breaks here. Leaving them unset (None)
means LangChain never sends them; we rely on `max_tokens` + timeout +
bounded retry only. (#18)

## Embeddings and retrieval

### `qwen3-embedding` default, `nomic-embed-text` for CI
`qwen3-embedding` (4096-dim) is the quality default; `nomic-embed-text`
(768-dim) is the small/fast alternative, env-swappable via
`OLLAMA_EMBED_MODEL`. Dimension is never hardcoded — `Embedder.dim()`
probes the live model once and Qdrant bootstraps from it — so the swap is
a single env var. CI keeps the lighter model in mind; the embedder pins
per-model query/passage prompt templates so a query embedding always
differs from a passage embedding of the same text. (#9)

### Direct httpx client for Ollama embeddings, not `langchain-ollama`
The embedder and reranker each talk to one Ollama JSON endpoint over the
already-vetted `httpx2`, not `OllamaEmbeddings`. One endpoint does not
justify the extra dependency, and the direct call lets us send
`truncate: false` so over-context input fails loudly instead of being
silently cut — a flag `langchain-ollama` does not expose. LangChain stays
only where it is required (generation). (#9, #15)

### Hybrid BM25 + vector with RRF fusion
Retrieval fuses BM25 (SQLite FTS5) and vector (Qdrant) results with
Reciprocal Rank Fusion (`RRF_K = 60`, Cormack et al. 2009). Fusing on
ranks rather than raw scores sidesteps the incomparable scales of BM25
(positive, corpus-relative) and cosine similarity (`[-1, 1]`). Agreement
between the two indexes beats any single first place. (#13)

### LLM pointwise reranker over a cross-encoder — quality vs latency
Reranking is **pointwise LLM scoring** via Ollama (`gemma4` asked for a
0–10 relevance score per candidate), not a sentence-transformers
cross-encoder — consistent with the Ollama pivot (no `sentence-transformers`
dependency anywhere). Measured against #14's pre-rerank baseline on an
8-question subset, gemma4 reranking **improves ranking quality**:
precision@1 0.625 → 1.000 and MRR 0.771 → 1.000 (it lifts the relevant
chunk to rank 1, which the already-saturated hit-rate@5 cannot show). The
cost is **~seconds per candidate** of gemma4 latency (a live `/chat`
request in this run spent ~10s in rerank vs ~0.4s in retrieval). Decision:
ship the reranker **enabled** for quality; latency-sensitive deployments
should gate it behind a flag or a smaller model. `make eval-rerank`
prints the full metric + latency table on demand. (#15)

### bge/query-prefix requirement generalized to per-model prompt templates
Embedding models need distinct query vs passage prompts (bge's original
constraint); the embedder pins a template per model rather than hardcoding
one, so a query embedding never collides with a passage embedding of the
same text. Unknown models fall back to a generic `search_document:` /
`search_query:` pair. (#9)

## Storage

### SQLite single-writer accepted for demo scale
The chunk store and BM25 index share one SQLite (WAL) database. SQLite's
single-writer model is fine for a demo: one ingest worker at a time, no
Elasticsearch, no extra infra. Horizontal scale is an explicit non-goal;
production would move to a service that supports concurrent writers. (#6,
#7, non-goals in #1)

### `check_same_thread=False` + a `RetrieverPool` for concurrent reads
Chat reads run off the event loop on a threadpool, so each retriever slot
gets its **own** SQLite chunk-store and BM25 read connections (never a
shared one), opened with `check_same_thread=False`. WAL mode keeps these
independent readers concurrent with, and isolated from, the single ingest
writer — the threading hazard called out in #12. The readiness probe gets
its own dedicated short-timeout connections for the same reason. (#12, #16,
#21)

### FTS5 as a plain table — portability over storage
The BM25 index is a *plain* (own-content) FTS5 table storing chunk text
alongside the inverted index. The contentless-delete options that would
avoid the second copy of the text require SQLite ≥ 3.47 (2024-10), which
neither the CI runner nor common distros (Ubuntu 24.04) bundle — so they
are deliberately avoided. The cost is a duplicate copy of chunk text in
the index; acceptable for a demo, and the fix for a real CI break was to
stay on the portable feature set. (#7, fix in PR #33)

## Ingestion

### Compensation-based atomicity, not cross-store 2PC
Qdrant cannot join a SQLite transaction, so true two-phase commit across
the chunk store, FTS index, and vector index is impossible. Instead the
pipeline chunks and embeds *before* any write (so the likeliest failure,
an Ollama call, aborts with nothing to undo), writes in a fixed order, and
compensates by deleting the just-created version on any failure — the
previously active version is never touched, so retrieval keeps serving it
throughout. Superseded versions are swept only after the new one is fully
written, flakiest store first, and the sweep re-checks every older version
so it *heals* a cleanup a prior ingest failed to finish. (#11)

### Synchronous in-request ingestion behind a single-permit limiter
Extraction (pypdf) and ingestion (embed + writes) are blocking work run
in-request but off the event loop via a single-permit `CapacityLimiter`:
one worker thread at a time, so a slow ingest never freezes concurrent
requests and `extract`'s process-global `warnings` filter can never
overlap another extraction. A 202/queue design was considered and left for
production; synchronous is simpler and honest for a demo. (#8, #12)

### pypdf, no OCR
PDF text is extracted with `pypdf`. Scanned/image-only PDFs yield no text
and are rejected as empty — OCR is an explicit non-goal. (#8, non-goals in
#1)

## Caching

### First-turn-only response cache
`fingerprint()` returns `None` (uncacheable) whenever there is prior
history, so only first-turn questions are cached. A fingerprint folding in
a conversation digest would almost never repeat across users, making the
cache decorative. First-turn FAQs ("how do I reset my password") dominate
support traffic, so caching only the first turn captures nearly all the
reuse while keeping keys shared across users. (#19)

### Fingerprint keys only on the normalized message
The cache key is `sha256` of the casefolded, whitespace-collapsed,
trailing-punctuation-stripped message — **and nothing else**. It does not
fold in the embedding model, reranker, prompt version, or corpus. So a
config change (swap the embedder/model, edit the system prompt, re-ingest)
serves first-turn answers cached under the *old* config until TTL expiry
(default 1h) or #20's doc-eviction. This bounded staleness is acceptable
for a demo; if config changes get frequent, fold a config-version tag into
the fingerprint (`sha256(config_version + normalized)`) or flush `cache:*`
on deploy. (#19)

### Fail-open cache (non-negotiable)
Every Redis error on the cache path is swallowed: a read error is a miss,
a write error is skipped, and doc-update invalidation never raises. A
cache outage must never take chat down or fail an otherwise-good ingest —
the worst case is a live answer computed instead of served from cache, or
a too-stale answer lingering until its TTL. Warnings log the operation
name only, never keys or content. (#19, #20)

### Doc-update invalidation by tag set
On write, each cached answer is tagged under `doc_tag:{doc_id}` for every
document it cited. When a document is re-ingested, the invalidator reads
that tag set and deletes every listed `cache:{fingerprint}` key plus the
tag set itself — so a stale answer is dropped the moment its source
document changes (the "money test" in the README quickstart). The
read-then-delete is two round-trips, not atomic; the one race it exposes
degrades to the same "one stale answer until TTL" the fail-open design
already accepts. (#20)

## Observability

### One structured `request_summary` line per chat request
`/chat` owns a request-scoped `StageTimings` and emits exactly one JSON
`request_summary` log line per request — ids, route, status, `cache_hit`,
per-stage durations (`retrieval_ms`/`rerank_ms`/`llm_ms`), and `total_ms`,
**never** any customer content. Stages that did not run are simply absent,
so a cache hit logs `total_ms` alone (see the README's log-schema section
for real captured lines). The readiness probe reports each dependency by
name only, never surfacing an exception that could embed a connection
string. (#21)

## Non-goals (deliberate)

Authentication/authorization, rate limiting, streaming responses,
multi-tenancy, OCR for scanned PDFs, horizontal scaling (SQLite
single-writer is accepted), and a UI — this is an API-only demo. Each is
listed with what a production build would add in issue #1's non-goals. (#1)
