# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

customer-service-rag — a Retrieval-Augmented Generation (RAG) system for customer service. Python 3.12 managed with uv, FastAPI, src layout (package `app` in `src/app/`, tests in `tests/`). Roadmap and architecture live in GitHub issue #1; an architecture overview is added here once the pipelines land (#22).

## Commands

- `uv sync` — install/refresh dependencies (pinned via `uv.lock`)
- `make test` — run the test suite (pytest)
- `uv run pytest tests/test_config.py::test_settings_defaults -v` — run a single test
- `make lint` — ruff lint + format check
- `make format` — auto-format and auto-fix lint
- `make run` — start the API locally (uvicorn with reload)
- `docker compose up --build` — run the full stack (app :8000, Qdrant :6333, Redis :6379); Ollama is expected on the host at :11434, not in compose
- `make compose-check` — validate compose.yaml

## Development Workflow

Every change, no matter how small, follows this workflow in order. These rules are non-negotiable.

### 1. Plan first — track everything on GitHub

- Before writing any code, produce a short written plan and file it as a GitHub issue (`gh issue create`).
- Break large features into multiple small issues; each issue must be deliverable as one small PR.
- Link every PR to its issue (`Closes #<n>`).

### 2. Test-driven development (red-green-refactor)

All code is written test-first:

1. **Red** — write a failing test that describes the desired behavior. Run it and confirm it fails for the expected reason before writing any implementation.
2. **Green** — write the minimum implementation that makes the test pass. Run the tests and confirm they pass.
3. **Refactor** — clean up the code while keeping all tests green.

Never write implementation code before a failing test exists, and never weaken or delete a test to get to green.

### 3. Subagent code review — before every PR

- After implementation is complete and tests pass, dispatch a code-review subagent (e.g. the `code-review` skill or a code-reviewer agent) to review the full diff.
- Fix all confirmed findings before opening the PR; note any disputed findings in the PR description.

### 4. Security check — before every PR

- Run a security review of the diff (e.g. `/security-review`) in addition to the code review, and fix what it finds.
- RAG-specific rules:
  - Treat retrieved documents and user messages as untrusted input — guard against prompt injection reaching system prompts or tool calls.
  - Never log or index customer PII without scrubbing it first.
  - All secrets (LLM API keys, database URLs) live in `.env` (gitignored) and are read from environment variables — never hardcoded.

### 5. Small PRs only

- Never commit directly to `main`. Create one branch per issue (`feat/<issue>-<slug>`, `fix/<issue>-<slug>`).
- Keep each PR small and focused — one issue, one concern, ideally under ~400 changed lines. If it grows beyond that, split it.
- A PR is ready only when it has: passing tests written via TDD, a completed subagent code review, a security check, and a linked issue.

### 6. Code standards

- Prefer simple, readable code over clever code; small functions with clear names.
- Handle errors explicitly — no silently swallowed exceptions.
- No dead code, commented-out code, or TODOs without a linked issue.
- Vet and pin new dependencies before adding them.
