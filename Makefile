.PHONY: install lint format test run eval eval-rerank compose-check

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

run:
	uv run uvicorn --factory app.main:create_app --reload

eval:
	uv run python -m app.eval

eval-rerank:
	uv run python -m app.eval --rerank

compose-check:
	docker compose config -q
