.PHONY: install lint format test run eval compose-check

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

compose-check:
	docker compose config -q
