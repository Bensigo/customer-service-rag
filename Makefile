.PHONY: install lint format test run

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
