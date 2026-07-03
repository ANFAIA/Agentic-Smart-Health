.PHONY: install test lint

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check
