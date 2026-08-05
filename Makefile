.PHONY: install hooks test lint docs

install: hooks
	uv sync

# Los hooks de git no viajan en el clon: `.git/hooks/` es local por diseño. Este
# objetivo apunta git a los del repositorio, que si estan versionados, para que
# quien monte el entorno los tenga sin tener que enterarse de que existen.
hooks:
	git config core.hooksPath .githooks

test:
	uv run pytest

lint:
	uv run ruff check

docs:
	uv run python scripts/docs_sync.py --write
