# Contributing to Agentic Smart Health

Thank you for your interest in contributing. This document explains how the repository is structured, how to set up your development environment, and the conventions you should follow when adding new applications or packages.

---

## Repository structure

This is a **monorepo managed with [`uv` workspaces](https://docs.astral.sh/uv/concepts/workspaces/)**. A single root `pyproject.toml` declares all workspace members:

```toml
[tool.uv.workspace]
members = ["apps/*", "packages/*"]
```

All members share a single virtual environment (`.venv/`) and a single lockfile (`uv.lock`) at the root. Internal dependencies are resolved via workspace references, not PyPI.

```
agentic-smart-health/
├── apps/              ← deployable applications and servers
├── packages/          ← shared libraries consumed by apps
├── pyproject.toml     ← workspace root
├── uv.lock            ← unified lockfile (commit this file)
└── Makefile           ← development commands
```

---

## Development setup

```bash
# 1. Clone the repository
git clone https://github.com/anfaia/agentic-smart-health.git
cd agentic-smart-health

# 2. Install all workspace dependencies
make install

# 3. Verify everything works
make test
make lint
```

---

## Branching and pull requests

- Work on a feature branch: `git checkout -b feat/short-description`.
- Keep commits small and focused. Write commit messages in the imperative mood.
- Open a pull request against `main`. Every PR must pass `make lint` and `make test`.
- Update `CHANGELOG.md` under `[Unreleased]` with a summary of your changes.
- If your change adds or modifies an agent, update `AGENTS.md` accordingly.

---

## Adding a new application (`apps/`)

Applications are deployable programs: API servers, agent orchestrators, MCP servers, CLIs, etc.

### Architectural principle: hexagonal architecture

Each application must follow a **hexagonal (ports & adapters) architecture**:

```
apps/<my-app>/
├── pyproject.toml
└── src/
    └── <my_app>/
        ├── domain/          ← pure business/domain logic; no I/O, no framework imports
        │   ├── models.py    ← domain entities (extend or reference core-schemas)
        │   └── ports.py     ← abstract interfaces (protocols / ABCs) the domain exposes
        ├── application/     ← use cases; orchestrates domain logic
        │   └── services.py
        ├── adapters/        ← concrete implementations of ports (I/O, external services)
        │   ├── inbound/     ← e.g. HTTP handlers, CLI entrypoints, MCP tool handlers
        │   └── outbound/    ← e.g. file system, database, external APIs
        └── main.py          ← composition root; wires adapters to ports
```

**Rules:**
- The `domain/` layer must not import from `adapters/` or any external framework.
- All cross-cutting data contracts must use schemas from `packages/core-schemas`.
- Agents must be registered in `AGENTS.md` before merging.

### Step-by-step

```bash
# 1. Scaffold the package
mkdir -p apps/my-app/src/my_app/{domain,application,adapters/{inbound,outbound}}

# 2. Create pyproject.toml
cat > apps/my-app/pyproject.toml << 'EOF'
[project]
name = "my-app"
version = "0.1.0"
description = "Short description of what this app does."
requires-python = ">=3.11"
dependencies = [
    "core-schemas",
]

[tool.uv.sources]
core-schemas = { workspace = true }
EOF

# 3. Sync the workspace so the new member is picked up
make install

# 4. Verify the new package is resolved
uv run python -c "import my_app"
```

---

## Adding a new shared package (`packages/`)

Packages are libraries consumed by one or more applications. They must not contain application entry points or framework-specific wiring.

### Architectural principle

Each package exposes a clean public API through its `__init__.py`. Internal modules are private by convention (prefix with `_`). Packages must not depend on other packages in `apps/`; they may depend on other packages in `packages/`.

### Step-by-step

```bash
# 1. Scaffold the package
mkdir -p packages/my-library/src/my_library

# 2. Create pyproject.toml
cat > packages/my-library/pyproject.toml << 'EOF'
[project]
name = "my-library"
version = "0.1.0"
description = "Short description of what this library provides."
requires-python = ">=3.11"
dependencies = []
EOF

# 3. Sync the workspace
make install

# 4. To use this package from an app, add it as a workspace dependency:
#    In apps/my-app/pyproject.toml:
#
#    dependencies = ["my-library"]
#
#    [tool.uv.sources]
#    my-library = { workspace = true }
#
#    Then run: make install
```

---

## Code style

- **Formatter / linter**: [`ruff`](https://docs.astral.sh/ruff/). Run `make lint` before committing.
- **Type hints**: required on all public functions and methods.
- **Docstrings**: Google style for public APIs.
- **Tests**: place tests under `tests/` (global) or `apps/<app>/tests/` (app-specific). Run with `make test`.

---

## Data and clinical sensitivity

- Never commit real patient data, even anonymised, to the repository.
- Use the synthetic datasets provided under `data/` for development and testing.
- Any change that affects how clinical data is ingested, stored, or exported must be reviewed by a maintainer before merging.
- See [SECURITY.md](SECURITY.md) for vulnerability reporting.

---

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
