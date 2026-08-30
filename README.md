# Agentic Smart Health

Multi-agent system for integrating, analysing and representing heterogeneous dental clinical data on a patient **Digital Twin**, built on Gaussian Splatting with per-point/per-region clinical attributes and support for time series.

[![tests](https://github.com/ANFAIA/Agentic-Smart-Health/actions/workflows/tests.yml/badge.svg)](https://github.com/ANFAIA/Agentic-Smart-Health/actions/workflows/tests.yml)

> Open source · Apache 2.0 · Python ≥ 3.13

---

## Project context

Dental care produces highly heterogeneous data: CBCT scans (DICOM), STL files from intraoral scanners, clinical reports as PDF and 2D photographs. That information lives fragmented in silos, one per vendor and one per clinic, which makes real longitudinal follow-up impossible and undermines the patient's sovereignty over their own health data.

**Agentic Smart Health** addresses this with a multi-agent architecture that autonomously organises, integrates and analyses heterogeneous dental data, projecting it onto a digital twin of the patient. The process is reversible: the system can regenerate STL files and images directly from the Digital Twin.

---

## How it all fits together (the short version)

Several **agents** (workers with a single responsibility) translate heterogeneous
clinical files (DICOM, STL, PDF, photo) into a **common document** — the
`TwinSnapshot` from [`core-schemas`](packages/core-schemas/) — then enrich it and
materialise it so a **viewer** can show it; an **orchestrator**
([`agent-orchestrator`](apps/agent-orchestrator/)) hands out the work. The
"model" (LLM) is not a central layer: it is the *brain* reasoning **inside** one
specific agent (today only `research-agent`), and not every agent needs one.

> 📐 **Full map of the 6 layers and the path the data takes** (written for newcomers):
> [`docs/architecture/multi-agent-pipeline.md` §0](docs/architecture/multi-agent-pipeline.md#0-vista-de-conjunto-para-quien-llega-nuevo).

## Quickstart

### Prerequisites

- Python ≥ 3.13
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed on the system

### Installation

Clone the repository and install every workspace dependency with a single command:

```bash
git clone https://github.com/anfaia/agentic-smart-health.git
cd agentic-smart-health
make install
```

That runs `uv sync`, which resolves and locks all dependencies (internal and external) and creates the virtual environment in `.venv/`.

### Available commands

<!-- generado: make — no editar a mano -->
| Comando | Ejecuta |
|---|---|
| `make install` | `uv sync` |
| `make hooks` | `git config core.hooksPath .githooks` |
| `make test` | `uv run pytest` |
| `make lint` | `uv run ruff check` |
| `make docs` | `uv run python scripts/docs_sync.py --write` |
<!-- /generado: make -->

`make install` also enables the repository's **git hooks**
(`git config core.hooksPath .githooks`), and the `pre-commit` hook does two things:

- **Stops the commit** if `data_guard.py` finds third-party data staged (a PDF, a
  mesh, a large binary). This is the only place where that is cheap to catch: once
  committed, removing it means rewriting history.
- **Regenerates the generated blocks** of the documentation — variable, script and
  command tables, and the agent registry — and **adds them to the same commit**, so
  the docs always travel with the change that affects them. It only touches what
  sits between the markers; never the prose.

If it ever gets in the way, `git commit --no-verify` skips it, and CI will still
flag it on the PR.

### Activating the environment (optional)

If you need to work inside the virtual environment directly:

```bash
source .venv/bin/activate
```

Or prefix any command with `uv run` to execute it inside the environment without activating it:

```bash
uv run python -c "import core_schemas; print('workspace OK')"
```

---

## Current status — MVP closed (week 8)

**Ingestion, fusion, segmentation and the four export channels are built and
tested**, and the full path input → twin → file has an integration test. The
deliverable is a **`.uos`** container: a real clinical case closes at 12 entries and
18 assets, UOS-Core + UOS-Vol conformance, 0 errors. Acquired data does not travel
inside — it is declared by its content address, with a per-slice hash for the CBCT's
397 slices — and the [reference viewer](https://github.com/lgarbayo/uos-viewer) opens
it in the browser without uploading anything.

What is **measured**, every number obtained by re-reading what was produced rather than by promising it:

| What | Measurement | On |
|---|---|---|
| Mesh reversibility | **3.8 × 10⁻⁶ mm** maximum deviation, against a budget of **0.1 mm** | real scan, 110,804 vertices | <!--const:REVERSIBILITY_BUDGET_MM-->
| CBCT ↔ intraoral registration | **0.452 mm** over the overlapping population | real patient |
| Render from the field | **PSNR 102 dB · SSIM 0.99999999**, byte-for-byte reproducible | twin → PLY → render cycle |
| Printable arch | **0.372 mm (p95) · bias −0.02 mm** — *this is not reversibility*: it measures the root reconstructor against the scanned crown | the only band with two measurements of the same tissue |

**Data contract** — `core-schemas` (Pydantic v2, schema **`1.6.0`**). The <!--const:SCHEMA_VERSION-->
`TwinSnapshot` is the common document, carrying `provenance` per value. Ingestion
agents are deterministic and *fail-loud*: they never raise, they return status and
confidence, and there is a **human-in-the-loop gate** on a threshold (0.7). The <!--const:DEFAULT_HITL_THRESHOLD-->
orchestrator honours a budget of <60 s. <!--const:LATENCY_BUDGET_S-->

**Not yet**: **per-pixel** colour — the signal is measured, what is missing is camera
pose — and the `pathology-agent`. `3dgs-engine` is a placeholder: reconstruction
lives in the notebooks with `gsplat`.

> 📋 **The honest closing inventory** — what is measured, what is unresolved, in what
> order to attack it, plus the milestones and the success metrics — is in
> [`docs/cierre-mvp.md`](docs/cierre-mvp.md), along with what CI **cannot** verify
> because it has no GPU.

---

## Monorepo architecture

The repository is organised as a **monorepo managed with [`uv` workspaces](https://docs.astral.sh/uv/concepts/workspaces/)**. The root `pyproject.toml` declares the workspace and automatically groups every member under `apps/` and `packages/`:

```toml
[tool.uv.workspace]
members = ["apps/*", "packages/*"]
```

This lets each application and package keep its own `pyproject.toml` and independent lifecycle while sharing a single virtual environment (`.venv/`) at the root and a common lockfile (`uv.lock`). Internal dependencies resolve through workspace references (`workspace = true`), without going through PyPI.

```
agentic-smart-health/          ← workspace root
├── pyproject.toml             ← uv workspace declaration
├── uv.lock                    ← unified lockfile
├── Makefile                   ← development commands
├── apps/
│   ├── agent-orchestrator/    ← orchestrator of the multi-agent system
│   ├── research-agent/        ← research agent (RAG + scientific literature)
├── packages/
│   ├── core-schemas/          ← shared Pydantic schemas (the TwinSnapshot contract)
│   ├── ingestion-agents/      ← 4 ingestion agents (mesh · cbct · report · image)
│   ├── fusion-agents/         ← geometric and semantic fusion over the twin
│   ├── analysis-agents/       ← anatomical segmentation: region_id (FDI) per Gaussian
│   ├── export-agents/         ← mesh, field and render regenerated from the twin, with the error measured
│   ├── gaussian-engine/       ← fitting anisotropic ellipsoids to the density the CBCT measured
│   ├── uos/                   ← Unified Oral Scene container: the whole case with its relations declared
│   ├── tooth-aggregation/     ← aggregating per-point labels into tooth instances
│   └── 3dgs-engine/           ← placeholder (3DGS reconstruction lives today in notebooks + gsplat)
├── data/
│   └── research-agent/        ← knowledge base of the research agent
├── schemas/                   ← published JSON Schema of the UOS manifest, per version (§12)
├── docs/                      ← documentation (see the note below)
├── notebooks/                 ← experimentation and exploration (01–09)
├── experiments/               ← test beds outside the pipeline (CBCT→Blender→3DGS, HU layers)
├── tests/                     ← global test suite
├── scripts/                   ← utilities: Blender render, PR auditor, dataset fetchers
└── .github/
    └── workflows/             ← CI: code review agent (ai-code-reviewer)
```

---

## Applications (`apps/`)

### `agent-orchestrator`

Central orchestrator of the multi-agent system. It coordinates the agents of each pipeline phase:

- **Ingestion** ✅ *(implemented)*: fires the 4 `ingestion-agents` in parallel over one acquisition (STL + CBCT + report + N photos), assembles the `TwinSnapshot` and applies the human review gate; budget of <60 s. <!--const:LATENCY_BUDGET_S-->
- **Fusion** ✅ *(implemented)*: `IngestionPipeline.fuse()` chains **two** `GeometricFusionAgent` runs — scanner↔scanner registration and the IOS↔CBCT ICP, each with its `rms_error_mm` and its verification status — and the `SemanticFusionAgent`, which hangs the report's findings off FDI codes and flags the conflict when report and geometry disagree.
- **Analysis** 🟡 *(the anatomical part, yes; the clinical part, no)*: the `segmentation-agent` runs **inside `fuse()`**, between the two fusion stages, and fills `PipelineResult.analysis`. ⚠️ Its quality is measured and is the MVP's main gap: **11 of 14 teeth are discarded on anatomical grounds** ([`docs/research/segmentacion-fdi-escaner.md`](docs/research/segmentacion-fdi-escaner.md)). Clinical reasoning — the `pathology-agent` — is still `planned`, and ships with mandatory human review by design.
- **Export** ✅ *(all four channels)*: `export-agents` regenerates from the `TwinSnapshot` the **mesh** as STL, the **Gaussian field** as PLY (in the twin's frame or in the CBCT's real millimetres) and a **multi-view render** as PNG by Beer-Lambert, each with its measured error — maximum and mean deviation for geometry, PSNR/SSIM for the image. The orchestrator fires them with `IngestionPipeline.exportar(result, destino)`, and the full path **input → twin → file** is tested end to end in `tests/test_e2e.py`.

It depends on `core-schemas` and `ingestion-agents` (via the workspace) so that data contracts stay shared with the rest of the system.

### Interoperability with [3D Slicer](https://www.slicer.org/) and other platforms

**Through open formats, not through a server.** The pipeline materialises every case as
STL, PLY and PNG, plus the `TwinSnapshot`'s own JSON, and Slicer reads all of them
natively. That is already interoperability: there is no protocol to negotiate and no
service to keep alive, and the file still opens ten years from now without us.

There used to be a `slicer-mcp-server` here and it **has been withdrawn**. It was a
directory holding a `server.py` of **zero lines**, described in this very README in the
present tense — "exposes an interface", "lets agents interact" — and unblocking it
depended on a third party confirming the call's format and direction. An empty piece we
cannot unblock ourselves is not architecture: it is an intention written where facts are
documented.

An MCP server would make sense for **live, bidirectional** interaction — an agent driving
the Slicer session, not reading a file. Nobody has asked for that yet, and when they do it
gets built. See issue #40, which is now a question for the partner rather than a component
of this repository.

### `research-agent`

An autonomous research agent that searches, ingests and summarises scientific literature on 3D Gaussian Splatting, the DICOM standard and clinical regulation. Built with Python, Anthropic Claude / Ollama, Qdrant and local embeddings.

**Main capabilities:**
- Semantic paper search on Semantic Scholar and arXiv
- Document ingestion and indexing through RAG (Qdrant + fastembed)
- Structured report generation in Markdown
- Local execution with Ollama (free, no API key)

**Run modes:**
- `uv run python -m src.main` — Claude with native tool calling (requires an API key)
- `uv run python -m src.main_local` — local Ollama (free, 100% private)

**Starting corpus.** The reference PDFs are **not in the repository**: they are
third-party binaries and the licence of many of them does not allow
redistribution. What is versioned is the inventory
([`manifest.yaml`](data/research-agent/knowledge_base/manifest.yaml): title, DOI or
arXiv ID, URL and the licence verified at the source for each document). To
materialise them:

```bash
uv run python scripts/fetch_knowledge_base.py          # download what is missing
uv run python scripts/fetch_knowledge_base.py --check  # check only
```

A couple of publishers (Wiley, AAAI) will not serve the PDF to a script: those are
left as manual downloads and the command prints the link. The agent works without a
corpus — `search_references` discovers new literature — but `read_directory` and
`index` will find nothing until it has been run.

**Layout:**
- `src/main.py` — CLI orchestrator with Claude
- `src/main_local.py` — local variant with Ollama
- `src/tools.py` — system tools (disk sandbox)
- `src/rag.py` — RAG engine (Qdrant + fastembed)
- `src/references.py` — paper discovery

It does not depend on `core-schemas`; it keeps its own internal models for RAG.

**Note:** this agent is a port of [jeicob](https://github.com/lgarbayo/jeicob), adapted to fit the monorepo.

---

## Shared packages (`packages/`)

One sentence per package; the full card for each agent is in [`AGENTS.md`](AGENTS.md).

### `core-schemas`

**Single source of truth** for the types: Pydantic v2 schemas shared across the whole
workspace — `TwinSnapshot`, `Provenance`, per-tooth FDI observations — versioned, today
`1.6.0`. <!--const:SCHEMA_VERSION-->

### `ingestion-agents`

The **4 ingestion agents** (`mesh` · `cbct` · `report` · `image`), one per modality.
Deterministic and *fail-loud*, with `Provenance` per value, a content-addressed
`ArtifactStore` and EXIF discarded by construction. To add one: the `add-ingestion-agent`
skill.

### `export-agents`

The only family that **writes** output files: STL, PLY, multi-view PNG and a printable
arch. All of them **measure what they produce by re-reading it**. Two things that surprise
people: the field's PLY is not a 3DGS `.ply`, and the render does not rasterise splats —
`density` is radiological attenuation, not opacity, so it composites by Beer-Lambert,
which is also order-independent and therefore byte-for-byte reproducible.

### `uos`

**The container and its manifest**, the project's deliverable: an uncompressed ZIP holding
the whole case with **the relations between its parts declared**. The rule that holds it
up is that **the measured and the inferred do not mix**: inference lives only under
`derived/`, and a `.uos` with no `derived/` is still valid and complete. Schema in
[`schemas/`](schemas/), format in
[`docs/spec/uos-format-spec-v0.2.tex`](docs/spec/uos-format-spec-v0.2.tex).

### `fusion-agents`

**Fusion** (ADR 004): geometric registration by ICP, always declaring its `rms_error_mm`
and whether anyone has verified it, plus anchoring the report's findings to FDI codes,
with the **conflict** flagged when report and geometry disagree.

### `analysis-agents`

**Anatomical analysis**: `region_id` per Gaussian and the `FDI → confidence` map. ⚠️ Its
quality is measured and is the MVP's main gap
([`docs/research/segmentacion-fdi-escaner.md`](docs/research/segmentacion-fdi-escaner.md)).

### `gaussian-engine`

**Field fitting**: from isotropic voxel-sized seeds to measured ellipsoids. The only
package that touches `torch`, and it imports it inside the function so it installs without
CUDA.

### `tooth-aggregation`

**Point → tooth aggregation** (instances + FDI). Deliberately free of `torch`: the
*forward* pass is the caller's business.

### `3dgs-engine`

**Placeholder.** 3DGS reconstruction lives today in the [notebooks](notebooks/) with
`gsplat` and Blender. It gets promoted to a package once the recipe stops being
experimental.

---

## Notebooks — proofs of concept (spikes)

Nine **technical validation spikes** (not the final system, and not clinical results) that
de-risk the architectural decisions before each link becomes an agent. They run on real,
gitignored datasets: **Teeth3DS+** (01–06) and **Bite2Text** (07). Notebook `07` is the one
that wires the ingestion agents into the reconstruction flow, with colour taken from the
photos and a 31.5 dB holdout.

What each one validates, its scope and how to run them:
[`notebooks/README.md`](notebooks/README.md).


## Code review and CI

Every Pull Request goes through a **static review guardian agent**
([`ai-code-review.yml`](.github/workflows/ai-code-review.yml)). It uses no LLM: it combines
Ruff and MyPy with a bespoke architecture auditor, and reviews **only the Python files the
PR touches**. It publishes inline annotations and a summary comment. Architecture
violations and coverage below 80% block the merge.

On top of that, [`docs_sync.py`](scripts/docs_sync.py) checks that this documentation does
not drift from the code — cited paths, agent registry, constants, the tree above — and a
pre-commit hook aborts the commit if it tries to version clinical data.

**Literature watch** — the repository's only scheduled job
([`literature-watch.yml`](.github/workflows/literature-watch.yml)): every Monday it
searches arXiv for what was published that week, reads the licence from arXiv's OAI-PMH
(it does not assume it) and opens a PR proposing new manifest entries. **It does not
merge.** No PDF is ever written to the runner: they are downloaded into memory to compute
`sha256` and released right there.


Repository utilities (this table is generated by `docs_sync.py`):

<!-- generado: scripts — no editar a mano -->
| Script | Qué hace |
|---|---|
| [`scripts/ablacion_recetas.py`](scripts/ablacion_recetas.py) | Ablación de la receta de entrenamiento: qué aporta cada pieza. |
| [`scripts/altura_corona.py`](scripts/altura_corona.py) | mide la altura de corona clínica sobre el escáner intraoral. |
| [`scripts/audit_pr.py`](scripts/audit_pr.py) | Guardián de las reglas de arquitectura del monorepo. |
| [`scripts/blender_render_views.py`](scripts/blender_render_views.py) | Render multivista de una malla intraoral con **Blender** (headless). |
| [`scripts/caso_completo.py`](scripts/caso_completo.py) | El pipeline entero sobre un caso clínico real, etapa por etapa. |
| [`scripts/composicion_cbct_ios.py`](scripts/composicion_cbct_ios.py) | Dientes segmentados en el CBCT + encía del IOS, en gaussianas. |
| [`scripts/data_guard.py`](scripts/data_guard.py) | Impide que datos ajenos entren al repositorio sin permiso. |
| [`scripts/desplazamiento_relativo.py`](scripts/desplazamiento_relativo.py) | ¿Se puede decir «esta pieza se desplazó X mm»? Referencia leave-one-out y umbral. |
| [`scripts/docs_sync.py`](scripts/docs_sync.py) | Comprueba que la documentación no le mienta al código. |
| [`scripts/entrena_diente_cbct.py`](scripts/entrena_diente_cbct.py) | Segmentador de diente en CBCT, contra el listón del umbral. |
| [`scripts/entrena_gs_escaner.py`](scripts/entrena_gs_escaner.py) | 3DGS de verdad sobre la superficie del escaner. |
| [`scripts/entrenar_3dgs.py`](scripts/entrenar_3dgs.py) | EXPERIMENTO con resultado NEGATIVO: 3DGS entrenado de una arcada. |
| [`scripts/eval_informes.py`](scripts/eval_informes.py) | ¿Cuánto de lo que dice un informe acaba en el contrato? |
| [`scripts/fetch_knowledge_base.py`](scripts/fetch_knowledge_base.py) | Materializa la knowledge base del `research-agent`. |
| [`scripts/fetch_teeth3ds.sh`](scripts/fetch_teeth3ds.sh) | Descarga reproducible de Teeth3DS+ desde el Google Drive oficial. |
| [`scripts/malla_mejorada.py`](scripts/malla_mejorada.py) | El STL mejorado, sacado del contenedor y de nada más. |
| [`scripts/metricas.py`](scripts/metricas.py) | las cuatro cifras del brief, MEDIDAS y no prometidas. |
| [`scripts/mide_segmentacion.py`](scripts/mide_segmentacion.py) | cuanto se puede DESCARTAR de la segmentacion FDI de un `.uos`. |
| [`scripts/prepara_toothfairy.py`](scripts/prepara_toothfairy.py) | Descarga ToothFairy2 caso a caso y lo deja entrenable. |
| [`scripts/promedio_y_escala.py`](scripts/promedio_y_escala.py) | Dos preguntas de diseño sobre el registro por diente, medidas en vez de argumentadas. |
| [`scripts/refina_3dgs.py`](scripts/refina_3dgs.py) | La fase que faltaba: el campo semilla optimizado como 3DGS. |
| [`scripts/registro_ios_cbct.py`](scripts/registro_ios_cbct.py) | mide si el escáner intraoral y el CBCT se pueden alinear. |
| [`scripts/resolucion_modalidades.py`](scripts/resolucion_modalidades.py) | Simula qué resolución alcanza cada modalidad dental. |
| [`scripts/segmentar_fdi.py`](scripts/segmentar_fdi.py) | etiqueta cada diente de una arcada con su código FDI. |
| [`scripts/seguimiento_histora.py`](scripts/seguimiento_histora.py) | cuánto se ha movido el margen gingival entre dos escaneos. |
| [`scripts/umbral_vs_verdad.py`](scripts/umbral_vs_verdad.py) | ¿Cuánto diente recupera un umbral, contra una verdad conocida? |
| [`scripts/verifica_contenedor.py`](scripts/verifica_contenedor.py) | que el `.uos` diga la verdad SOBRE SI MISMO. |
| [`scripts/watch_literature.py`](scripts/watch_literature.py) | Vigila la literatura y propone entradas del manifiesto. |
<!-- /generado: scripts -->

Development tools install with `uv sync --group dev` (group `dev`: `ruff`, `mypy`). Full agent card in [`AGENTS.md`](AGENTS.md).

---

## Environment variables

Copy the example file and set the variables you need:

```bash
cp .env.example .env
```

`.env.example` documents **only the variables the code actually reads**, together
with who uses them and what happens if they are unset. This table is generated by
[`scripts/docs_sync.py`](scripts/docs_sync.py) from the code, and CI fails if it
drifts — which is why it is not edited by hand. A `—` in the last column means the
call carries no default (the module may have its own fallback):

<!-- generado: env-vars — no editar a mano -->
| Variable | Se lee en | Por defecto |
|---|---|---|
| `ANTHROPIC_API_KEY` | `apps/research-agent/src/main.py` | — |
| `ASH_PSEUDONYM_SALT` | `packages/ingestion-agents/src/ingestion_agents/cbct_agent.py` | `dev-salt-no-usar-en-produccion` |
| `OLLAMA_HOST` | `apps/research-agent/src/main_local.py` | `http://localhost:11434` |
| `QDRANT_PATH` | `apps/research-agent/src/rag.py` | — |
| `RESEARCH_AGENT_LOCAL_MODEL` | `apps/research-agent/src/main_local.py` | `qwen2.5:7b` |
| `RESEARCH_AGENT_MODEL` | `apps/research-agent/src/main.py` | `claude-opus-4-8` |
<!-- /generado: env-vars -->

None of them is needed to run `make test`. The pseudonym salt is the only one that
is a **secret**: without it the pipeline still runs, but the pseudonyms it emits are
not fit for patient data — and if it changes later, they stop matching the ones
already emitted.

---

## Documentation

> **Note:** the `docs/` directory is reserved exclusively for research and architecture documentation. It holds no user documentation and no usage tutorials.
>
> - `docs/architecture/` — design decisions, architecture diagrams and ADRs (Architecture Decision Records).
> - `docs/research/` — bibliographic references and research notes on Gaussian Splatting, DICOM/STL standards, clinical interoperability and applicable regulation (GDPR, HIPAA).
> - `docs/spec/` — the normative specification of the `.uos` format and the project white paper, in LaTeX.

Technical documentation aimed at developers and contributors stays in this README and in each component's `pyproject.toml`.

### Where to start reading

| Document | Answers |
|---|---|
| [`docs/cierre-mvp.md`](docs/cierre-mvp.md) | what is measured, what is unresolved and what is left for later |
| [`docs/spec/uos-format-spec-v0.2.tex`](docs/spec/uos-format-spec-v0.2.tex) | the format specification: what a `.uos` carries, how it is read, how it is extended |
| [`docs/spec/uos-white-paper.tex`](docs/spec/uos-white-paper.tex) | why a new format is needed, which hypotheses were tested and with what results |
| [`docs/research/segmentacion-fdi-escaner.md`](docs/research/segmentacion-fdi-escaner.md) | why FDI segmentation is not solved, with the measurement |
| [`docs/research/frontera-encia-desde-foto.md`](docs/research/frontera-encia-desde-foto.md) | where the tooth-gum boundary actually is, and what is missing to use it |
| [`docs/research/color-por-pieza-desde-foto.md`](docs/research/color-por-pieza-desde-foto.md) | the shade of each crown, and how the flash falloff is discounted without inverting it |
| [`docs/research/segmentacion-diente-cbct.md`](docs/research/segmentacion-diente-cbct.md) | how far a classifier gets on the CBCT, and where it stops getting there |

---

## License

[Apache License 2.0](LICENSE)

---

*ANFAIA Summer Grants 2026 · July – August 2026*
