# AGENTS.md — Central Agent Registry

This document is the single source of truth for every agent in the **Agentic Smart Health** system. Every autonomous or semi-autonomous agent of the multi-agent system must be registered here with its role, available MCP tools and delegation rules.

Update this file whenever you add, modify or retire an agent. Design decisions affecting the agent architecture must also be recorded in `docs/architecture/`.

---

## Agent design principles

- **Single responsibility**: each agent has a bounded role and does not replicate another agent's logic.
- **Data contracts**: agents communicate exclusively through the schemas defined in `packages/core-schemas`.
- **Human-in-the-loop**: clinically sensitive decisions require explicit human supervision before they execute. This must be stated in the corresponding agent's delegation rules.
- **Traceability**: every agent must record what data it ingested, what transformation it applied and what output it produced.
- **Data sovereignty**: no agent may retain, forward or persist clinical data outside the authorised storage defined in the architecture.

---

## Agent registry

Summary generated from the code — the cards below are not: inputs, outputs and
tools are prose and are written by hand.

<!-- generado: agentes — no editar a mano -->
| Agent | Implemented in |
|---|---|
| `cbct-agent` | [`packages/ingestion-agents/src/ingestion_agents/cbct_agent.py`](packages/ingestion-agents/src/ingestion_agents/cbct_agent.py) |
| `composite-export-agent` | [`packages/export-agents/src/export_agents/compuesto.py`](packages/export-agents/src/export_agents/compuesto.py) |
| `composite-mesh-export-agent` | [`packages/export-agents/src/export_agents/malla_compuesta.py`](packages/export-agents/src/export_agents/malla_compuesta.py) |
| `export-agent` | [`packages/export-agents/src/export_agents/stl.py`](packages/export-agents/src/export_agents/stl.py) |
| `field-export-agent` | [`packages/export-agents/src/export_agents/field.py`](packages/export-agents/src/export_agents/field.py) |
| `geometric-fusion-agent` | [`packages/fusion-agents/src/fusion_agents/geometric.py`](packages/fusion-agents/src/fusion_agents/geometric.py) |
| `image-agent` | [`packages/ingestion-agents/src/ingestion_agents/image_agent.py`](packages/ingestion-agents/src/ingestion_agents/image_agent.py) |
| `mesh-agent` | [`packages/ingestion-agents/src/ingestion_agents/mesh_agent.py`](packages/ingestion-agents/src/ingestion_agents/mesh_agent.py) |
| `render-export-agent` | [`packages/export-agents/src/export_agents/render.py`](packages/export-agents/src/export_agents/render.py) |
| `report-agent` | [`packages/ingestion-agents/src/ingestion_agents/report_agent.py`](packages/ingestion-agents/src/ingestion_agents/report_agent.py) |
| `segmentation-agent` | [`packages/analysis-agents/src/analysis_agents/segmentation.py`](packages/analysis-agents/src/analysis_agents/segmentation.py) |
| `semantic-fusion-agent` | [`packages/fusion-agents/src/fusion_agents/semantic.py`](packages/fusion-agents/src/fusion_agents/semantic.py) |
| `uos-export-agent` | [`packages/uos/src/uos/agente.py`](packages/uos/src/uos/agente.py) |
| `viewer-export-agent` | [`packages/export-agents/src/export_agents/visor.py`](packages/export-agents/src/export_agents/visor.py) |
<!-- /generado: agentes -->


### `research-agent` — Research agent

| Field | Value |
|---|---|
| **Name** | `research-agent` |
| **Version** | `0.1.0` |
| **Location** | `apps/research-agent/` (`src/main.py` · `src/main_local.py`) |
| **Status** | `active` |
| **Pipeline phase** | Knowledge ingestion and synthesis (does not touch patient clinical data) |
| **Brain (LLM)** | Claude (`claude-opus-4-8` by default) via the Anthropic SDK's Tool Runner. Free local variant with Ollama (`main_local.py`, hand-written ReAct loop). |

**Role / Purpose**

> A conversational (CLI) agent that **discovers, ingests, indexes and synthesises**
> scientific literature on 3D Gaussian Splatting, the DICOM standard and clinical
> regulation. It retrieves papers from open academic sources, loads them into a
> local vector store (RAG) and produces structured Markdown reports (abstract +
> full explanation) in `docs_output/`. It operates in the **knowledge ingestion**
> phase: it feeds the project bibliographic context; it **does not process patient
> clinical data**.

**Tools it has access to** (native `@beta_tool` tool calling, **not** MCP)

| Tool | Backend | Permissions | Notes |
|---|---|---|---|
| `read_directory` | Filesystem (`data/research-agent/knowledge_base/`) | read | Lists the documents available in the corpus. |
| `read_file` | Filesystem (`knowledge_base/`) | read | Reads a full PDF/MD/TXT (truncated to 100k chars). Sandboxed against *path traversal*. |
| `ingest_corpus` | RAG — on-disk Qdrant + `fastembed` | write (index) | Chunks (1000 with 150 overlap), embeds and indexes the whole corpus. Idempotent. |
| `search_corpus` | RAG — Qdrant | read | Semantic search (cosine, `top_k=5`) across documents. |
| `search_references` | HTTP — Semantic Scholar → arXiv (fallback) | read (network) | Discovers external papers; no API key. |
| `download_reference` | HTTP + Filesystem (`knowledge_base/`) | write (file) | Downloads a PDF (http(s) only, validates `%PDF`, max 50 MB) and auto-indexes it. |
| `write_summary` | Filesystem (`docs_output/`) | write | Persists the final report; validates its structure (abstract + longer explanation). Forces `.md` and a base filename. |

> **Security boundaries:** reads confined to `knowledge_base/`, writes confined to
> `docs_output/`; `../` and escaping symlinks are blocked before touching disk. The
> default model goes through `ANTHROPIC_API_KEY` (`.env`); the Ollama variant sends
> no document off the machine.

**Expected inputs**

```
A natural-language query from the user (interactive CLI), e.g.:
  "Find recent literature on 3DGS in dental imaging and summarise it."
Starting corpus (optional): .pdf/.md/.txt files under
  data/research-agent/knowledge_base/
  The PDFs are NOT versioned (third-party licences): they are materialised with
  `uv run python scripts/fetch_knowledge_base.py` from manifest.yaml
```

**Generated outputs**

```
- Markdown report at apps/research-agent/docs_output/resumen_<topic>.md
  Structure: # Title · > Source · ## Abstract · ## Full explanation
             · ## Key points (optional)
- Side effect: Qdrant vector store persisted at
  data/research-agent/.qdrant_data/ (collection "papers")
- Conversational answers citing the source (document name)
```

**Delegation rules**

- Autonomous discovery flow: `search_references` → `download_reference`
  (auto-indexes) → `search_corpus`/`read_file` → `write_summary`.
- **Requires no human approval**: it operates on public literature and its own file
  sandbox; it accesses neither clinical data nor authorised patient storage.
- It cannot delegate to other agents in the system (there is no orchestrator
  integration yet); it is a self-contained, single-turn interactive agent.
- Failure policy: tools **never** raise to the caller — they return the error as
  text (`ERROR: …`) so the model can react or retry. Semantic Scholar falls back to
  arXiv automatically on any failure (including 429).
- Concurrency: on-disk Qdrant locks the directory to a single process; real
  concurrency would mean migrating to server-side Qdrant without changing the interface.

**Change history**

| Date | Version | Change |
|---|---|---|
| 2026-07-14 | 0.1.0 | Initial registration: local RAG (Qdrant + fastembed), filesystem tools, external discovery (Semantic Scholar/arXiv) and report generation. Claude and Ollama variants. |

---

### `ai-code-reviewer` — CI/CD guardian agent

| Field | Value |
|---|---|
| **Name** | `ai-code-reviewer` |
| **Type** | Dev-time guardian agent (not part of the production system) |
| **Location** | `.github/workflows/ai-code-review.yml` + `scripts/audit_pr.py` |
| **Status** | `active` |
| **Technology** | **Static review, no LLM**: Ruff + MyPy + a bespoke architecture auditor |

**Role / Purpose**

> Automatically audits every Pull Request before merge to guarantee code quality,
> typing and compliance with the monorepo's hexagonal architecture.
> It fires on `pull_request` events (`opened`, `synchronize`, `reopened`) and
> reviews **only the Python files the PR touches** (focused on the diff).

**Checks it runs**

| Check | Tool | Blocks the merge? | How it reports |
|---|---|---|---|
| **Style / lint** | `ruff check` | **Yes** | GitHub's native inline annotations |
| **Types** | `mypy` (only `*/src/`) | **Yes** | Inline annotations (`::error`/`::warning`) |
| **Architecture** | `scripts/audit_pr.py` | **Yes** | Review comment on the affected line + summary |

> **`ruff format` is not checked**, and that is a decision, not an oversight. This
> repository's formatting is deliberate — tuples grouped by meaning, comments aligned with
> the data they explain — and `ruff format` undoes it: 46 of 87 files would change, and
> `_PROPIEDADES` would go from three grouped lines (position · scale · rotation) to eleven
> loose ones. A permanently red check nobody can fix teaches people to ignore checks.
> Whatever is mechanisable about formatting — line length, import order — is already
> covered by `ruff check`.

> **MyPy only looks at `src/`**, which is the code that runs in production and is clean
> today. What stays out, measured before deciding: `scripts/` (17 errors; exploratory,
> loads modules through `importlib` and lives on untyped arrays), the tests (49; almost
> all of them a `TwinSnapshot | None` the test itself already asserts is not None, and
> whose gate is pytest), and `packages/3dgs-engine/`, whose hyphen is not a valid Python
> identifier and makes MyPy abort before looking at anything. Adding any of the three
> would leave the gate red from day one, which is the fastest way for a gate to end up
> disabled.

**Architecture rules audited** (rule → the violation it detects)

1. **Strict Pydantic v2 in `packages/core-schemas`**: forbids the `pydantic.v1` shim,
   v1 decorators and styles (`@validator`, `@root_validator`, `class Config`) and
   `BaseSettings` (moved to `pydantic-settings`).
2. **No cross dependencies inside `apps/`**: a component under `apps/` cannot import
   another app's package. Shared code must live in `packages/`
   (e.g. `core-schemas`) and communicate through its data contracts.

**Permissions (GitHub Actions token)**

| Permission | Level | Reason |
|---|---|---|
| `contents` | `read` | Check out the PR's code |
| `pull-requests` | `write` | Publish inline comments and the review summary |
| `checks` | `write` | Mark the check as failed when there are architecture violations |

> The agent has **no** write permission over the code (`contents: read`): it cannot
> apply changes or merge; it only comments and passes/blocks the check. It uses the
> workflow's ephemeral `GITHUB_TOKEN`, with no external secrets and no access to
> clinical data.

**Failure policy**

- Architecture violation → the check fails (`core.setFailed`) and blocks the merge.
- Ruff/MyPy errors → the step fails and blocks the merge, besides annotating the line.
- **Every gate gives its verdict even when another one closes**: later steps carry
  `!cancelled()`, so a style error does not leave the PR unseen by the data guardian.
  The job fails all the same; what is not lost is the full diagnosis.
- The summary publishes each gate's **real** result (`clean` / `blocks` / `not
  applicable`), not an "executed" that said the same thing with a clean linter and with
  twenty errors.
- Any unparseable file is skipped by the auditor (Ruff/MyPy catch those).

**Pinned versions.** `ruff` and `mypy` are pinned with `==` in `pyproject.toml`, not
`>=`. A linter that updates itself premieres new rules on code nobody has touched and
turns CI red because of a `uv sync` on a branch that changed nothing — which happened
with `ruff>=0.9`, ending up installing 0.15. Bumping a version is a deliberate commit
that fixes whatever the new version finds.

**Change history**

| Date | Version | Change |
|---|---|---|
| 2026-07-14 | 0.1.0 | Initial registration of the CI guardian agent |
| 2026-08-17 | 0.1.0 | Ruff and MyPy go from informative to a **merge gate**; versions pinned; `ruff format` withdrawn; `scripts/` out of MyPy; later gates no longer skipped when one fails. Debt paid off to be able to close them: 27 Ruff errors and 9 MyPy errors. |

---

### `data-guardian` — Data and licence guardian

| Field | Value |
|---|---|
| **Name** | `data-guardian` |
| **Version** | `0.1.0` |
| **Location** | [`scripts/data_guard.py`](scripts/data_guard.py) |
| **Status** | `active` |
| **Trigger** | `pre-commit` hook (**blocks**) · `ai-code-review.yml` · `literature-watch.yml` |
| **Technology** | Typed code, **no LLM** |

**Role / Purpose**

> Stops third-party, incompatibly licensed or clinical data from entering git history.
> It is the only guardian that **halts the commit**, and the asymmetry is deliberate: a
> file already in history cannot be removed without rewriting it. That happened
> (issue 45: 156 MiB of PDF, `filter-repo` and a `force-push`). Here it is cheaper than
> anywhere else.

**What it checks** (every one born from a real failure)

| Check | What it catches |
|---|---|
| `extensiones` | `.pdf`, meshes (`.ply/.stl/.obj/.glb/.gltf`), volumes (`.dcm/.nii/.nrrd`…), ML weights |
| `tamano` | any versioned file above 5 MiB |
| `ignore` | asks `git check-ignore` about **probe paths**: it verifies behaviour, not the text of `.gitignore`, so it survives someone reordering them and detects that they were deleted |
| `manifiesto` | that the entries in `manifest.yaml` are complete and parse |
| `procedencia` | a versioned notebook with embedded images must cite its dataset card |
| `fichas` | that every dataset used has one |

> The identifiers are not decorative: they come from the script's `COMPROBACIONES`
> registry, and the `docs-guardian` compares this table against it. Adding a check and
> not documenting it here — or removing one and leaving it announced — fails in CI.

**Delegation rules**

- **It blocks, it does not warn**: it exits non-zero and the commit does not happen. To
  skip it you have to type `--no-verify` by hand, which leaves a trace of intent.
- It does not adjudicate doubtful licences: it vetoes by extension and size, which are
  mechanical criteria. Judging whether a licence permits redistribution is human work.

> ⚠️ **Known blind spot (2026-08-09, still open and NOW WIDER).** It vetoes `.stl`,
> `.ply` and `.obj`, but an `.html` with that same geometry embedded as base64 walks
> straight past: the only barrier is the size limit, which is not the right criterion.
> It was found while building a viewer with real patient geometry. The repository is
> public.
>
> Raising the limit from 2 to 5 MiB on 2026-08-18 **widens this gap**: a 4 MiB viewer
> that used to be stopped now passes. Half-mitigated because `notebooks/visores/` is
> ignored, but that protects one specific path, not the pattern. Really closing it means
> looking **inside** the HTML — hunting base64 blocks with a mesh signature — and that is
> outstanding work.

**Change history**

| Date | Version | Change |
|---|---|---|
| 2026-08-05 | 0.1.0 | Initial registration: six checks, probe paths for the ignore rules, wired into the hook and both workflows. |
| 2026-08-11 | 0.1.1 | No behavioural change: the checks are now declared in a registry with stable identifiers, and this card reproduces it. |

---

### `docs-guardian` — Documentation ↔ code coherence guardian

| Field | Value |
|---|---|
| **Name** | `docs-guardian` |
| **Version** | `0.12.0` |
| **Location** | [`scripts/docs_sync.py`](scripts/docs_sync.py) |
| **Status** | `active` |
| **Trigger** | `pre-commit` hook (**does not block**) · `ai-code-review.yml` · `literature-watch.yml` |
| **Technology** | Typed code + `ast`, **no LLM** |

**Role / Purpose**

> Documentation drifts silently: nobody gets a red error for writing a number that
> stopped being true. This agent turns that into a visible failure.
> Real cases from this repository: a test count that said 166 when it was 265, a
> `.env.example` with five variables nobody read, and the README tree announcing a
> notebook that had already been deleted.

**It checks, it does not write** — that is its design rule and its boundary

> It only **generates** what is a mechanical copy of a source of truth (the tables
> between `<!-- generado: … -->` markers). Everything else it **verifies**. An agent
> that "fixed" the documentation on its own would make the document adapt to the code
> **even when it is the code that is wrong**: the 166 would have become 265 without
> anyone finding out it had been lying for months.

| Check | What it catches |
|---|---|
| `env` | variables read by the code ↔ `.env.example` ↔ README |
| `rutas` | files cited in the documentation ↔ versioned files |
| `agentes` | the classes' `name` attribute ↔ the `AGENTS.md` registry |
| `versiones` | the class's `version` attribute ↔ the **Version** row of its card |
| `guardianes` | scripts run by the workflows or the hooks ↔ a card in `AGENTS.md` |
| `comprobaciones` | each guardian's `COMPROBACIONES` registry ↔ the table on its card (this one) |
| `constantes` | a number or string written in the prose ↔ the value of the constant it comes from |
| `inventario` and `arbol` | that what exists is cited, not merely that what is cited exists |
| `vacios` | a component with a README card ↔ that it has code, or that the card declares it a placeholder |
| `bloques` | that the generated tables match the code |

> **Why the guardians are not versioned.** Giving them a `__version__` and comparing it,
> as is done with the `*Agent` classes, was considered. It was not done, for three
> reasons. An agent's `version` **is used**: it travels in `qualified` inside
> `provenance`, so a saved fusion says which formula computed it. In a script nobody
> would read it, and the check would end up comparing two constants nobody has any
> reason to touch: permanently green, worse than not checking. It would also force every
> guardian to have its own card with a version row, when `audit_pr.py` lives —
> correctly — inside the `ai-code-reviewer`'s. And SemVer with nobody importing the
> module means nothing. What does change, and what matters to a reader, is **what it
> checks**, and that is what gets compared.

**Delegation rules**

- **It does not block the commit.** A hook that stops you working over a documentation
  problem ends up uninstalled; if it fails, it warns, and CI says so on the PR.
- **Source of truth = `git ls-files`**, not the disk: otherwise an ignored file that
  exists locally would make something pass on your machine that fails in CI.

**How a number is tied to its constant.** `constantes` does not guess: it needs a
marker, in an HTML comment invisible when rendered.

```markdown
...four orders of magnitude below the budget of **0.1 mm** <!--const:REVERSIBILITY_BUDGET_MM-->
```

From then on, if someone changes `REVERSIBILITY_BUDGET_MM` in the code and does not touch
this sentence, CI says so with the file and the line. There are 13 marked numbers today.

> **Why a marker rather than searching the text for the constant's name.** Because the
> dangerous prose **does not name it**: it says "ε = 0.5 mm" or just "< 0.1 mm". Of the
> 25 places in the repository where a number comes from the code, only 3 cited the
> constant — a name-based check would have watched 12% while giving the impression of
> covering everything. The marker inverts the burden: whoever writes the number declares
> where it comes from, once, and CI watches it forever.

> ⚠️ **What is still uncovered.** A prose claim **with no number** that becomes false.
> The fusion card said "the coarse stage is still pending" long after it stopped being
> so, and no constant backs that up. The other half of that same case — "ε = 0.5 mm" —
> has been covered since 0.4.0.

**Change history**

| Date | Version | Change |
|---|---|---|
| 2026-08-04 | 0.1.0 | Initial registration: env, paths, agents, inventory, tree and generated blocks. |
| 2026-08-10 | 0.2.0 | **Version** check (the card declares what the class declares) and **guardian** check (a script that runs by itself needs a card) — this one. |
| 2026-08-11 | 0.3.0 | **Check-of-checks**: each guardian's `COMPROBACIONES` registry against the table on its card. |
| 2026-08-17 | 0.4.0 | **Constants** check: numbers cited in the documentation against the real value in the code, tied together with a `<!--const:NAME-->` marker. It was the extension this very card declared pending. |
| 2026-08-18 | 0.5.0 | `constantes` accepts **strings**, not only numbers. `SCHEMA_VERSION` asked for it: the README announced schema `1.2.0` while the contract was already at `1.3.0`, and the check stayed green because "1.2.0" is not a number. |
| 2026-08-24 | 0.6.0 | `versiones` finds the card by its **subject** rather than by any mention. The `export-agent` card names `render-export-agent` in its channel table and comes earlier, so the check was validating against the neighbour's version: invisible as long as everyone declared `0.1.0`. Along the way, the **Version** field of this card, which had stayed at `0.3.0` with the history already at `0.5.0`. |

---

### `literature-watcher` — Scientific literature watcher

| Field | Value |
|---|---|
| **Name** | `literature-watcher` |
| **Version** | `0.2.0` |
| **Location** | [`scripts/watch_literature.py`](scripts/watch_literature.py) + [`.github/workflows/literature-watch.yml`](.github/workflows/literature-watch.yml) |
| **Status** | `active` |
| **Trigger** | `cron` Mondays 06:00 UTC · `workflow_dispatch` |
| **Technology** | Typed code against arXiv's API and OAI-PMH, **no LLM** |

**Role / Purpose**

> The `research-agent` already knows how to search literature, but it is a REPL: it only
> discovers while somebody is sitting in front of it. This agent does the repetitive
> part — look at what came out, discard what is already inventoried, and find out under
> which licence it was published — and leaves the judgement *(is it relevant?)* where it
> belongs: with a person reviewing a PR.

**What it does, and its boundaries**

- **Seven queries** with a per-query topical gate: four dental (3DGS, CBCT +
  segmentation, intraoral scanner, digital twin) and three on standards (DICOM,
  FHIR/HL7, interoperability) — the latter filtered **by title only**, because including
  the abstract let false positives through.
- **A quota per round**, not "the N newest": the standards queries produce far more
  volume, and sorting by date left PRs without a single dental paper.
- **No PDF touches the disk or the repository.** It is downloaded into memory to compute
  `sha256` and size, then released. Downloading is not redistributing.
- **The licence is verified at the source** (OAI-PMH, `verb=GetRecord`), not assumed. A
  licence guessed from the title is worse than none, because it looks like data.

**Delegation rules**

- **It does not merge.** Its output is a branch and a PR; the decision is human, and that
  PR goes through the same guardians as any other.
- **"There is nothing" and "nobody answered" are different outcomes**, with different
  exit codes (`0` and `2`). A watcher that exists so nobody has to remember to look
  cannot report being broken by staying quiet.
- It retries only what is transient (429, 5xx, network drops): a 400 from a malformed
  query does not heal by waiting, and giving it three passes only delays the diagnosis.
- If the organisation forbids Actions from opening PRs, it **does not fail**: it leaves
  the branch pushed and the PR body in the run summary, ready to paste.

**Change history**

| Date | Version | Change |
|---|---|---|
| 2026-08-05 | 0.1.0 | Initial registration: queries with a topical gate, licence verification at the source, per-round quota and PR opening. |
| 2026-08-10 | 0.2.0 | First real cron firing. Retries with growing backoff, a distinction between "no news" and "no query answered", declaration of partial failures in the PR, and tolerance for the organisation blocking the automatic PR. |

---

### Ingestion agents — `mesh-agent` · `cbct-agent` · `report-agent`

| Field | Value |
|---|---|
| **Location** | `packages/ingestion-agents/` (`mesh_agent.py` · `cbct_agent.py` · `report_agent.py`) |
| **Version** | `0.1.0` |
| **Status** | `active` |
| **Pipeline phase** | 1 · Ingestion (the raw → contract boundary) |
| **Common contract** | `IngestionOutput` + `BaseIngestionAgent` in `ingestion_agents/base.py` |
| **Orchestrator** | `apps/agent-orchestrator` (`IngestionPipeline`) |

> **Why they live in `packages/` and not in `apps/<modality>-agent/`:** the
> orchestrator (an `app`) has to import them, and the `ai-code-reviewer` forbids
> cross dependencies between `apps/`. The monorepo rule forces shared code into
> `packages/` — it is the "or `packages/` if several share it" branch of the
> `add-ingestion-agent` skill.

**Role / Purpose**

> They translate **one** raw clinical file into a fragment of the `core-schemas`
> contract, declaring its `Provenance`. The rule: **1 modality = 1 support = 1
> agent**. They are the **only** components that touch raw files: from the
> `TwinSnapshot` onwards nobody goes back to the original.

| Agent | Input | `Modality` | `Support` | Produces | Brain |
|---|---|---|---|---|---|
| `mesh-agent` | intraoral OBJ / STL | `mesh` | surface | `surface_ref` (float64 positions + faces + normals + colour) | deterministic |
| `cbct-agent` | DICOM series directory | `cbct` | volumetric | `gaussian_field_ref` (seed σ field) | deterministic |
| `report-agent` | PDF / TXT / MD | `report` | regional | `list[RegionalObservation]` (pH, root anatomy and findings per FDI) + `list[Medida]` | deterministic (`rules`) · optional LLM (`llm`), **same fields** |
| `image-agent` (PoC) | JPG / PNG / HEIC photo | `image` | surface | `artifact_ref` (RGB pixels, **no EXIF**) | deterministic |

**Tools and permissions** (typed code, **no** MCP and no tool calling)

| Resource | Permissions | Notes |
|---|---|---|
| The modality's raw file | read | The system's only read of raw data. |
| `ArtifactStore` (`data/interim/artifacts/`) | write | Heavy blobs by SHA-256 content hash; never embedded in Pydantic. |
| Quarantine directory | write | Only the path and the failure traceback; **never** clinical content. |
| Anthropic API | network (only `report-agent` with `backend="llm"`, **off by default**) | Requires `ANTHROPIC_API_KEY`; without it the agent fails by declaring, it does not raise. |

**Generated outputs**

```
IngestionOutput
  ├─ ingestion : ModalityIngestion (ok/missing/failed) — ALWAYS present
  ├─ provenance: Provenance (source_file, modality, agent, confidence)
  ├─ artifact_ref / n_primitives   (mesh, cbct)
  ├─ regional  : list[RegionalObservation]  (report)
  └─ latency_s, quarantine_ref
```

**Delegation rules**

- They do not delegate to each other and decide nothing: they produce contract
  fragments. The `agent-orchestrator` fires them **in parallel** (the three
  modalities are independent) and assembles the `TwinSnapshot`.
- **Fail-loud, never fail-fast**: a corrupt file returns `status=FAILED` +
  `detail`; it never propagates the exception. One modality failing cannot take the
  other two down with it.
- **Human-in-the-loop**: the agent does **not** decide what gets persisted. It emits
  `Provenance.confidence` and the orchestrator applies the threshold
  (`DEFAULT_HITL_THRESHOLD = 0.7`); below it, the snapshot needs human review before
  being persisted.
- **Data sovereignty**: the `cbct-agent` pseudonymises the DICOM `PatientID`
  (HMAC-SHA256 with a salt from `ASH_PSEUDONYM_SALT`); no direct identifier reaches
  the contract.

**Per-modality rules**

- 🔒 `mesh-agent` — **reversibility guardrail**: it preserves the source surface
  losslessly (`float64` positions + full topology), not a resampled cloud. Round trip
  with **zero** error (test `test_round_trip_de_superficie_sin_perdida`). It accepts
  **OBJ** (per-vertex colour, Teeth3DS+) and **STL** (always bare → `color=None`,
  Bite2Text); the uniform grey of the Teeth3DS+ OBJ is an exporter *placeholder* and is
  likewise treated as **absence** of colour.
- `cbct-agent` — it **wraps** the RGS-style reconstruction, it does not reimplement its
  residual algorithm. It produces the isotropic seed (normalised σ, identity
  quaternion) that an optimiser would refine.
- `report-agent` — it validates every value against the **minimal clinical ontology**
  (`ingestion_agents/ontology.py`) before writing it into the contract.

**Change history**

| Date | Version | Change |
|---|---|---|
| 2026-07-22 | 0.1.0 | Initial registration: the three ingestion agents go from `planned` to `active`. Common `IngestionOutput` contract, content-addressed store, quarantine, pseudonymisation, minimal ontology, synthetic case generator and phase 1 orchestration. |

---

### Fusion agents — `geometric-fusion-agent` · `semantic-fusion-agent`

| Field | Value |
|---|---|
| **Location** | `packages/fusion-agents/` (`geometric.py` · `semantic.py` · `registration.py` · `twin.py`) |
| **Version** | `geometric-fusion-agent` **0.2.0** · `semantic-fusion-agent` `0.1.0` |
| **Status** | `active` |
| **Pipeline phase** | 2 · Geometric fusion **and** 4 · Semantic fusion (separated by segmentation) |
| **Common contract** | `FusionOutput` + `BaseFusionAgent` in `fusion_agents/base.py` |
| **Orchestrator** | `apps/agent-orchestrator` (`IngestionPipeline.fuse()`) |
| **Decisions** | [ADR 004 — Fusion](docs/architecture/004-fusion.md) |

> **Why they are two agents and not one:** **segmentation** runs between them, so a
> single agent would have to be invoked twice with flags and keep state across calls.
> They also have different material and different acceptance criteria: the semantic one
> is validated against the report, the geometric one against CBCT+IOS pairs.

**Role / Purpose**

> They enrich an **already assembled** `TwinSnapshot`: they touch no raw files and never
> go back to the original. The geometric one aligns two measurements of the same physical
> object and leaves an **invertible** record of the transform; the semantic one hangs
> observations off the correct tooth and **flags what does not add up** rather than
> deciding.

| Agent | Input | What it produces | What it does not do | Brain |
|---|---|---|---|---|
| `geometric-fusion-agent` | `TwinSnapshot` + two `(N,3)` clouds in mm | `Provenance.transform` (invertible `RigidTransform`) + confidence from the residual + **per-Gaussian colour** from the mesh | does not read or write `region_id`; does not use the photos (non-rigid error) | deterministic |
| `semantic-fusion-agent` | `TwinSnapshot` + `detected: FDI → confidence` | `RegionalObservation`s anchored with the propagated confidence | touches no geometry and transforms nothing | deterministic |

**Tools and permissions** (typed code, **no** MCP and no tool calling)

| Resource | Permissions | Notes |
|---|---|---|
| `TwinSnapshot` (in memory) | read | They **never** go back to the raw file: ingestion is the only boundary with original data. |
| Quarantine directory | write | Only `acquisition_id` + traceback; **never** clinical content (not even the pH). |

> They do not use the `ArtifactStore`: `detected` is passed **explicitly** rather than
> reading the Gaussian field from the store. These are ~14 FDI codes — loading millions
> of primitives to obtain them would be absurd, and this way the agents are testable
> without a store and without a GPU.

**Generated outputs**

```
FusionOutput
  ├─ status      : ModalityStatus (ok/missing/failed) — ALWAYS present
  ├─ snapshot    : TwinSnapshot | None  — NEW, never the input mutated
  ├─ hitl_reasons: list[str]            — empty = no review needed
  └─ latency_s, quarantine_ref, detail
```

**Delegation rules**

- **Fail-loud, never fail-fast**: the failure is returned as `status=FAILED` +
  `detail`; it never propagates the exception. The orchestrator keeps the ingestion
  snapshot — fusion failing does not destroy what ingestion did achieve.
- **Human-in-the-loop**: the agent does **not** decide what gets persisted. It emits
  `hitl_reasons` and the orchestrator applies the threshold (`DEFAULT_HITL_THRESHOLD = 0.7`).
- **They never mutate** the input snapshot and **preserve the `acquisition_id`**: that is
  the visit's identity, and it is what makes re-running fusion **replace** rather than
  inflate the patient's history with visits that never happened
  (`insert_snapshot`, ADR 004 §2.5).

**Per-stage rules**

- 🔒 `geometric-fusion-agent` — **reversibility guardrail**: the transform is stored as a
  `RigidTransform` (quaternion + translation), **not** as a 4×4 matrix. A 4×4 can encode
  scale and shear; if an ICP returned a spurious scale, reversibility would break
  silently. This form makes that *impossible to express*, and a validator rejects a
  non-unit quaternion.
- ⚠️ `geometric-fusion-agent` — **confidence comes from the residual of the overlapping
  population**, not of the whole cloud: `clamp(1 − rms_overlap/ε, 0, 1)`. Measured on a
  real patient with CBCT and scanner ([`scripts/registro_ios_cbct.py`](scripts/registro_ios_cbct.py)):
  **4.98 mm** over the entire cloud against **0.452 mm** over the points that do have a
  counterpart, for the same registration. The first does not measure the registration, it
  measures what fraction of the scan is palate. And below `min_overlap` (20%) the
  confidence is **0**: with four matched points there is always a pose that brings them
  together.
- ⚠️ `geometric-fusion-agent` — **ε is per modality pair**, not a constant.
  `clamp(1 − rms/ε) ≥ 0.7` is equivalent to `rms ≤ 0.3·ε`, i.e. 0.15 mm with ε = 0.5 — <!--const:DEFAULT_HITL_THRESHOLD--> <!--const:DEFAULT_EPSILON_MM-->
  below the physical floor of a CBCT with 0.30 mm voxels and a 425 µm PSF, so with that
  value intraoral↔CBCT fusion **could never pass the gate**. ε reads as *the error beyond
  which the result stops being useful*: **0.5 mm** for a mesh derived from the volume <!--const:DEFAULT_EPSILON_MM-->
  itself, **1.5 mm** (`EPSILON_IOS_CBCT_MM`) for intraoral↔CBCT. Neither is the brief's
  0.1 mm metric: that one measures the reversibility of *one* mesh, these measure <!--const:REVERSIBILITY_BUDGET_MM-->
  alignment between *two* modalities.
- `geometric-fusion-agent` — the algorithm lives behind a `Protocol` (`Registrar`), with
  two implementations: `icp` (the **fine** stage, with optional outlier trimming) and
  `icp_global` (the **coarse** stage, an SO(3) sweep). The coarse stage closes the hole
  ADR 004 left for RANSAC-FPFH, by brute force and without dragging in Open3D. Without
  it the fine ICP **does not blow up**: it converges to a local minimum of ~0.43 mm — a
  figure that looks like a good registration — with the wrong pose. It is a failure you
  have to go looking for. **Trimming and searching fight each other**: aggressive
  trimming makes a bad pose score well, so we sieve loosely and refine hard.
- `geometric-fusion-agent` —
  **It transfers colour from the mesh** (ADR 004 §2.8): each Gaussian within the ε band
  takes the colour of its nearest vertex. The **photos stay out** — notebook 07 measured
  that the photo↔mesh error is **non-rigid** (ICP stalled at IoU ≈ 0.55), so it is not the
  same problem as rigid registration. With a bare mesh or a grey *placeholder* the result
  is **absence of colour**, which is a valid answer and not a bug.
- `semantic-fusion-agent` — confidence is the **weakest link**,
  `min(observation_confidence, FDI_confidence)`: anchoring a pH to a tooth cannot be more
  reliable than knowing which tooth it is.
- ⚠️ `semantic-fusion-agent` — **on an FDI conflict it picks no winner**. If the report
  references a tooth that segmentation did not find, it keeps the FDI *from the report*
  (the clinical source), sets confidence to **0.0**, and with that it falls to the gate
  and goes to human review. The reason is **measured**: the model's dominant error is a
  shift to the neighbouring tooth ([Point Transformer
  experiment](notebooks/exercise-point-transformer-teeth3ds.md)), so that is the least
  reliable part — but the report is not infallible either. Resolving it silently would be
  the failure [ADR 003](docs/architecture/003-verification-fault-tolerance.md) singles out
  as the worst: silent and irreversible, on clinical data.
- `semantic-fusion-agent` — each observation's `Provenance` **keeps the report's**
  (`source_file`, `modality`, `agent`); only `confidence` is rewritten. The value still
  comes from the PDF. Who performed the fusion is recorded in the snapshot's `Provenance`.

**Change history**

| Date | Version | Change |
|---|---|---|
| 2026-08-04 | 0.1.0 | Initial registration. ADR 004; `RigidTransform` in `core-schemas` (contract 1.3.0); complete semantic fusion; geometric fusion with multiscale ICP (coarse stage missing); idempotent insertion into the time series; wired into the orchestrator. |
| 2026-08-10 | 0.2.0 (geometric) | Registration measured against a real patient with CBCT + scanner ([`scripts/registro_ios_cbct.py`](scripts/registro_ios_cbct.py)). Confidence now comes from the **overlapping population**, ε becomes **per modality pair** (0.5 mm volume-derived · 1.5 mm intraoral↔CBCT), a **minimum overlap** gate is added, and the **coarse stage** (`icp_global`) ADR 004 left pending is implemented. Additive: a registrar that does not measure overlap behaves exactly as before. |

---

### `segmentation-agent` — Anatomical segmentation agent

| Field | Value |
|---|---|
| **Location** | `packages/analysis-agents/` (`segmentation.py` · `base.py`) |
| **Version** | `0.1.0` |
| **Status** | `active` |
| **Pipeline phase** | 3 · Analysis · segmentation (**between** the two fusion stages) |
| **Common contract** | `AnalysisOutput` + `BaseAnalysisAgent` in `analysis_agents/base.py` |
| **Orchestrator** | `apps/agent-orchestrator` (`IngestionPipeline.fuse()`, stage 2 of 3) |
| **Point → tooth aggregation** | `packages/tooth-aggregation` |

**Role / Purpose**

> It puts a **tooth name** on every Gaussian. It is the pipeline's **semantic
> anchor**: without `region_id`, semantic fusion has nothing to validate the tooth the
> report cites against, and the regional layer (pH and the rest) hangs off a code
> nobody has confirmed exists in that mouth.

| Input | What it produces | What it does not do | Brain |
|---|---|---|---|
| `TwinSnapshot` + the Gaussian field it references | `region_id` (FDI) per Gaussian in a **new** artifact + `detected: FDI → confidence` + review reasons | it does not correct the report or vice versa (semantic fusion declares that); it neither trains nor runs the model | a point-cloud segmentation model, **injected** |

**Tools and permissions** (typed code, **no** MCP and no tool calling)

| Resource | Permissions | Notes |
|---|---|---|
| `TwinSnapshot` (in memory) | read | Never goes back to the raw file: ingestion is the only boundary with original data. |
| Artifact store | read + write | Through the `GaussianStore` `Protocol`, **not** through `ArtifactStore`: the store is a *seam* that `3dgs-engine` will replace. |
| Segmentation model | call | Through the `Segmenter` `Protocol`. **No default**: a toy model by omission would produce invented anatomical labels that look entirely convincing. |
| Quarantine directory | write | Only `acquisition_id` + traceback; **never** coordinates or clinical content. |

**Generated outputs**

```
SegmentationOutput
  ├─ status              : ModalityStatus (ok/missing/failed) — ALWAYS present
  ├─ snapshot            : TwinSnapshot | None  — NEW, with a labelled gaussian_field_ref
  ├─ detected            : {FDI: confidence}    — the semantic-fusion-agent's exact input
  ├─ n_teeth             : int
  ├─ unassigned_fraction : float                — fragmentation discarded by size
  ├─ hitl_reasons        : list[str]            — empty = no review needed
  └─ latency_s, quarantine_ref, detail
```

**Delegation rules**

- **Fail-loud, never fail-fast**: the failure is returned as `status=FAILED` +
  `detail`. If segmentation fails, the orchestrator **does not anchor** the report
  against an anchor that does not exist: semantic fusion simply does not run.
- **Human-in-the-loop**: a `detected` passed to the orchestrator by hand **overrides
  the model** and the stage does not even execute. That is how clinician-reviewed
  labels get in; without that door, the review gate would be pointless.
- **It never mutates** the input snapshot and **preserves the `acquisition_id`**.

**Specific rules**

- 🔒 **The label is additive.** The new artifact carries the same
  `centers`/`scales`/`rotations`/`density` **byte for byte** plus a `region_id` array
  (`0` = unassigned). The previous blob stays in the store, so segmenting cannot degrade
  the geometry or break reversibility. Because the store is content-addressed,
  re-segmenting the same field with the same model returns **the same reference**.
- ⚠️ **It is verified that the model returns log-probabilities, not logits.** A logit has
  the same shape and the same `argmax`: the labels would come out fine and the
  **confidences** would be false, with nothing screaming. It is the same expensive failure
  mode as the DICOM modality in the `cbct-agent`, and it is closed the same way — by
  verifying the premise (`∑ⱼ exp(logprob) = 1`) instead of trusting it.
- **A tooth's confidence is the geometric mean of the per-point probability**
  (`exp` of the instance's mean log-probability), in `[0, 1]` and comparable with the
  HITL threshold and with the rest of the pipeline.
- **The honest metric is per-TOOTH accuracy, not per point** — it is measured that
  per-point accuracy *underestimates* per-tooth identification ([Point Transformer
  experiment](notebooks/exercise-point-transformer-teeth3ds.md)).
- **`enforce_unique` ships disabled on purpose.** Imposing "one FDI per arch" over
  fragmented instances *invents* errors: the constraint presupposes "one instance = one
  tooth", and with fragments that premise is false (measured).
- **Four things go to human review** rather than resolving themselves: confidence below
  the threshold, the same FDI on two instances, a class with no FDI code in the mapping,
  and more than 10% of the tooth points discarded through fragmentation — the risk is
  that aggregation eats them **silently**, not that it eats them.
- ⚠️ **A gum island inside a crown is a hole, and it is closed by CONNECTIVITY.**
  Neighbourhood filling stops where the hole is larger than the neighbourhood, and in the
  viewer that is what shows most: colour is per vertex and gets interpolated, so one
  misplaced vertex stains the six triangles that touch it. An arch's gum is **a single
  connected region**, margin included; any other component is surrounded by tooth and that
  is not anatomy. The first attempt went by distance to "real gum" defined as being
  surrounded by gum, and **it is circular**: a large hole has its interior surrounded by
  itself and passes as good gum. On real data it slipped through — the worst kind of
  failure.
- ⚠️ **And the symmetric case: a tooth vertex surrounded by gum is gum.**
  `rellena_etiquetas` only went one way — it gave a label to what had none and never took
  one away from what had it wrong — so after the whole pipeline **692 tooth vertices were
  left floating in the gum**. `quita_motas` cleans them with the same 0.85 threshold,
  precisely so as not to move the margin: a margin vertex has neighbours of both classes
  and never reaches it. And `rellena_etiquetas` now **iterates**, because closing one hole
  exposes the next: holes go 3,299 → 1,519 → 1,198 → 1,096 → 1,051 and converge.
- ⚠️ **The boundary between ADJACENT teeth comes out blurred, and is sharpened
  separately.** Interproximal contact is a continuous surface in the mesh, so the model
  has no geometric edge there to hold on to. Measured on a clinical case: **7.0% of the
  scan's area** (308 mm²) sat on faces whose vertices belonged to two different teeth, and
  the pairs were all neighbours — (26,27), (15,16), (21,22), (11,21). You see it when
  highlighting a tooth in the viewer: a piece of the one next door lights up too.
  `analysis_agents.afina_fronteras` votes by majority among neighbours and brings it down
  to **92 mm²**, reassigning 3,624 vertices. **It does not touch the crown/gum margin** —
  12.8% of the area — because that one is a clinical boundary and blurring it would erase
  what a periodontist comes to look at.
- ⚠️ **The scanner's labels are SEEDS, so a mislabelled island there becomes a whole
  tooth here.** Measured on a real case: the scanner's segmenter put a third molar's code
  on 275 vertices stuck to the distal face of its neighbour — median 0.85 mm — and that
  code grew towards the root and landed on the maxillary tuberosity. The case went from 14
  teeth to 15, and the ghost travelled to the viewer, the views and the clinical layer.
  `analysis_agents.absorbe_islas` kills it before seeding, and **leaves a record of every
  absorption**: a partially erupted third molar looks a lot like an island, so the
  decision goes to the gate and not to a log.

**Change history**

| Date | Version | Change |
|---|---|---|
| 2026-08-25 | 0.1.0 | `quita_motas` and iterative filling: tooth specks in the gum drop from 692 to 311 and gum holes in crowns from 3,299 to 1,048. No contract change. |
| 2026-08-25 | 0.1.0 | `afina_fronteras`: the boundary between adjacent teeth is no longer blurred (308 → 92 mm² of area with two teeth on the same face). And the per-tooth apical trim is withdrawn: measured, the axis taken from the CBCT comes out 60° off the real one because the cloud drags bone along. No contract change. |
| 2026-08-24 | 0.1.0 | `absorbe_islas`: tiny teeth embedded inside another stop seeding a ghost tooth in the CBCT. No contract change. |
| 2026-08-06 | 0.1.0 | Initial registration. `analysis-agents` package with its base contract; point → tooth aggregation over `tooth-aggregation`; `region_id` persisted as an additive artifact; wired into `IngestionPipeline.fuse()` between the two fusion stages. |

---

### Analysis agents (`planned` stubs)

Clinical pipeline agents **not yet implemented**. They are registered here as design
*stubs* (intended roles and contracts) to close Task 3 of the multi-agent architecture
issue. Their high-level design lives in
[`docs/architecture/multi-agent-pipeline.md`](docs/architecture/multi-agent-pipeline.md).
All of them **consume and enrich** a `TwinSnapshot` through `packages/core-schemas` —
they never go back to the raw file — and leave their own `Provenance`.

| Agent | Status | Phase | Intended role | Input → output | Human-in-the-loop |
|---|---|---|---|---|---|
| `pathology-agent` | `planned` | Analysis · clinical findings | Flag possible pathology (σ density, colour, geometry) as **candidate findings for clinical review**. | `TwinSnapshot` → `RegionalObservation` with candidate findings (not a diagnosis) | **Yes** — clinically sensitive |
| `clinical-poc-agent` | `planned` (PoC) | Analysis · proof of concept | A basic visual metric: inflammation from gum colour and the gum-tooth gap. | `TwinSnapshot` → text report (log) | Yes |

> **Design boundary:** these stubs have **no** MCP tools, permissions or final
> delegation rules yet; those will be detailed on implementation, each with its full
> card (like `research-agent`) and, where appropriate, its ADR. Registering them now
> fixes their **role and contract**, not their implementation.

> **Clinical and regulatory framing (important).** The human-in-the-loop analysis agents
> (`pathology-agent`, `clinical-poc-agent`, and any clinically sensitive measurement such
> as the gum↔bone **periodontal phenotype**) produce **candidate findings and
> measurements for clinician review**, with traceable `Provenance` — they **do not issue
> a diagnosis** and do not replace clinical judgement. This is **investigational /
> demonstrator use**, and keeping it that way leaves the system **outside** the medical
> device (SaMD) category. The moment an output is declared "diagnostic" or a clinician
> relies on it to treat, it enters regulated territory (EU **MDR** / **FDA**) — outside
> the scope of this phase.

**Ingestion agents:** `cbct-agent`, `mesh-agent` and `report-agent` are already
`active` — full card in the [previous section](#ingestion-agents--mesh-agent--cbct-agent--report-agent).
The `image-agent` (2D photo) is the **`IngestionPipeline`'s 4th modality**: it ingests
JPG/PNG/HEIC, **discards the EXIF** (privacy) and stores the pixels as an artifact.
It is the only **0..N** modality (one acquisition brings several photos, 5 in Bite2Text),
so the orchestrator ingests each photo and the `TwinSnapshot` collects them in
**`image_refs: list[str]`** (pre-fusion appearance, like `surface_ref`). It does not
reconstruct 3D from a photo — that is fusion — but it leaves the appearance ready and
traceable. Ingestion contract detail in the
[multi-agent pipeline](docs/architecture/multi-agent-pipeline.md#2-tarea-1--contratos-de-ingesta).

---

### `export-agent` — Export agent (mesh regeneration)

| Field | Value |
|---|---|
| **Location** | `packages/export-agents/` (`stl.py` · `base.py`) |
| **Version** | `0.1.0` |
| **Status** | `active` |
| **Pipeline phase** | 6 · Export (the contract → file boundary) |
| **Common contract** | `ExportOutput` + `BaseExportAgent` in `export_agents/base.py` |
| **Orchestrator** | `IngestionPipeline.exportar(result, destino)` fires the three channels |
| **Decisions** | [Pipeline §5](docs/architecture/multi-agent-pipeline.md#5-tarea-4--formato-y-pipeline-de-exportación) · [ADR 001](docs/architecture/001-digital-twin-core-schemas.md) (*fail-loud* refs) · [ADR 004 §2.2](docs/architecture/004-fusion.md) (invertible `transform`) |

> **Why it is a fourth family and not a loose function.** It is the **only** one that
> writes output files, just as ingestion is the only one that reads input files. And its
> output does not fit the others: `FusionOutput` and `AnalysisOutput` return an enriched
> `TwinSnapshot`, and an exporter **enriches nothing** — it returns a file and a
> measurement of how closely it resembles what went in.

**Role / Purpose**

> It closes the **reversibility** loop: ingestion turned the scanner's mesh into a
> contract and this agent returns it to a file, with the **reconstruction error
> measured** rather than promised. It is the phase that turns the brief's metric ("mesh
> error < 0.1 mm") into something checked on every run.

| Agent | Input | What it produces | What it does not do | Brain |
|---|---|---|---|---|
| `export-agent` | `TwinSnapshot` + output path | **Binary STL** from `surface_ref` + `max_deviation_mm` and `mean_deviation_mm` measured by re-reading the file | does not mesh the Gaussian field; does not modify the twin; does not write to the store | deterministic |
| `field-export-agent` | `TwinSnapshot` + output path | **Binary PLY** from `gaussian_field_ref`, in the twin's frame or in CBCT millimetres | does not invent colour or opacity; does not convert to INRIA's splat format | deterministic |
| `render-export-agent` | `TwinSnapshot` + directory | **Multi-view PNG** by Beer-Lambert + the cycle's `psnr_db` / `ssim` | does not rasterise splats; does not reproduce the intraoral photographs | deterministic |

> The **metadata channel** (the `TwinSnapshot` to JSON) deliberately has no agent: it is
> Pydantic's `model_dump()`. Wrapping a one-line call in a failure contract that cannot
> fail would be ceremony, not design.

**Tools and permissions** (typed code, **no** MCP and no tool calling)

| Resource | Permissions | Notes |
|---|---|---|
| `TwinSnapshot` (in memory) | read | Never goes back to the raw file. |
| Artifact store | **read** | Through the `SurfaceStore` `Protocol`, which **only declares `load`**: an exporter not writing to the store is not a promise in prose, it is something the type does not let you express. |
| Output file (STL) | write | **Atomic** write (temporary + `replace`): a half-written STL must not be left where someone could mistake it for the good one. |
| Quarantine directory | write | Only `acquisition_id` + traceback; **never** geometry or clinical content. |

**Generated outputs**

```
ExportOutput
  ├─ status          : ModalityStatus (ok/missing/failed) — ALWAYS present
  ├─ path            : Path | None      — the file written
  ├─ format · frame  : 'stl' · 'source' | 'twin'
  ├─ n_vertices · n_faces
  ├─ max_deviation_mm: float | None     — MEASURED, by re-reading the file (the brief's metric)
  ├─ hitl_reasons    : list[str]        — empty = no review needed
  └─ latency_s, quarantine_ref, detail
```

**Delegation rules**

- **Fail-loud, never fail-fast**: like the other three families, it does not raise. A
  dangling reference, an out-of-range face or an impossible destination come back as
  `status=FAILED` + `detail` and go to quarantine.
- **Human-in-the-loop**: the agent does **not** decide whether the file is delivered. It
  emits `hitl_reasons` (`DEFAULT_HITL_THRESHOLD = 0.7`) and the caller decides.
- **Read-only over the twin**: it does not mutate the snapshot or rewrite artifacts. If an
  export could rewrite the blob it exports, the faithful copy of the surface — the
  `mesh-agent`'s guardrail — would stop being faithful at the first round trip.

**Specific rules**

- 🔒 **Geometry comes from `surface_ref`, never from the Gaussian field.** The
  `mesh-agent` stores the source surface as-is (`float64` + full topology) so that a
  faithful copy exists; the round trip's only error is the one the **format** imposes
  (`float32`). Measured on a real Teeth3DS+ scan (110,804 vertices, an 86 mm arch):
  **3.8·10⁻⁶ mm**, four orders of magnitude below the budget, with the export taking
  0.07 s. Getting the mesh out of the volume by *marching cubes* would be **a different
  thing**: it is measured
  ([`scripts/resolucion_modalidades.py`](scripts/resolucion_modalidades.py)) that the
  isosurface is only well defined where the gradient is strong — enamel, 364 HU/voxel —
  and that over trabecular bone the area **does not exist** as a magnitude. A snapshot
  with no mesh declares itself `MISSING`; it is not "rescued" by interpolating.
- ⚠️ **The deviation is measured by re-reading the file, not by estimating.** An estimate
  of the `float32` error survives an endianness or vertex-order bug intact; a re-read does
  not. And with no measurement `within_budget` is `False`: an exporter that does not
  verify cannot claim it meets the 0.1 mm.
- **Two reference frames, made explicit.** `frame="source"` (the default) writes the mesh
  as it came in — that is the one that measures reversibility; `frame="twin"` applies the
  `RigidTransform` recorded by geometric fusion to superimpose it on the CBCT. Asking for
  `twin` on a snapshot that never went through fusion **fails by declaring**: the
  alternative would be handing over the mesh in the scanner's frame while passing it off
  as the CBCT's.
- **A partial snapshot does not reach export quietly** (ADR 001). The exporter is the last
  point where that can be said, and it says it twice: in `hitl_reasons` and **inside the
  file itself**, stamping `PARCIAL` into the STL's 80-byte header alongside the
  `acquisition_id`. That is traceability which survives someone copying the file out of
  the system, where there is no longer any `Provenance` to consult.
- **What the format cannot carry is declared, not faked.** The STL is "bare": no
  per-vertex colour, no per-vertex normals and no shared topology. The twin's
  `color_superficie` does not fit in the file and stays alive in `surface_ref`.

**Change history**

| Date | Version | Change |
|---|---|---|
| 2026-08-13 | 0.1.0 | Initial registration. `export-agents` package with its base contract (`ExportOutput` · `BaseExportAgent` · `SurfaceStore`), mesh regeneration to binary STL from `surface_ref`, verification by re-reading against the 0.1 mm budget, export in the scanner's frame or the twin's, and declaration of a partial snapshot in the file header. |
| 2026-08-17 | 0.1.0 | `mean_deviation_mm` alongside the maximum: it is the round trip's **Chamfer**, computed over the known correspondence (vertex *i* re-read against vertex *i*) and not by nearest neighbour, which would hide a permutation. And the deviation becomes a **per-vertex Euclidean distance** instead of a maximum per-coordinate error: a 1 mm diagonal displacement is √3 ≈ 1.73 mm, and measuring per coordinate reported it as 1.0. |

---

### `field-export-agent` — Gaussian field export

| Field | Value |
|---|---|
| **Location** | `packages/export-agents/` (`field.py` · `base.py`) |
| **Version** | `0.1.0` |
| **Status** | `active` |
| **Pipeline phase** | 6 · Export (the contract → file boundary) |
| **Common contract** | `ExportOutput` + `BaseExportAgent` |
| **Orchestrator** | `IngestionPipeline.exportar(result, destino)` |

**Role / Purpose**

> It materialises what the scanner **cannot see**: the interior the CBCT seeded.
> The mesh channel returns the measured surface; this one returns the volume.

**Specific rules**

- ⚠️ **The PLY is not a 3D Gaussian Splatting `.ply`, and that is deliberate.** INRIA's
  convention stores `opacity` and spherical harmonics `f_dc_*` — colour and transparency.
  This field has `density` (σₙ, Beer-Lambert attenuation) and a CBCT **does not measure
  colour**. Giving it that header would make it openable in any splat viewer, which would
  paint an invented colour over a physical magnitude: worse than not opening it. What
  colour a density field is painted in is a product decision and remains pending the
  render-engine ADR.
- 🔒 **`origin` and `hu_range` travel in the artifact, not in `Provenance`.** The
  `cbct-agent` centres the field (`centers -= mundo.mean(0)`) and normalises density to
  `[0, 1]`. Until 2026-08-17 the offset **was discarded**, which made the field
  irreversible: it depends on the data, so no version of the agent recomputes it. It was
  not put into `Provenance` because `transform` already means geometric fusion's alignment
  (ADR 004) and reusing it would collide, and because the schema declares the snapshot
  **self-contained**: what makes a blob reversible travels with it.
- **Asking for `frame="cbct"` on an artifact with no `origin` fails by declaring** and
  says a re-ingestion is needed. Handing it over centred while passing it off as CBCT
  coordinates would displace everything measured on top of it, and look fine doing it.
- **Positions are written in `double`, not `float32`.** That way `max_deviation_mm` comes
  out **exactly 0** and the verification measures *bugs* — endianness, property order, a
  miscomputed stride — rather than the format's rounding, which would hide a small one. A
  conventional splat uses float32; here the measurement is what matters.
- **Both frames come from the data, not from the caller.** Unlike `frame="twin"` in the
  mesh channel — which applies `Provenance.transform`, written by registering against
  whatever the caller passes as `target`, with the pipeline not fixing that frame — here
  both are defined by arrays in the artifact itself. Closing that hole belongs to the
  inter-agent protocol, not to an exporter.
- **`region_id` is written if the field comes segmented, and only then.** It is the
  per-Gaussian FDI label the `segmentation-agent` produces, i.e. the only thing the
  pipeline knows about anatomy; dropping it on export left the PLY with correct geometry
  and **mute about its content**, forcing a re-segmentation of something already computed.
  A column of zeros is not written when it is missing: in the agent's convention `0` means
  "unassigned", which is a different claim from "nobody has segmented it yet".

**Change history**

| Date | Version | Change |
|---|---|---|
| 2026-08-17 | 0.1.0 | Initial registration. Out of `planned`: binary PLY with the properties the field **has**, two frames (`twin` centred / `cbct` in real millimetres), reversibility measured by re-reading, and `densidad_a_hu` to undo the normalisation. It required the `cbct-agent` to start storing `origin` and `hu_range`. |
| 2026-08-17 | 0.1.0 | `region_id` (optional `short` property) travels to the PLY when the field is segmented. It surfaced when segmentation was tied into the end-to-end run: both stages passed their tests separately and the label was lost exactly between them. |

---

### `render-export-agent` — Multi-view render of the field

| Field | Value |
|---|---|
| **Location** | `packages/export-agents/` (`render.py` · `base.py`) |
| **Version** | `0.1.1` |
| **Status** | `active` |
| **Pipeline phase** | 6 · Export (the contract → file boundary) |
| **Common contract** | `ExportOutput` + `BaseExportAgent` |
| **Orchestrator** | `IngestionPipeline.exportar(result, destino)`; `render=False` skips it |

**Role / Purpose**

> It turns the field into something a person can **look at and approve**. It is the
> channel that makes the twin's interior reviewable, and the one that closes the brief's
> metric for what is not measured in millimetres.

**Specific rules**

- ⚠️ **These are not the intraoral photographs.** They are renders of the twin. Only
  appearance samples of the originals were kept (`image_refs`), so there is nothing to
  compare them against and we do not pretend there is: what is measured is the *twin →
  file → render* cycle, with `psnr_db` and `ssim`.
- **It does not rasterise splats: it composites by Beer-Lambert.** A 3DGS rasteriser
  mixes colour with `alpha blending`, which depends on order. `density` **is not
  opacity**, and its corresponding integral, `I = exp(−∫σ ds)`, is additive in optical
  depth and therefore **order-independent**. That is not about efficiency: it makes the
  render deterministic without depth sorting, and turns it into a synthetic radiograph
  rather than an invented photograph.
- ⚠️ **Mass is deposited, not amplitude.** Evaluating τ at the pixel centre — the obvious
  approach — produces aliasing, not an image: with `s = 0.15 mm` and ~0.7 mm pixels the
  centre falls more than 4σ from almost every Gaussian. And clamping σ to half a pixel to
  cover that inflates the amplitude: measured on `histora`, peak τ went from 34 at 256 px
  to **226** at 128 px, i.e. the image depended on detector size. Mass is deposited with a
  normalised profile and divided by the pixel area; there is a regression test.
- **Views are named by angle, never by anatomy** (`az090_el+00`). An axis's anatomical
  meaning depends on how the scanner writes the DICOM, and in this project assuming it
  rather than reading it went wrong three times on the same patient. A name like
  `az090_el+00` cannot lie; `occlusal` can.
- ⚠️ **The PLY's frame is READ from its header, not inferred from the data.** The verifier
  re-centred by subtracting the cloud's centroid, which was exact *while* `origin` was the
  mean — the `cbct-agent` writes it that way. The `gaussian-engine` fits ellipsoids to the
  density and **moves** the centroid: measured, `[0.04, −1.84, 3.97]` mm. That harmless
  subtraction became a 4.4 mm translation over an 87 mm scene and the cycle fell to
  **14.3 dB** against a budget of 40, silently and with good-looking images. Now
  `comment frame twin|cbct` is read from the header, which the exporter already wrote and
  nobody read; a PLY that does not declare its frame is an **error**, not a default case.
  It is the same rule as the view names: what the file says is not assumed.
- **The framing is common to every view.** If each image picked its own, two renders of
  the same field would not be comparable pixel by pixel and SSIM would be measuring the
  framing; besides, one view could crop what another shows.
- **Both PSNR and SSIM are returned, not one.** They measure different things: PSNR
  averages per-pixel error and is fooled by a global brightness shift, SSIM compares local
  structure. Measured on a smooth image, a 10-level bias and noise of the same energy give
  almost identical PSNR and SSIM of 0.99 versus 0.57.

**Change history**

| Date | Version | Change |
|---|---|---|
| 2026-08-17 | 0.1.0 | Initial registration. Multi-view Beer-Lambert render, byte-for-byte reproducible, with the cycle's PSNR/SSIM against the exported PLY and budgets `RENDER_PSNR_BUDGET_DB = 40` / `RENDER_SSIM_BUDGET = 0.99`. | <!--const:RENDER_PSNR_BUDGET_DB--> <!--const:RENDER_SSIM_BUDGET-->
| 2026-08-24 | 0.1.1 | The verifier **reads** the frame from the PLY header instead of re-centring by the centroid. The inference was only exact while `origin` was the mean; with the fitted field the cycle fell to 14.3 dB. Verified on the real case: from 14.3 dB to **exact**. |

---

### `composite-export-agent` — CBCT teeth + scanner gum

| Field | Value |
|---|---|
| **Location** | `packages/export-agents/` (`compuesto.py` · `base.py`) |
| **Version** | `0.1.0` |
| **Status** | `active` |
| **Pipeline phase** | 6 · Export (the contract → file boundary) |
| **Common contract** | `ExportOutput` + `BaseExportAgent` |
| **Orchestrator** | `IngestionPipeline.exportar(...)`; with no scanner or no registration only the field comes out, and it says so |

**Role / Purpose**

> The model the project is after: each modality contributes the one thing it knows how to
> measure and none is worth anything alone. The **CBCT** sees below the gingival margin —
> it is the only thing that gives the root — but its soft-tissue surface is noise at
> 0.30 mm voxels; the **intraoral scanner** measures gum to tens of microns and sees
> nothing below the margin. Separately they are a tooth with no gum and a gum with
> truncated teeth.

**Why it is needed and why what existed was not enough.** The `field-export-agent`'s PLY
already carried the teeth with their `region_id`, so it looked done. It was not: **the gum
was never written**. The script that claimed to produce it counted the gum vertices,
announced them in a comment inside the file itself — `compuesto dientes-CBCT + encia-IOS` —
and then emitted `element vertex 498407`, exactly the size of the CBCT field and not one
point more. The comment was true about the intention and false about the content.

**Specific rules**

- ⚠️ **The gum enters with `density = 0.0`, and that is a declaration, not a value.** A
  scanner vertex carries no radiological attenuation: the IOS measures shape, not density.
  Giving it a plausible σ would make it indistinguishable from a Gaussian measured in the
  CBCT, and anyone later projecting the field would be integrating a number nobody
  measured.
- ⚠️ **Every Gaussian declares its `origen`** (`0` = CBCT, measured density; `1` = IOS,
  measured shape). A composite that does not distinguish its two halves lies by omission,
  and this is what lets the viewer channel split them into layers.
- **The gum's scale comes from the mesh's real spacing**, not from a constant: an
  intraoral scanner has a very different resolution from the CBCT, and using the volume's
  σ would give a gum that is either pixelated or bloated. It is measured as the median
  nearest-neighbour distance, which is that cloud's natural σ.
- **What was just written gets re-read**, like the other channels, and here with more
  reason: the composite mixes two sources with different frames, so a fit error between
  them is exactly what a round-trip number catches and a visual inspection does not.

**Change history**

| Date | Version | Change |
|---|---|---|
| 2026-08-18 | 0.1.0 | Initial registration. CBCT + IOS composite in one PLY, with `origen` per Gaussian, `density = 0` declared for the gum and its σ measured from the mesh's spacing. |

---

### `composite-mesh-export-agent` — The printable arch and one tooth per file

| Field | Value |
|---|---|
| **Location** | `packages/export-agents/` (`malla_compuesta.py` · `solido.py` · `anatomia.py` · `compuesto.py` · `stl.py` · `base.py`) |
| **Version** | `0.3.0` |
| **Status** | `active` |
| **Pipeline phase** | 6 · Export (the contract → file boundary) |
| **Common contract** | `ExportOutput` + `BaseExportAgent` |
| **Orchestrator** | `IngestionPipeline.exportar(..., etiquetas_ios=...)`; with no scanner, no registration or no segmentation it returns `MISSING` and says which one is absent |

**Role / Purpose**

> This is the exporter that justifies the twin, which is why it is worth saying first what
> it is **not**. The other channels return what came in: the `export-agent` reconstructs
> the STL from `surface_ref` with format-level error, and that proves the container is
> **honest**, not that the twin is useful. A clinician who only wants to print the arch
> they scanned already has their file. What they do not have, and no machine gives them,
> is **a tooth with its root**.

**Two outputs, and the difference between them is a clinical decision, not a technical
one.** The **arch** comes out as the scanner measured it, **without roots and closed as a
solid with a flat base**: a study model or a guide sits on a base, and fifteen dangling
roots prevent that. Each **tooth** goes in its own file with a measured crown and a
reconstructed root, which does get printed — planning a complex extraction, an impacted
canine, a teaching model — and rests on the crown.

**Specific rules**

- ⚠️ **The arch is CLOSED, because an intraoral scan is a shell** — it measures the
  surface the camera sees, with no interior and no floor. It looks fine on screen (it bit
  us once already: the palate came out as a black hole because only the front face was
  painted) and it does not print, because a slicer needs to know what is inside. See
  `solido`.
- ⚠️ **The base is perpendicular to the OCCLUSAL axis — the direction the tooth bites in —
  and goes to the opposite end.** Not to the file's Z, which is whatever the machine had
  and means nothing; and above all **not to the superior axis**: in a maxilla they are
  opposites, and confusing them gives you not a tilted model but an inverted one. Measured:
  with the superior axis the plane fell at −9.82 mm when the crowns reached −7.82, so the
  skirt came down in front of the teeth and wrapped them in a box — 21.7 cm³ of volume
  against the correct 11.9. The axis is measured from the scan's own FDI codes, and
  **without them the shell is delivered with a declaration** rather than inventing a
  vertical.
- ⚠️ **A body with no ENAMEL is not a tooth, however much it looks like one.** The
  pipeline once exported a "28" on a real case with a tooth's size (19.4 mm), a tooth's
  shape and the same separation from its neighbour as two adjacent molars: it was the
  maxillary tuberosity. What gives it away is density — the fourteen real teeth gave
  p95 = 1.000 exactly and the intruder 0.747, a hair above the 0.710 of unlabelled bone.
  It is checked **before** reconstructing: reconstructing first would produce an STL with
  a tooth's name made of jaw.
- **It declares rather than trims when a root is longer than its type admits.** Trimming
  would turn the apex into an assumption, and measuring root length on top of it would be
  measuring what was assumed. The warning is grouped into one line rather than twelve:
  noise is how a gate gets disabled.
- ⚠️ **A printed tooth carries no header, and no file fixes that.** Everything this agent
  declares about the reconstructor disappears the moment the object leaves the printer:
  whoever holds it sees a root without knowing it is inferred. That is why reconstructed
  geometry goes ONLY into the per-tooth files, which somebody asks for deliberately, and
  never into what gets printed by default.
- ⚠️ **It is not marching cubes, and not out of preference.** The twin's field is a cloud
  of Gaussians with covariance, not a voxel volume: meshing by isosurface would require
  rasterising it to a grid first, i.e. throwing the representation away. And it is measured
  in `scripts/resolucion_modalidades.py` that over trabecular bone the isosurface's area
  **does not exist as a magnitude**.
- ⚠️ **Per tooth, never in bulk.** A global reconstruction would join neighbouring teeth
  where their alpha complexes touch and out would come a bar of fused teeth.
- ⚠️ **The crown comes from the SCANNER**, by cropping its mesh with the `region_id` the
  scan carries per vertex. Where both modalities see the same surface, the one that
  measures it to tens of microns wins, not the one with 0.4 mm voxels — the same rule as
  the `viewer-export-agent`. Without `etiquetas_ios` the tooth comes out with the root
  alone and **that is declared**, rather than filling in with the CBCT's crown.
- ⚠️ **What it declares as `max_deviation_mm` is the RECONSTRUCTOR's error, not the
  format's.** The STL's `float32` gives ~1e-5 mm, which would look excellent and would say
  nothing about the half of the file that may be wrong. The same method is measured **in
  the crown band**, where the scanner is the reference, and extrapolated to the root
  **declaring it as a hypothesis**: nothing else measures roots.
- ⚠️ **This channel does NOT count towards `PipelineResult.reversible`**, unlike the
  viewer. The others re-materialise what came in and their deviation measures the round
  trip; this one writes geometry that never came in, so its number answers a different
  question. Putting them together turned the run red over 0.37 mm of a surface nobody had
  measured before, while the channels that do promise reversibility gave 0.000000 mm.
- ⚠️ **Bias is declared separately from p95 and with a sign**, and the sign does NOT come
  from the mesh's normals: a normal points wherever its faces' winding says, which is a
  file convention and not anatomy. It comes from the tooth's own radial direction.
- **The bodies are not welded.** Stitching crown and root would require deciding where one
  ends and the other begins, which is exactly what this agent cannot answer.
- **On synthetic data it returns `MISSING`, and that is correct:** `synthetic.write_case`
  generates a mesh and a field describing the same surface, i.e. a mouth with nothing under
  the gum.

**Change history**

| Date | Version | Change |
|---|---|---|
| 2026-08-24 | 0.3.0 | The arch is closed as a solid with a flat base perpendicular to the **occlusal** axis (with the superior axis it came out inverted). Enamel guardrail: a body with no enamel crown is not exported as a tooth. And it is declared when a reconstructed root is longer than its type admits. |
| 2026-08-24 | 0.2.0 | The arch now comes out WITHOUT roots — it is what gets printed and a printed tooth carries no provenance — and the reconstructed geometry moves to one STL per tooth with its crown measured from the scanner. |
| 2026-08-24 | 0.1.0 | Initial registration. STL of the composite by alpha complex per tooth, with the reconstructor's error measured in the crown band and the bias declared with a sign. |

---

### `viewer-export-agent` — The package `dental-3dgs-viewer` opens

| Field | Value |
|---|---|
| **Location** | `packages/export-agents/` (`visor.py` · `anatomia.py` · `base.py`) |
| **Version** | `0.3.0` |
| **Status** | `active` |
| **Pipeline phase** | 6 · Export (the contract → file boundary) |
| **Common contract** | `ExportOutput` + `BaseExportAgent` |
| **Orchestrator** | `IngestionPipeline.exportar(...)`; receives the gate's reasons and the scanner's labels |

**Role / Purpose**

> It emits what a splat viewer needs to show the case: **N PLY files in the INRIA
> profile**, one per switchable layer, and a **JSON sidecar** with the clinical layer —
> teeth, findings, non-regional measurements, gate reasons — the per-Gaussian `region_id`
> and the measured framing.

**Why not our own viewer.** One was written, in raw WebGL, and withdrawn: it drew round
`gl.POINTS` of uniform size, which is not splatting but a point cloud with good
presentation. `dental-3dgs-viewer` already rasterises real splats, has a layer panel,
dimensions over the model and per-case framing.

**Specific rules**

- ⚠️ **The viewer's PLY is NOT the twin, and that line cannot be erased.** For a
  rasteriser to open it you have to give it three things the CBCT did not measure:

  | INRIA property | where it comes from | is it data? |
  |---|---|---|
  | `opacity` | `alfa = 1 − exp(−g·sigma)`, then logit | **no**: a visualisation gain |
  | `f_dc_*` | false colour by FDI code | **no**: a CBCT does not measure colour |
  | `scale_*` | `log(sigma_mm)` | yes, the same measurement on another scale |

  The first two are **declared in the file's own header** and the sidecar repeats the
  transfer function. The reversible twin is still the `field-export-agent`'s PLY, with
  uncapped `density` and scales in millimetres; this one is for looking at.
- ⚠️ **Crowns come from the SCANNER, not from the CBCT.** The scanner already brings the
  teeth separated and with an FDI code, at tens of microns and with no holes; the CBCT's
  field covers 51% of each tooth's volume, and unevenly. Showing the CBCT composite as if
  it were "the tooth" presents as complete something that is not. What the CBCT does
  contribute and the scanner cannot is the **root**, which goes in its own layer and is
  declared partial.
- ⚠️ **The ORBIT axis is measured, not written by hand.** The viewer rotates around its
  `cameraUp`, and that was a world axis eyeballed in its `config.ts` — `[0, 0, 1]`. When
  the arch's occlusal axis does not coincide with it, dragging the mouse does not turn the
  arch around: **it tips it over**, and there is no way to reach a tooth's buccal face. The
  sidecar now carries an `encuadre` block with the occlusal axis derived from the FDI
  labels (see `export_agents/anatomia.py`), and the `fov_grados` the distance was computed
  with, because with a different field of view it does not frame. **It is measured over the
  scanner's vertices and only those**: in the composite, `fdi == 0` includes the gum *and*
  the CBCT's bone and skull, and letting that in could invert the sign of the occlusal
  axis. With no labels nothing is invented and the gate says so.
- **Splat size and opacity come from the measured 3DGS regime**, not from what looks good:
  σ inflated to ~2.5 times the spacing left after decimation — without that the splats do
  not touch and the surface comes out full of holes — and α above the viewer's discard
  threshold (`5/255 = 0.0196`), which was throwing away 46% of the root and background
  splats.
- **One colour per tooth, stable across cases.** The hue comes from the FDI code multiplied
  by the golden angle, so the same tooth comes out the same colour in any case and any
  visit, with no table to maintain.
- **The patient's name does not travel**, neither in the PLY nor in the sidecar: the gate
  reasons have their path stripped and the hash remains, which identifies the file without
  identifying anyone.
- **The package is NOT reversible and declares it.** It is decimated on purpose, and its
  opacity and colour are interpretation. The reversible files are the twin's STL and PLY,
  with their measured deviation.

**Change history**

| Date | Version | Change |
|---|---|---|
| 2026-08-21 | 0.2.0 | Initial registration of the channel: layers switchable by provenance and anatomy, INRIA-profile PLY derived from the field, and a sidecar with the clinical layer and the per-Gaussian `region_id`. |
| 2026-08-24 | 0.3.0 | Splat size and opacity tuned to the **measured 3DGS regime** — the previous α leaned on the viewer's discard threshold and threw away 46% of root and background — per-tooth colour stable across cases, and an `encuadre` block with the anatomical orbit axis. |

---

### `uos-export-agent` — The whole case as a UOS scene

| Field | Value |
|---|---|
| **Location** | `packages/uos/` (`agente.py` · `manifiesto.py` · `contenedor.py` · `validador.py` · `vistas.py` · `procedencia.py` · `volumen.py` · `escena.py` · `derivados.py` · `clinico.py`) |
| **Version** | `0.6.0` |
| **Status** | `active` |
| **Pipeline phase** | 6 · Export (the contract → file boundary) |
| **Common contract** | `ExportOutput` + `BaseExportAgent` |
| **Orchestrator** | `IngestionPipeline.exportar(...)`; with no pseudonym it declares `FAILED`, with no mesh `MISSING` |

**Role / Purpose**

> The other five channels materialise **one** thing — the mesh, the field, the composite,
> the viewer package, the render. This one packages the whole case with its relations
> declared: which asset comes from which visit, which frame each one lives in and with what
> transform they align. It is the difference between delivering files and delivering a
> scene.

It implements the **UOS-Core** and **UOS-Vol** levels of the *Unified Oral Scene* v0.2 draft
spec: an **uncompressed** ZIP whose first physical entry is `manifest.json`, and which
**references intact native formats** rather than transcoding them. No existing format does
that — DICOM does not model Gaussians and does not transmit well over the web, glTF models
neither volumes nor clinical metadata, OpenUSD is foreign to the clinical ecosystem — and
that is why the container is our own and the validator can run over a file somebody else
wrote.

**Specific rules**

- ⚠️ **It references, it does not transcode.** Files go in as they are and their `sha256`
  goes in the manifest: what comes out is byte-identical to what went in. The zero
  deviation it reports is **measured** — the container is re-read and each asset's hash
  recomputed — not asserted. On the real case: an 11,004,334-byte mesh inside and out.
- ⚠️ **With no pseudonym it declares `FAILED`, and does NOT fall back to
  `acquisition_id`.** That identifier comes from the case directory's name, which in a real
  system carries the patient's name or their record number. A default pseudonym that turns
  out to be the identifiable datum is worse than having none, because `phi_state` would say
  `pseudonymized` and be lying.
- ⚠️ **No vendor filename travels inside.** Real ones carry identifiers — this case's mesh
  was called `1574 UpperJawScan.stl` and `1574` is the case number. Inside, everything is
  named by its role: `scene/scan.stl`, `scene/appearance.ply`, `images/img_000.jpg`.
  Traceability comes from the `sha256`, which is stronger than a name and identifies
  nobody.
- **The canonical frame is the SCANNER, not the CBCT**, and that inverts what the pipeline
  does. Here we work centred on the CBCT because that is where the Gaussian field lives;
  UOS makes the scanner the hub because it is a dental case's reference geometry — microns
  against 0.3 mm voxels. The inversion happens at the boundary and **is declared** as
  `reg.ct_to_ios`, with its matrix, its method and its error, rather than rewriting
  geometry. It is consistent with fusion annotating rather than transforming.
- **The views' anatomical axes are MEASURED, not assumed.** It is the same rule that
  governs the CBCT's apico-coronal axis, which is read from `ImagePositionPatient`. A
  scanner mesh brings no header saying so, so every direction comes from the FDI labels:
  *occlusal* from the gum towards the crowns, *right* from the centroid of quadrants 2 and
  3 to that of quadrants 1 and 4, *anterior* from the molars to the incisors. **With no
  labels there are no views and it says so**: naming the cloud's principal axes produces
  plausible and sometimes inverted names, and a view called "right buccal" that shows the
  left is worse than not having it. It is the same reason the `render-export-agent` names
  its own by angle.
- **Only ANNOTATED teeth get their own view.** One per labelled tooth would be sixteen
  equivalent entries; what makes a deep link useful is that it points where somebody
  looked. Teeth the report cites and the scanner does not bring are grouped into **one**
  warning, not one per tooth: the gate already carries one that explains the whole thing,
  and repeating it buries the reasons that appear only once.
- **A `.uos` is logically append-only.** Modifying is not editing: it is writing a new
  version of the manifest that points at the previous one's `sha256`. The authority is
  `prev_manifest_sha256`, inside the manifest; `provenance/chain.json` materialises it so
  it can be walked without opening every version. The validator checks that the chain and
  the manifests **tell the same story** — touching up a manifest, tearing out a link or
  pasting in another case's chain all invalidate.
- **Structured data lives only in the manifest.** `ExportOutput` is `extra="forbid"` and is
  shared by six channels: widening it with `n_assets` or `conformidad` would give two
  places where the same truth can diverge and would force the other five to carry fields
  they do not use.
- ⚠️ **The container carries the TWIN, not just the inputs.** For a while it carried the
  scanner's mesh, the photos and the DICOM — i.e. what the clinic already had — and the
  fitted Gaussian field stayed in a PLY next to it. That was a misreading of §1.1: it says
  UOS does not re-encode **source data**, and it names DICOM; §3.1 draws the mesh as a
  "converted STL". Today the field, the composite and the appearance all travel.
- ⚠️ **Three Gaussian layers share a `kind`, so each carries a descriptor.** The field is
  density **measured** by the CBCT; the composite, a measurement from two modalities with
  its `origen` column; the appearance, a reconstruction against renders. Without the
  `.gs.json` a consumer sees three `mesh_gs_scene` and cannot tell which one it may measure
  with. The descriptor carries the contract's `esquema_campo`, which is what prevents the
  silent failure: the de facto 3DGS PLY uses `scale_0..2` with the same names and stores
  the **logarithm**; here they are linear millimetres, and a standard viewer would
  exponentiate our millimetres and render very good-looking garbage.
- **The scene is a binary glTF and the GS layers hang off it** (§5.1). The mesh's root node
  **is** the canonical frame and each Gaussian layer sits underneath with its `matrix`,
  which is the registration. ⚠️ glTF stores matrices **column-major** and the manifest
  row-major: confusing them does not blow up, it places the cloud rotated and mirrored. The
  payloads are pointed at with `extras.uos_gs_uri`, which is the declared fallback while
  `KHR_gaussian_splatting` is not ratified — §13 says it is withdrawn in v1.0.
- **The original STL travels as a `document`, not as a scene.** §5.1 says so. It stays
  declared for **provenance and layer**, not for resolution: a binary STL stores vertices as
  `float32` by definition, so converting back to `float32` for glTF loses *nothing* — the
  `float64` in `mesh-agent` is a widening of the data, not precision that was there. From an
  **OBJ**, which is decimal text with ~8 digits, the loss is real. The claim used to be that
  the conversion is lossy full stop, which was false for half the inputs (T-1).
- ⚠️ **Neither `extras.uos_fdi` nor `_REGION_ID` is emitted**, even though §5.1 allows the
  first "if the mesh comes segmented". Ours does not come segmented: we segment it with a
  model, and that is Layer 3. Baked into the scene, removing `derived/` would stop removing
  the inference and §5.5's hard rule breaks. The labels go in `derived/`, indexed by vertex
  — exact because the scene preserves the ingested mesh's ordering.

  This was violated between 0.4.0 and 0.5.0: the scene shipped split into one *primitive*
  per tooth, so that a third-party viewer could do §11.3's picking, and the manifest kept
  declaring it Layer 1. An external review of the specification flagged it as blocking
  (B-1) and it is reverted. **The price is real**: a foreign glTF viewer opens the
  container, draws the arch, and cannot select a tooth. Picking now requires `derived/`,
  which is the only way the three-plane separation is true rather than documented.
- ⚠️ **The DICOM series travels WHOLE and is verified slice by slice.** The volume asset is
  a directory (`volume/ct_001/`) and the manifest declares one `Parte` per file, with its
  name and its hash. A hash of the whole set is not enough: that one says "this series does
  not add up", and these say **which** of the 397 slices. It is checked both ways — a slice
  that is inside and undeclared is as serious as a declared one that is missing, because it
  means the series coming out is not the one that went in. The directory's `sha256` is
  defined as the hash over sorted `name\0hash`, and **it is defined in the contract and not
  in the writer**: if the validator computed it differently, a valid container would come
  out invalid and nobody would know which of the two was right.
- **Slice names DO travel, and that is correct** even though elsewhere in the container it
  would not be: the ordering of a DICOM series is clinical data, and renaming them to
  `IM0001.dcm…` would be rewriting the file we claim to deliver intact.
- **The volume's sidecar (§5.2) is read from the same bytes that travel**, not from a
  derived copy computed earlier somewhere else. It exists so that a web viewer does not
  need a full DICOM parser just to know what it is getting, and it declares the frame and
  **nothing else about alignment**: the transform to the canonical frame lives in
  `registrations` and is not duplicated. Orientation is read from
  `ImageOrientationPatient`; if it is absent, the identity is declared **as a warning**,
  because a volume whose orientation is assumed renders just as well and mirrored.
  `value_range` is left null when the DICOM does not declare it, rather than sweeping
  259 MB or putting in a CBCT's typical range — which a viewer would use for its window.
- ⚠️ **A series with identifiable data in its headers does NOT travel.** The DICOM goes
  intact — that is the point of the format — so its tags would go with it, and the manifest
  asserts `phi_state: pseudonymized`. A container that says it is pseudonymised and carries
  the patient's name inside is **worse** than one declaring `identified`: whoever receives
  it trusts the field and does not open 397 headers to check. `PatientName`,
  `PatientBirthDate`, the institution, the physicians and the accession number are all
  checked; the volume stays out, the case drops to UOS-Core and the gate says **which tag**,
  never its value. `PatientID` is deliberately off the list: an anonymised export fills it
  with an opaque identifier, and demanding that it be empty would reject perfectly
  anonymous series.
- ⚠️ **With no registration, the volume stays out and the rest of the case ships anyway.**
  Its frame would not connect to the canonical one, so a viewer would place it in the wrong
  spot with no way to detect it — worse than not carrying it. And killing the whole export
  over that would be disproportionate: the mesh, the photos and the views are fine. It is
  declared and the case drops to UOS-Core.
- **The volume sits behind a flag** (`--con-volumen`), because of weight: hundreds of
  megabytes that multiply the container tenfold. Carrying them is the exporter's decision.
- **The clinical layer is OUR extension, and is declared as such.** The draft does not
  define per-tooth clinical attributes: its §9 sends them to FHIR's `Observation`, i.e. to a
  server, and then a standalone `.uos` cannot answer "what does the report say about the
  24?". It goes in `clinical/observations.json` with the 32 observations, the measurements
  that do not fit on a single tooth and the gate's reasons. **Layer 1 is the file's
  default, not a claim about its content**: it is the transcription of a report a person
  signed, and putting it in `derived/` would make it detachable — but the `color` block is
  Layer 2, computed by the pipeline from the photographs, and says so in its own entry.
  Every value carries its own `regulatory`, `derivation` and `confidence`; nothing floats
  over the tooth any more. How each value was extracted is stated by its `derivation` —
  `deterministic`, `inferred`, or `null`, which means **not declared** and is not the same
  as deterministic.

  ⚠️ **`confidence` is not how sure the extractor was.** It is the weakest link in the
  chain that attached a value to *this* tooth, and the segmenter's confidence in the FDI
  code dominates it — which is why a `deterministic` value legitimately arrives at 0.745.
  The file says so in its own `regulatory` block, because every reader that does not know
  reads it as a contradiction. `color` carries none: that number is about the
  report-to-tooth chain and says nothing about a colour measurement.
- **The manifest declares its extensions** (`extensions`, `extensions_used`,
  `extensions_required`), and this **is not in v0.2**: it is our proposal, copied from glTF,
  which UOS leans on. Without it, an outside reader ignores what we added **without
  realising it is ignoring it**, and an open format becomes one only its issuer can read in
  full. ⚠️ **Nothing of ours goes in `required`**: everything we add adds information, and a
  conforming viewer must be able to open the case without understanding any of it. There is
  a test by that name so that if something ever lands there, it is deliberate and visible.
- **The `fhir_map` declares the resource TYPE, not a reference.** The spec's example writes
  `ImagingStudy/is-9911`, i.e. a resource that exists on a specific server; this case has
  not been through any PMS and there is no identifier to cite. Inventing one would make a
  connector try to resolve it. What is true today is asserted — `Media` for the photos,
  `ImagingStudy` for the volume, `DocumentReference` for the mesh and for the whole `.uos` —
  and `resource` stays empty.
- **Layer 3 lives in `derived/` and only there.** That is what makes it possible to detach
  the AI module by deleting a directory and its manifest entries, and to distribute the case
  in jurisdictions where it is not enabled. The validator checks it both ways.

**What is NOT there, said so nobody assumes otherwise**

- The **signals** — the `UOS-Sig` level. There is no T-Scan and nothing to record yet.
- The **rendering** of the Gaussians in the reference viewer: step 3 of §11.2. The scene
  already declares them; the rasteriser is missing.
- The **Ed25519 signatures**. It is not the code that is missing: it is deciding which key
  signs — the issuing clinic, the platform, or both — and where it lives. Signing with an
  invented key would give a `.uos` that *looks* signed, which is worse than one declaring it
  is not. The validator warns if it finds `provenance/signatures/` so as not to ignore them
  silently.

**Change history**

| Date | Version | Change |
|---|---|---|
| 2026-08-24 | 0.1.0 | Initial registration. UOS-Core level: manifest, ZIP/STORE container, validator with conformance levels, views with measured anatomical axes and a provenance chain between versions. Verified on the real clinical case: `VALID`, 10 assets, 19 views, byte-identical mesh. |
| 2026-09-03 | 0.12.0 | **T-1 to T-7.** Ten checks the text declared normative and the algorithm never ran: undeclared ZIP entries, `derived/` self-description, `derived_from` resolving, the `KHR_gaussian_splatting` invariants and complete SH degrees, the segmentation join by count and by a hash of the source `POSITION` accessor (T-4), view/volume coherence, and chain ordering. `value_range` is measured instead of left null "because sweeping the series is expensive" — the writer already reads every byte. And the reversibility argument is corrected: a binary STL is `float32` by definition, so the conversion loses nothing from an STL and does lose from an OBJ; the original stays declared for provenance and layer, not resolution. |
| 2026-09-03 | 0.11.0 | **D-4, D-5, D-6, D-9.** Findings travel coded — `{system, code, display}` with the issuer's closed vocabulary demoted to `display`, where it belongs — and the SNOMED `code` is deliberately `null`: assigning one is clinical terminology, and a guessed code is indistinguishable from a right one to the connector that resolves it. Occlusion gets a reserved id and must be declared even when the answer is `single_arch`; registrations declare what they were measured **fit for**, because an RMS average does not license guided surgery; sites that are not teeth get a vocabulary. Normative enum values that were Spanish (`"lineal"`, `"sigma_normalizada"`) are now English, since a reader branches on them. The specification names where UOS maps onto DICOM's Spatial Registration Object, Encapsulated STL, FHIR `Provenance` and R5's `ImagingSelection` rather than reinventing them. |
| 2026-09-03 | 0.10.0 | **D-1, D-2, D-3, D-7, D-8.** Frames anchor to the DICOM Frame of Reference UID and declare their anatomical convention — "right-handed" fixes chirality, not which direction is the patient's anterior. A slice's identity stops being its file's hash, which any de-identification changes, and becomes the SOP Instance UID plus a hash of PixelData; verification splits into identity and exact bytes, reported separately. A `measured` layer must declare the occupancy threshold that decides which tissue appears. And the container stops calling CBCT grey values Hounsfield units: it published `±N HU` for a scanner that is not calibrated in them, and the volume sidecar now declares `calibrated_hu`. |
| 2026-09-02 | 0.9.0 | **B-6 and part of G-1.** `UOS-Distributable`, a profile orthogonal to the conformance levels and derived the same way: the levels say whether a reader can *open* a container, and nothing said whether it is in a condition to *leave* the organisation that issued it — the question asked immediately before attaching a case to an email. It is B-1, B-3 and B-4's conditions at once, because separately they decide nothing. Not being distributable is not an error; it is the normal state of a case inside the clinic, and it now reaches the human gate with what is missing. Separately: the published JSON Schema is now entirely in English. Field names were already the wire format, but `title` and `description` were generated by pydantic from Spanish class names and docstrings, so the one artifact that exists for an outside implementer had an unreadable half. |
| 2026-09-02 | 0.8.0 | **B-3 and B-4.** `phi_state` alone could not sustain what it claims: it is a statement about DICOM tags, and the container identifies a person without any — `scene/field.ply` is CBCT density including soft tissue, and a facial surface reconstructs from it. `pseudonymized` and `anonymized` now require a `deidentification` block in the vocabulary of DICOM PS3.15 Annex E, and the agent **computes** `phi_state` from what it actually ships: carrying a `measured` volume-derived layer without cleaning recognisable features, it declares `identified` and sends the reason to the human gate rather than overclaiming. `acquisition.device` loses its free-form map (and with it any chance of shipping a serial number); `date_shift_days` may only travel in an already-identified container. B-4 adds `subject.consent` and `purpose_of_use` from one closed vocabulary, the second required to be contained in the first — a container cannot be issued for a use the patient did not consent to — and neither is ever defaulted. |
| 2026-09-02 | 0.7.0 | **B-2 and B-5.** Regulatory layer **per value** in `clinical/observations.json` (`ash-clinical/2.0`): the file declared Layer 1 for everything, and the `color` block — CIELAB per crown third, flash falloff regressed out — is computed by the pipeline, not transcribed from a signed report. It is now Layer 2 with its `derived_from`. **Layer 2 is defined for the first time**: computed by a deterministic, reproducible procedure from Layer 1, with no trained model. It existed in fact — the ICP transform, the STL converted to glTF, the trained appearance — and all of it shipped as Layer 1 because the document defined only 1 and 3. `status` + `jurisdictions` are replaced by `clearances[]` with a closed status vocabulary; an empty array means *not declared*, by written definition, and the validator warns. `Registro.regulatory` loses its default so that an automatic registration has to declare its layer instead of inheriting one. |
| 2026-09-02 | 0.6.0 | **B-1, second pass.** The first pass removed the FDI code from `scene/scene.glb`, which is what the external review named — it reviewed the specification, not one of our containers. Opening a real one showed the same violation in two more files: `scene/field.ply` and `scene/composite.ply` shipped a `region_id` column, the same code from the same segmenter, in two more assets declared Layer 1. The column is now extracted on the way into the container and rewritten as `derived/seg_gaussians.<layer>.bin`, the per-Gaussian sibling of `derived/seg_teeth.bin`. The working files on disk are untouched: what is regulated is the container. Check 17b extended to PLY headers, and an end-to-end test now walks **every** layer of `scene/` rather than the one that was fixed. |
| 2026-09-02 | 0.5.0 | **B-1 of the external review**: `scene/scene.glb` stops being split by tooth and stops carrying `extras.uos_fdi`, and the `KHR_gaussian_splatting` primitive stops carrying `_REGION_ID`. Both are FDI codes, both come from a segmenter, and both were baked into an asset the manifest declared Layer 1 — so deleting `derived/` no longer removed the inference that §3.1 promises can be removed. The labels keep travelling whole in `derived/seg_teeth`; picking is rebuilt in the reader by vertex index. A new validator check parses every non-Layer-3 GLB and rejects both attributes, because the previous check verified where Layer 3 is **declared**, not where its content **is**. |
| 2026-08-24 | 0.4.0 | Conformance with the draft: the registration stops writing `rms_error_mm: null` while holding the measured residual — a naming bug `getattr` was covering — and stops crediting the ICP to the wrong agent; the scene is split into one *primitive* per tooth with `extras.uos_fdi` (§5.1), which is what enables §11.3's picking in a third-party viewer; the photos declare `projection` (§5.3); the views carry `mpr` and `clip_planes` when the volume travels (§7); and the **per-version JSON Schema** §12 requires is published, checked by the validator itself. |
| 2026-08-24 | 0.3.0 | The container carries the **twin** and not only the inputs: a glTF scene with the GS nodes hanging off it and their registration as a `matrix`, field and composite with a descriptor declaring whether they are **measured**, segmentation in `derived/`, the clinical layer, and an **extension mechanism** proposed to the draft. |
| 2026-08-24 | 0.2.0 | **UOS-Vol** level: the whole DICOM series as a directory asset, verified slice by slice, with its §5.2 sidecar read from the headers that travel, and rejected if those headers contradict the `phi_state`. And the `fhir_map` populated by resource type. |


---

### Phase 6 in the orchestrator

`IngestionPipeline.exportar(result, destino)` fires the **six** channels over a
`PipelineResult` and returns a **new** one, with `exports` and the accumulated review
reasons. Five materialise a piece of the twin — STL, field, composite, viewer package,
render — and the sixth, `uos-export-agent`, packages the whole case as a scene.

It sits in a separate method — not inside `run` — by contract: exporting **writes files**,
and `run`/`fuse` are pure with respect to disk except for the artifact store.

Three rules worth keeping to hand:

- **The channels are independent.** No mesh does not prevent exporting the field, and a
  failing render does not delete the STL already written.
- **`PipelineResult.reversible`** requires every channel that ran to be within *its* budget
  — millimetres for geometry, PSNR/SSIM for images — that none is `FAILED`, and that
  something was exported. With no run there is no reversibility to assert.
- **`MISSING` is not a human review reason.** An acquisition being CBCT-only is normal and
  says nothing about the file that was written; a `FAILED` is declared.

Tested end to end in
[`apps/agent-orchestrator/tests/test_e2e.py`](apps/agent-orchestrator/tests/test_e2e.py):
**ingestion → fusion → segmentation → export**, with fusion registering the twin's real
clouds — the `mesh-agent`'s mesh against the `cbct-agent`'s field — and not a synthetic blob
against itself. Three checks carry the weight:

- **The geometry that comes out is the one that went in**, comparing the original OBJ's
  bounds with the regenerated STL's. It exists because each agent's internal metrics measure
  against what *that agent* wrote: a lost axis or a mirror would pass all of them and not
  this one.
- **Fusion unblocks `frame="twin"`**, which needs the `RigidTransform` only fusion writes.
  It ties the phases together through a real dependency and not through call order: if
  someone disconnected fusion, it would go red even though every stage still passed its own.
- **Tooth labels reach the file**: the exported PLY's `region_id` has to be the one
  segmentation inferred. This is the one that found something — see below.

The run's segmentation model is a **test double**, because the real one needs a GPU and an
unversioned checkpoint. It does not weaken what is being tested: an integration test has to
prove the stage is *wired in*, and for that the labels' origin is irrelevant. The model's
quality is measured elsewhere and is not a test.

⚠️ **Tying segmentation into the run exposed a silent loss.** Both stages passed their tests
separately — the `segmentation-agent` stored `region_id` in the artifact, the
`field-export-agent` wrote a correct PLY — and the label was lost exactly between them: the
file came out with correct geometry and nothing saying which Gaussian is which tooth. It is
the same failure mode as the `cbct-agent`'s discarded centroid, and invisible for the same
reason: **no stage test looks at what the next stage needs**.

⚠️ **Confidence and reversibility are not the same thing**, and the run shows it:
registering the scanner's mesh against the CBCT's hard-tissue voxels gives 0.605 mm over a
52.7% overlap, so the human gate **fires** — correctly, they are not the same surface. And
the export still comes out with its metrics green, because reconstruction error measures
whether the file reproduces the twin, not whether the twin is well registered.

---

### Development agents (dev-time)

External AI tools the team uses to assist development. **They are not part of the production system** and have no autonomous access to the runtime: all of their output enters the repository as proposed code and goes through a Pull Request + human review (and through the `ai-code-reviewer` guardian) before being merged.

| Tool | Role in the project | Model | Governance notes |
|---|---|---|---|
| OpenCode / Claude Code | Interactive coding assistants: generation, refactoring, tests and documentation under the direction of a team member | Claude (Opus/Sonnet depending on the session) | Human-driven (not autonomous); no access to clinical data; all output via PR + human review. No clinical or architectural decisions are delegated to them. |

> They are documented at row level (not with an agent card) because they are **interactive** assistants, not system agents: they have no data contract, no pipeline phase and no delegation rules of their own. Register here any other dev-time AI tool that gets adopted.
