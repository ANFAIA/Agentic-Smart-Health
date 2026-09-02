# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Guidelines

- **Added** — new features or capabilities introduced in this release.
- **Changed** — changes to existing functionality (non-breaking where possible).
- **Deprecated** — features that will be removed in a future release.
- **Removed** — features removed in this release.
- **Fixed** — bug fixes.
- **Security** — fixes for vulnerabilities. See also [SECURITY.md](SECURITY.md).

Keep entries in reverse chronological order (newest first). Each entry should be
human-readable and linked to the relevant pull request or issue where applicable.

---

## [Unreleased]

### Added

- **UOS format specification** (`docs/spec/uos-format-spec-v0.2.tex`): a self-contained
  LaTeX source covering the container, data model, manifest, scenes, volume, clinical
  layer, `derived/`, views, provenance, extensions, the 22-check validation algorithm and
  a cookbook of seven recipes. Written to be sufficient on its own: someone who has never
  seen this repository should be able to open, render, verify, extend and re-emit a `.uos`
  from it.
- `Informe.externos` in the UOS validator: how many acquired originals a container
  references without holding.
- `uos.version`: version branching for readers. A container declaring the reader's own
  version or older is parsed strictly; a higher **minor** is parsed leniently, ignoring and
  naming the fields it does not know; a higher **major** is refused. The danger in the last
  case is not the unknown fields — those could be ignored too — but the known ones, which a
  major is free to redefine.
- Versioning policy and a recipe for regenerating the improved mesh, both in the format
  specification.

<!-- List new features, agents, schemas, or capabilities added since the last release. -->

### Changed

- **BREAKING — `scene/scene.glb` no longer carries the FDI code, in either form.** The mesh
  stops being split into one *primitive* per tooth with `extras.uos_fdi`, and the
  `KHR_gaussian_splatting` primitive stops carrying `_REGION_ID`. Both are FDI codes, both
  come from a segmentation model — Layer 3 — and both were baked into an asset the manifest
  declared Layer 1, so deleting `derived/` stopped short of removing the inference the
  three-plane separation promises it removes. Raised as **B-1** in an external review of the
  format specification, and blocking: the claim that Layer 1 is not regulated medical
  software is what the separation buys, and a documented compromise does not sustain it in
  front of an auditor.

  Nothing is lost. `derived/seg_teeth.bin` already shipped with every partitioned container,
  and per-tooth selection is rebuilt in the reader by vertex index — for Gaussians, by the
  same nearest-crown-vertex join that produced them. **What is lost is interoperability, and
  it is deliberate**: a foreign glTF viewer opens the container, draws the arch, and cannot
  select a tooth. Picking now requires `derived/`, which is what makes the separation true
  rather than merely documented.
- **BREAKING — regulatory layer is declared per value, not per file** (`ash-clinical/2.0`).
  `clinical/observations.json` declared `layer: 1` once, for all of it, on the grounds that
  it is "the transcription of a report a person signed". Most of it is. The `color` block —
  CIELAB per crown third with the flash falloff regressed out — is not: the pipeline
  computes it from the photographs, and nobody signed it. The format had extraction
  provenance per value (`derivation`, answering *how*) and no regulatory layer per value,
  so nothing answered *who answers for it*: a reader who deleted `derived/` kept computed
  measurements believing they kept a signed report. Raised as **B-2**.

  `confidence`, `agent` and `derivation` move down with it. They used to be written once
  per tooth over a `setdefault`, so with two observations on one tooth `findings`
  accumulated while those three were overwritten by the last — and, worse, the same
  `confidence` floated over `color`, which comes from an entirely different chain. The
  file-level `regulatory` stays as the declared **default**.
- **The specification defines layer 2 for the first time**, and `regulatory` changes shape.
  `layer` accepted `1..3` while the document defined only 1 and 3, so every deterministic
  computation — the ICP transform, the STL converted to glTF, the subsampled density field,
  the trained appearance — shipped as Layer 1 without anybody saying so. Layer 2 is
  *computed by a deterministic, reproducible procedure from layer-1 assets, with no trained
  model*; it may live outside `derived/` but **must** declare `derived_from`, because a
  reproducibility claim the container cannot substantiate is not a claim. Raised as **B-5**.
- **BREAKING — `regulatory.status` and `regulatory.jurisdictions` are replaced by
  `clearances[]`.** `status` was free text and `jurisdictions: []` was ambiguous — none, or
  not declared? — which is exactly the ambiguity the format forbids in `fdi_targets` and
  `derivation`. A clearance is `{jurisdiction, regime, status, reference}` with `status`
  closed to `not_a_device | investigational | submitted | cleared | withdrawn`. An empty
  array now means *not declared* by written definition, and the validator warns for every
  Layer 3 asset that leaves it empty.
- **BREAKING — `Registro.regulatory` loses its default.** With `default_factory` every
  registration arrived carrying `layer: 1`, so a transform computed by an ICP was, on
  paper, as acquired as the CBCT it came from, and "declared layer 1" was indistinguishable
  from "nobody declared it". It is now optional and the validator requires it whenever
  `operator` begins with `auto:`.
- New validator checks 17c and 17d: layer 2 declares `derived_from`, layer-3 assets without
  clearances warn, automatic registrations declare their layer; and inside `clinical/`, no
  value is layer 3, no `inferred` value is layer 1, and every layer-2 value names its
  sources.
- **BREAKING — the Gaussian layers stop carrying `region_id` too.** `scene/field.ply` and
  `scene/composite.ply` shipped a `region_id` column: the same FDI code, from the same
  segmenter, in two more assets the manifest declares Layer 1. The external review named
  only the two forms inside the glTF because it reviewed the specification, not one of our
  containers. The column is now extracted as the file enters the container and rewritten as
  `derived/seg_gaussians.<layer>.bin` — the per-Gaussian sibling of `derived/seg_teeth.bin`,
  one file per layer because each has its own count and order, each with its `.meta.json`.
  The working files the export agents leave on disk are untouched: what is regulated is the
  container, not the working directory.

  This also closes the gap that made the column necessary. A Gaussians-only container
  carries no mesh, so `seg_teeth.bin` indexes nothing and per-tooth selection had no
  mechanism other than the column — the coupling `uos.agente` documented as "the two things
  go together". With a per-Gaussian file, both profiles rebuild selection from `derived/`
  and neither needs Layer 3 inside a Layer 1 payload.
- The stripped PLY also loses the header `comment` lines that describe the column, and
  gains one naming where the codes went. `composite-export-agent` writes "de las del CBCT,
  79118 llevan codigo FDI en region_id" — true of its working file and false of the
  container. A header describing an absent value is worse than describing nothing: a reader
  goes looking for bytes that are not there.
- §16 of the specification declares that no partitioned scene is emitted under `derived/`.
  B-1 point 5 offers that second GLB as the conforming way to keep per-tooth selection for a
  generic glTF reader, and the reference writer does not produce it: the reader it would
  serve barely exists — reaching `derived/scene_partitioned.glb` means already knowing
  enough about the format to join `derived/seg_teeth.bin` instead, at 0.22 MB against 17.8 —
  and a duplicate of the same geometry with nothing declaring which copy governs is worse
  than the loss. Declared rather than left silent, so a second implementer reads it as a
  choice and not an oversight.
- New validator check (17b): every GLB in an asset that is not Layer 3 is parsed, and both
  `extras.uos_fdi` and `_REGION_ID` are rejected; every PLY is checked for a `region_id`
  property. Check 17 verifies where Layer 3 is *declared*; this verifies where its content
  *is*. Without it, check 17 passes on a
  container whose Layer 1 scene is full of model output — which is exactly what happened
  between `uos-export-agent` 0.4.0 and 0.5.0.
- `scene/appearance.gs.json` stops declaring a `region_id` column, which the shipped
  primitive no longer has. A descriptor that names a column the container does not carry is
  the same failure the descriptor's own derivation was written to prevent.
- **The public documentation surface is now entirely in English.** The root `README.md`,
  `AGENTS.md`, `CONTRIBUTING.md`, `SECURITY.md`, `docs/cierre-mvp.md` and the three
  package READMEs (`ingestion-agents`, `fusion-agents`, `tooth-aggregation`) are
  translated, including the headers of the four blocks generated by `docs_sync.py`.
  Internal documentation (`docs/architecture/`, `docs/research/`, `notebooks/`) and every
  code identifier stay in Spanish: a package README is a cover page, a module name is an
  API.
- Each script under `scripts/` declares its own one-line English summary for the README
  table (`RESUMEN_EN`, or `# resumen-en:` in shell). `docs_sync.py` reads it with `ast`
  and **fails in red** if a script does not declare one, so a newly added script cannot
  slip a Spanish row into the English cover page. The docstrings themselves — essays of
  dozens of lines that argue what each script does and does not measure — stay in
  Spanish and untranslated, and `--help` with them.
- `CONTRIBUTING.md` describes the `develop`/`main` workflow: branches and pull requests
  target `develop`, `develop` into `main` only for a release, tags and releases from
  `main`.
- `SECURITY.md` lists the components that actually exist. The withdrawn
  `slicer-mcp-server` is gone from the scope, and `apps/research-agent` — the only
  component holding an API key and reaching the network — and
  `packages/tooth-aggregation` are now in it.
- **BREAKING — `UOSExportAgent.export()` no longer accepts `sin_originales`.** Acquired
  originals — the DICOM series, the scanner STL, the photographs, the reports — are always
  referenced by content address and never carried. Callers passing the argument get a
  `TypeError`; there is no replacement, because there is no longer a mode in which they
  travel. Carrying them was never a profile: a duplicated `= False` in the orchestrator's
  signature once beat the agent's `= True` and shipped 216 MB of source data inside a
  container that declared it carried none.
- The validator reports referenced originals **once per container** instead of once per
  asset. When every original is external, a per-asset warning fires on every container and
  distinguishes nothing — on the example case it was 13 identical lines burying the one
  warning that mattered.
- Tests that need a carried DICOM series now build that container themselves with
  `escribe_uos` instead of asking the exporter for one. They exercise `_valida_serie`, and
  a validator has to work on files another emitter wrote.
- The published JSON Schema is regenerated: the `Extension` docstring it embeds now names
  the glTF version that introduced `extensionsUsed`/`extensionsRequired` instead of dating
  it as "years ago".

<!-- List modifications to existing behaviour, APIs, or agent configurations. -->

### Fixed

- **`uos_version` was written and never read.** Neither the reader nor the validator
  compared it with their own, so the compatibility a version number exists to provide was
  unenforceable: every model sets `extra="forbid"`, and an optional field added by a later
  minor did not warn, it broke parsing. A reader that ignores fields it does not understand
  is additionally barred from re-emitting that case — it would drop them, and the provenance
  chain would still verify a successor with content silently missing.
- **A registration is provisional by who computed it, not by which algorithm.** The rule
  required `method == "auto_dl"`, and the only registration the pipeline emits declares
  `icp_surface`: automatic, unverified, 0.666 mm of residual — and it did not trip the
  warning that exists for it. It now keys on an `operator` beginning `auto:` with no
  `verified_by`.
- The manifest schema's `$id` resolves. It pointed at `histora.dev`, a domain nobody
  registered; it now points at the schema in this repository, pinned to the
  `uos-spec-v0.2` tag so the identifier always names the same document.
- The scene's `uos_note` no longer describes the scanner STL as carried inside the
  container. It is referenced by content address, and the mesh is regenerated from the
  scene rather than returned.

<!-- List bugs or incorrect behaviours that have been corrected. -->

---

<!-- Releases will be added above this line as version tags are cut. Example:

## [0.2.0] - YYYY-MM-DD

### Added
- Agent-orchestrator: fusion agent for multimodal temporal integration.

### Fixed
- core-schemas: corrected STL export schema validation edge case.

[0.2.0]: https://github.com/anfaia/agentic-smart-health/compare/v0.1.0...v0.2.0

-->

[Unreleased]: https://github.com/anfaia/agentic-smart-health/commits/main
