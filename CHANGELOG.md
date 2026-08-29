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
  layer, `derived/`, views, provenance, extensions, the 21-check validation algorithm and
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
