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
- **Ten validator checks the specification declared normative and the algorithm never ran**
  (**T-3**). The two that matter most are structural: every ZIP entry must be declared by
  the manifest — the opposite direction was checked and this one was not, and an undeclared
  file travels with no hash accrediting it and no regulatory layer, which is the shape a
  leak would take — and every asset under `derived/` must carry a sidecar naming its model,
  its sources and its encoding, where before only *declared* sidecars were checked. The
  rest: `derived_from` resolves to declared ids, `KHR_gaussian_splatting` invariants
  (`SCALE ≥ 0`, `OPACITY` in [0,1], unit `ROTATION`) and complete spherical-harmonic
  degrees, the segmentation join by byte count, view/volume coherence, camera sanity, and
  chain ordering (`version` consecutive, `created` not going backwards).

  The check the review asked for that is **not** in the algorithm is the external asset's
  content address. It was already enforced one layer earlier, by a model validator, so a
  manifest with that inconsistency never parses. Adding it again would be code that cannot
  run, which suggests a coverage that comes from somewhere else; the specification now says
  where the rule lives instead.
- **`derived/seg_gaussians.*` are declared as assets.** They shipped inside the ZIP and
  outside the manifest — no hash, no regulatory layer, nothing linking them to the layer
  they index. Found by the undeclared-entries check above, not by rereading: an omission
  from B-1's second pass.
- **`value_range` is measured during the `parts[]` hash, not left null** (**T-2**). The
  justification for leaving it null was that sweeping every pixel on every export is too
  expensive, and it does not hold: the writer already reads every byte of every slice for
  the file hash and, since D-3, for the PixelData hash, so the minimum and maximum come out
  of a read already paid for. A cost that did not exist was being paid with a defect that
  did — a viewer with no display window. The specification now states where it is computed,
  and the export walks the series **once** for the parts, their DICOM identity and the
  range together: three reads per slice down to two. `null` is reserved for the one honest
  case, an issuer that never had the pixels.
- **The reversibility argument is corrected** (**T-1**). The specification said converting
  to glTF is lossy because "the ingested mesh is `float64`". A binary STL stores vertices as
  `float32` *by definition of the format*, so the `float64` is the loader's widening and the
  conversion loses **nothing** — the measured residual is the STL's own `float32`. From an
  OBJ, which is decimal text, the loss is real. The claim was true for half the inputs and
  stated for all of them. `scene.glb` still must declare `derived_from`, for provenance and
  regulatory layer rather than resolution.
- **`source_positions_sha256` in the segmentation sidecar** (**T-4**). The join between
  `derived/seg_teeth.bin` and the scene is positional, and the specification admitted that
  breaking it "breaks that join silently". A format should not have silent joins: the
  sidecar now accredits the source `POSITION` accessor and the validator recomputes it.
- The specification names **JWS or COSE Sign1** as the destination for signatures rather
  than leaving a private scheme to be invented (**T-5**), records that the Gaussian layers
  are **monolithic** so addressable is not confused with streamable (**T-7**), states that
  check 4 must run *after* the version branch, and updates the reference registration to
  carry its `regulatory` and `fit_for` (**T-6**).
- **BREAKING — clinical findings travel coded.** `findings: ["aparato_ortodoncico"]`
  interoperates with nothing: not with the FHIR `Observation` it maps to, which expects a
  coded `code`; not with a foreign practice-management system; not with a second
  implementer. A finding is now `{system, code, display}`, with this issuer's closed
  vocabulary demoted to `display` where it belongs. Raised as **D-5**.

  **The SNOMED `code` is `null`, deliberately.** Assigning one is an act of clinical
  terminology, not of programming: it needs a licensed browser and somebody who answers for
  the equivalence. A guessed code would be indistinguishable — to the connector that
  resolves it against a terminology server — from a correct one, which is the plausible,
  silent, already-inside-the-clinical-record failure ADR 003 calls the worst kind. `null`
  says *not mapped*; a number put there by hand says *mapped*, and lies. The file declares
  why, so nobody reads it as a gap.
- **BREAKING — normative enum values that were Spanish are now English**: `scale: "lineal"`
  → `"linear"`, `unit: "sigma_normalizada"` → `"normalised_sigma"`. A reader branches on
  these strings, so they are wire format rather than labels for us. `role`, `note` and
  `meaning` stay in Spanish, being human labels.
- **Occlusion, sites and registration fitness** (**D-9**). `reg.mandible_to_maxilla` is a
  reserved id: mandible-to-maxilla is the clinically most important registration of a dental
  case and the format did not name it, which invites every writer to call it something
  different. `occlusion` must be declared even when the answer is `single_arch` — silence is
  never "there is none". `Registration` gains `max_error_mm`,
  `target_registration_error_mm` with its region, and `fit_for`: an RMS average does not
  license guided implant surgery, where the maximum local error is what decides, and an
  empty `fit_for` means *not declared* rather than *fit for nothing*. `uos_site` reserves a
  vocabulary for implant sites, edentulous spans, pontics and abutments, which `uos_fdi`
  could not name.
- The specification names where UOS maps onto what already exists rather than reinventing
  it (**D-4**, **D-6**): DICOM's Spatial Registration Object and Encapsulated STL Storage,
  FHIR `Provenance` for the manifest chain, and FHIR R5's `ImagingSelection` for
  `projection.fdi_targets` — the last so nobody invents a private extension for something
  R5 already has. It also documents that a US reader sees `"27"` as a different tooth under
  the Universal Numbering System.
- **BREAKING — a DICOM slice's identity is its SOP Instance UID and its pixels, not its
  file's hash.** `Part.sha256` covers the whole file, so any de-identification changes it —
  and de-identification is the step every clinical flow takes, and the one B-3 now requires.
  The traceability the format promises, "whoever holds the series can prove it is this
  case's, slice by slice", broke at exactly the most common step. `Part` gains
  `sop_instance_uid` and `pixel_data_sha256`; the volume asset gains `series_instance_uid`
  and `study_instance_uid`; and the `parts[]` digest is computed over identity when every
  part declares it, falling back to name and file hash so older containers still verify.
  Verification is now two levels, reported separately — a de-identified file that preserves
  identity passes the first and fails the second, and that is information, not an error.
  Raised as **D-3**.
- **Frames anchor to DICOM and declare their anatomical convention.**
  `dicom_frame_of_reference_uid` `(0020,0052)` is required on every frame a `volume`
  declares: `frame.ct_001` is a string the writer invented, and a reader receiving the
  series through another channel could only trust it. `anatomical` is `LPS`, `RAS` or
  `device` — "right-handed" fixes chirality, not orientation, and DICOM, an intraoral
  scanner and glTF can all be right-handed while agreeing on nothing useful. Volume frames
  must be LPS. Raised as **D-1** and **D-2**.
- **A `measured` Gaussian layer declares its occupancy threshold** and its subsampling
  method. "One primitive per *occupied* voxel" hides a decision that changes which tissue
  appears — raising the threshold erases dentine before enamel — and a descriptor that
  says `measured: true` while staying silent about it applies the silence rule backwards.
  Raised as **D-8**.
- **BREAKING — the container stops calling CBCT grey values Hounsfield units.** It
  published the fit error as `±N HU` for a scanner that is not calibrated in them: CBCT
  greys depend on the device, the field of view and the position inside the volume. The
  error is now expressed in the field's own unit with the caveat attached, and the volume
  sidecar declares `calibrated_hu`, true only for a conventional CT. Raised as **D-7**.
- The synthetic DICOM writer now emits `FrameOfReferenceUID`. It never had, while the
  fixture's docstring claimed "complete headers" — found by D-1's new rule, not by reading.
- **`UOS-Distributable`**, a profile orthogonal to the conformance levels. Core/Vol/Sig/Full
  describe which asset kinds a container holds — whether a reader can *open* it. Nothing
  described whether it is in a condition to *leave* the organisation that issued it, which
  is the question asked immediately before attaching a case to an email. A container earns
  the profile when `phi_state` is not `identified`, `deidentification` is present and
  consistent with the payload, `purpose_of_use` is declared and within `consent.scope`, no
  asset declares layer 3 outside `derived/`, and it validates against its own manifest. One
  profile and not five flags, because separately they decide nothing: a container with its
  purpose declared and a reconstructible face inside is no more sendable than one with
  neither. Raised as **B-6**.

  Failing it is **not an error** — it is the normal state of a case living inside the
  clinic. It is reported, with what is missing, and the export sends that to the human gate.
  Our own containers do not currently earn it, for the reason B-3 gives.
- **The published JSON Schema is entirely in English.** Field names and enum values already
  were, being the wire format; `title` and `description` were generated by pydantic from
  Spanish class names and docstrings, so `schemas/uos-manifest-0.2.schema.json` shipped
  `"title": "Desidentificacion"` over twelve Spanish lines about Safe Harbor. This is the
  one artifact that exists so somebody outside can check their reader against something
  other than their own output, and half of it was unreadable to its audience. The `$defs`
  keys are the English titles and every `$ref` is rewritten to match. Generating the schema
  now raises, naming the models, as soon as one appears without a translation, and a test
  asserts the published file carries no Spanish. Part of **G-1**; what remains of it is the
  breaking rename of module names, public API and normative enum values.
- **BREAKING — `pseudonymized` and `anonymized` now require a `deidentification` block.**
  `phi_state` treated de-identification as a property of DICOM tags, and the container
  identifies a person with no tags at all: `scene/field.ply` is the CBCT density including
  soft tissue, and a facial surface reconstructs from it — an "image comparable" to a
  full-face photograph under HIPAA Safe Harbor, a biometric datum under the GDPR. The
  dentition identifies on its own. The block names what was actually done, in the
  vocabulary DICOM PS3.15 Annex E already defines, because *what was done* is the only one
  of the two statements anybody can check. Raised as **B-3**.

  **The exporter now computes `phi_state` from what it ships.** Our pipeline pseudonymises
  the patient identifier with an HMAC — correct, and enough for the tags — and does not
  implement defacing. So a container carrying a `measured` Gaussian layer derived from the
  CBCT declares `identified`, with the reason sent to the human gate. It is the honest
  answer available today; the day defacing exists, the same container declares
  `pseudonymized` truthfully. The `options` list declares only what actually runs: an
  option named but not executed is the worst kind of lie, the kind an auditor reads as a
  guarantee.
- **BREAKING — `acquisition.device` is an object with fixed keys** (`manufacturer`,
  `model`, `software_version`) and no serial number. A free `str→str` map in a clinical
  container ends up holding the serial, which Safe Harbor lists as a direct identifier
  alongside the patient's name.
- `deidentification.date_shift_days` may only travel in a container already declared
  `identified`. Shifting every date by the same amount preserves longitudinality and is the
  recommended PS3.15 option; publishing how far undoes exactly the measure it claims.
- **`subject.consent` and `purpose_of_use`**, from one closed vocabulary (`treatment`,
  `lab_manufacturing`, `second_opinion`, `research`, `model_training`). A container leaving
  for a dental laboratory, for a second opinion, and for a training pipeline are three
  different legal acts and the format distinguished none of them, so every recipient had to
  assume one. `purpose_of_use` must be contained in `consent.scope`, and neither is ever
  defaulted — assuming a legal act is the failure the field exists to prevent. Raised as
  **B-4**.
- Recipe §14.9 becomes *prepare a container for distribution*: removing `derived/` is one
  of three steps, and doing only it produces a container that is regulatorily clean and
  still legally undeclared. The other two are declaring the purpose and checking the PHI
  claim against the payload.
- The reference implementation already satisfied B-3's pseudonym rule: `cbct_agent.pseudonymize`
  is an HMAC-SHA256 under `ASH_PSEUDONYM_SALT`, not the plain truncated hash the
  specification described. The specification's wording was what was wrong, and it now
  states the requirement normatively.
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
