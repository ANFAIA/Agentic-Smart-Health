# MVP close-out — what is measured, what is unresolved and what is left for later

**Date: 2026-08-25.** This corresponds to the week 8 milestone of the brief: "MVP tested,
preliminary validation with the partner organisation, final technical documentation".

This document is the honest inventory. What works comes with its number; what does not
work comes with its number too, and with the measured cause where there is one. An MVP
closed by reporting only what went well is useless for deciding what to do next.

---

## 1 · The four figures from the brief

Measured with [`scripts/metricas.py`](../scripts/metricas.py), not promised. Three are met;
the fourth is not, and it is declared:

| Commitment | Target | Measured | |
|---|---|---|---|
| Ingestion latency for a complete set (STL + CBCT + report) | < 60 s <!--const:LATENCY_BUDGET_S--> | **12.7 s** | met |
| Fidelity of the mesh regenerated from the Digital Twin | < 0.1 mm <!--const:REVERSIBILITY_BUDGET_MM--> | **4.59 × 10⁻⁶ mm** | met |
| Automated test coverage | > 80% | **95.1%** | met |
| Reliability of the ingestion agents | > 95% | **93.8%** (N = 16) | **not met** |

**On reversibility.** The error that remains is the `float32` of the STL itself: the
`mesh-agent` stores positions in `float64` along with the complete face topology, so
regeneration does not reconstruct, it **returns**. There is no marching cubes anywhere
along the path, and that is deliberate.

**On reliability, which is the one that fails.** The failure is a clinical report scanned
**without a text layer**: the agent extracts nothing and declares itself `FAILED`, which is
the correct behaviour. With N = 16 real cases, a single failure is 6.2 points. In other
words: the figure sits below target by one case, and fixing it is OCR, not architecture. It
is published this way rather than raising N with synthetic cases until the percentage looks
good.

**And one that is not in the brief but holds up the rest:** the `mesh-agent` over **120
meshes** from Teeth3DS+ gives **100%**. Both figures travel with their N beside them on
purpose: a small N that is declared is defensible; hidden behind a percentage, it is not.

---

## 2 · What is finished and measured

**The container.** A real case closes at **12 entries, 18 assets (13 external), UOS-Core +
UOS-Vol conformance, 0 errors, 18 views**. Acquired data does not travel inside: the CBCT
series is declared by its content address, with a per-slice hash for its 397 slices, and
there are tests that check this by removing a slice, slipping in an extra one and altering
one.

**The agent pipeline.** Ingestion of the four modalities → geometric fusion (ICP, rms
0.67 mm) → semantic fusion → segmentation → composition → export, runnable end to end over
a real clinical case with a single command.

**The printable composite.** An arch closed as a watertight solid (12,025 closing faces,
11,936 mm³, 3.2 s) plus one STL per tooth with a measured crown and a reconstructed root.
Each file declares its provenance in the STL header itself.

**The viewer.** Opens a `.uos` in the browser without uploading anything: mesh with
anatomical colour, Gaussian field layers, named views, a card per tooth and highlighting of
the selected tooth.

**The discipline around negative results.** Three things were implemented, measured, did
not work and **were withdrawn rather than shipped**: the per-tooth apical axis (60° of
deviation), the second CBCT segmenter (it won the benchmark and lost the real task), and
distance-based hole filling (it worked on real data for a circular reason, which is the
worst kind of success). All three are documented with their measurement.

---

## 3 · What is unresolved

### 3.1 · FDI segmentation — the main gap

**11 of 14 teeth can be discarded on anatomical grounds. Upper bound on correct ones:
21%.** The full card, with the table, the two criteria and the measured causes, is in
`docs/research/segmentacion-fdi-escaner.md`.

In one line: the model was trained on Teeth3DS+ (0.932 per-tooth FDI **on its own test
set**), there are no labels for these patients to measure accuracy against, and the
dominant error is not interproximal contact but that **the crown eats into the gingival
margin and the attached gingiva**. It shows up as "not all the gum gets coloured": it is
not uncoloured, it is coloured as tooth.

What is missing is not tuning a threshold: it is **colour**, the signal a clinician uses to
see where enamel ends, and which today never reaches the model. And it is inside the
container already, in the clinical photographs — measured in
`docs/research/frontera-encia-desde-foto.md`.

### 3.2 · The volume looks like scattered points, and it is a subsampling bug

Measured, with a concrete cause that is **not** a fundamental limitation:

- the `cbct-agent` seeds σ = **half a voxel on each axis** — (0.075, 0.075, 0.225) mm over
  a voxel of 0.15 × 0.15 × 0.45. That part is correct;
- but the volume brings ~**12 million** hard-tissue voxels and the cap is 1.5 M, so the
  agent subsamples with `occupied[::step]`, `step = 9`, leaving 1,341,421;
- `[::9]` slices an array in **raster order**, so it eats eight out of every nine **along a
  single axis**. Measured spacing between consecutive Gaussians within a row: **1.35 mm in
  73% of cases**;
- which makes **σ/spacing = 0.056 on that axis**: the Gaussians are eighteen times smaller
  than their own spacing. Probed inside the densest bone, the field runs from 1.21 at the
  centre of a Gaussian to 0.016 between them — **99% ripple**.

In other words: the exported field is not a cloud, it is **dense planes 1.35 mm apart**.
And this is not the price of the data being measured: **it is that the subsampling is
anisotropic by construction and σ is not rescaled with it.**

Two fixes, both small:

1. **Decimate in space, not in the raster.** Keeping one voxel in two *on each axis* is the
   same 8:1 but isotropic: 0.30 mm on all three axes instead of 1.35 on one.
2. **Scale σ with the decimation factor** (×`step`^⅓). Today it does not grow at all.

With both, σ/spacing returns to ~0.5 and the field reads as a volume **without ceasing to
be a measurement**.

⚠️ **And a conformance consequence, quite apart from how it looks:** the field that travels
in the container **is not the volume, it is a one-in-nine subsample**. The agent knows this
— it lowers its confidence to 0.9 because of it — but the `.gs.json` sidecar **does not
declare it**. Anyone measuring on that field is measuring a subsample without the file
saying so.

In the viewer, meanwhile, each sprite is drawn inflated to the cloud's measured spacing.
That is a patch, and an insufficient one: that spacing (0.212 mm) is the distance to a
neighbour in another row, not the real 1.35 mm gap.

### 3.3 · The reconstructed root is a blob

~2,000 Gaussians per tooth through an alpha complex. Good enough to print context; not good
enough to measure. And 12 of the 14 roots come out longer than their type admits: this is
**declared**, not corrected, because trimming them to anatomical length is a *prior* and not
a measurement, and measuring root length on the result would be measuring what was assumed.

### 3.4 · The rest, briefly

- **`value_range` is still `null`**, with its warning. The measured alternative is sweeping
  259 MB per export. It closes when the volumetric renderer lands (§3.2).
- **`verified_by` on registrations is `null`** in every case: nobody has signed any of
  them. The viewer paints it as "unverified", which is what it is.
- **The gingival margin is not anchored** to the cervical curvature ridge. The cleanup is
  forbidden from moving it, and that prohibition is correct.
- **No clinical accuracy figure exists**, because there is no ground truth for these
  patients. Everything published is either an upper bound or a consistency check.

---

## 4 · What the next step would take, in order

The order matters: the first three unblock things that today cannot even be measured.

1. **The tooth-gum boundary, from the colour of the clinical photographs.** Unblocks §3.1,
   and it is the **measured** route: an Otsu threshold on `a*` separates tooth from gum at
   3.4–4.3 σ across the four arch photographs the container **already carries**, and traces
   the cervical scalloping tooth by tooth. What is missing is not the colour: it is
   **camera pose**, which today goes out as `projection: null` with the field already
   defined in the schema. Card: `docs/research/frontera-encia-desde-foto.md`.
2. **Annotate ten arches of our own.** Without this there will never be an accuracy figure,
   only bounds. It changes more than another training run on Teeth3DS+ — and that is
   measured in `docs/research/segmentacion-diente-cbct.md`, where the better model lost on
   the real task.
3. **Fix the field's subsampling** (§3.2): decimate in space rather than in the raster,
   scale σ with the factor, and declare it in the `.gs.json`. It is small, and it is the
   actual cause of the field looking like scattered points.
4. **A volumetric renderer in the viewer.** Unblocks §3.4 and is what turns the container's
   CBCT into something a clinician looks at rather than an ornament.
5. Anchor the margin to the cervical curvature (§3.4).
6. OCR for scanned reports: it is the single failure behind the reliability figure (§1).
7. A second real visit, to exercise `visits[]` properly and turn longitudinal follow-up
   into a subtraction.

---

## 5 · What someone takes away from here

A container format with a published schema, a validator, chained provenance and a rule that
holds up on its own: **the measured and the inferred do not mix, and the inferred can be
deleted without breaking the case**. That is what makes a stage that does not work — today,
segmentation — a separable piece rather than a contamination of the deliverable.

And a pipeline that goes from a clinic's raw files to that container, with the brief's four
figures measured, three of the four met, and the fourth failing by one case and for a named
reason.

What nobody takes away: a clinically validated product. It was never promised, and there
isn't one.

---

## Project milestones

| Week | Milestone | |
|---|---|---|
| 2 | Multi-agent architecture review and clinical attribute schema for the Digital Twin | ✅ |
| 4 | PoC demo: ingestion agents + first version of the Digital Twin on synthetic data | ✅ |
| 6 | Integrated system: fusion and export agents, STL regeneration from the Digital Twin | ✅ |
| 8 | MVP tested, preliminary validation with the partner organisation, final technical documentation | 🟡 |

🟡 **Week 8 is half done, and the missing half is the half that does not depend on code.**
The MVP is tested and the technical documentation is closed (this document); what has not
happened is the **validation with the partner organisation**, which needs someone outside
the project to open a `.uos` we did not write ourselves.
