# `fusion-agents` — dental Digital Twin fusion

Implements the fusion layer decided in
[ADR 004](../../docs/architecture/004-fusion.md). The pipeline splits it into **two
stages separated by segmentation**:

```
[GEOMETRIC FUSION]   mesh↔CBCT registration (ε band) · colour · does NOT use FDI
   ▼
[SEGMENTATION]       populates region_id (FDI)
   ▼
[SEMANTIC FUSION]    anchors pH/observations to the FDI · does NOT use geometry
```

| Agent | Stage | Status |
|---|---|---|
| `SemanticFusionAgent` | FDI anchoring | **implemented** |
| `GeometricFusionAgent` | mesh↔CBCT registration | **implemented** (fine stage) |

They are **two agents and not one** because another stage runs between them, and
because their material and their acceptance criteria are different.

## Usage

```python
from fusion_agents import GeometricFusionAgent, SemanticFusionAgent, insert_snapshot

# 1 · geometric registration (before segmentation)
geo = GeometricFusionAgent(epsilon_mm=0.5)
out = geo.fuse(snapshot, source=puntos_malla, target=puntos_cbct)
out.snapshot.provenance.transform.inverse()      # registration is reversible

# 2 · semantic anchoring (afterwards)

agente = SemanticFusionAgent()                      # HITL threshold 0.7 by default
out = agente.fuse(snapshot, detected={"46": 0.95, "47": 0.91})

if out.hitl_required:
    print(out.hitl_reasons)                         # what to look at, and why
twin = insert_snapshot(twin, out.snapshot)          # idempotent by acquisition_id
```

`detected` is the `FDI → confidence` map produced by the `segmentation-agent`. It is
passed explicitly instead of reading the Gaussian field from the store: there are ~14
codes, and loading millions of primitives to obtain them would be absurd. As a bonus,
the agent stays testable without a store and without a GPU.

## Registration

The algorithm lives **behind a `Protocol`** (`Registrar`), not inside the agent: that
way it can be swapped without touching the contract, and the agent is tested with a
trivial registrar. The default is **multiscale ICP** in numpy + scipy.

**What is missing, stated plainly.** The **coarse** stage from the ADR — RANSAC over
FPFH descriptors — is not there: it is the one that provides an initial pose when the
clouds are badly misaligned, and it depends on Open3D. Consequence: **the ICP here
converges only if the initial pose is already reasonably close**. With a mesh derived
from the volume itself that holds by construction; with an intraoral scan and a CBCT
captured separately, it has to be measured before calling the registration good.

**Confidence comes from the residual**, `clamp(1 − rms/ε, 0, 1)` with ε = 0.5 mm. The
default gate (0.7) amounts to requiring `rms ≤ 0.3·ε`. A registration outside the band
**does not fail**: it is delivered with low confidence and asks for review. That is the
difference between a declared failure and a silent one.

## Colour

It comes from the **mesh**, not from the photos: every Gaussian inside the ε band takes
the colour of its nearest vertex (`transfer_surface_color`). Photos are left out because
notebook 07 measured that the photo↔mesh error is **non-rigid** — ICP stalled at
IoU ≈ 0.55 due to the perspective of an uncalibrated photo — so it is not the same
problem as rigid CBCT↔mesh registration.

With a bare mesh (STL) or a flat grey *placeholder* (Teeth3DS+) the result is **absence
of colour**: a valid answer, not a bug. And `transfer_color` **persists nothing** —
`color_superficie` lives in `GaussianPrimitive`, and the snapshot only stores a
hash reference to the field; materialising it belongs to whoever owns the
`ArtifactStore`.

## The three decisions behind semantic anchoring

**Confidence is the weakest link**, `min(...)` and not the product. Chaining products
sinks everything below the threshold by arithmetic, not by real distrust. Anchoring a pH
value to a tooth cannot be more reliable than knowing which tooth it is.

**Faced with a conflict, the agent does not pick a winner.** If the report references a
tooth that segmentation did not find, the FDI **from the report** is kept — it is the
clinical source — confidence is set to **0.0**, and with that it falls below the gate
and goes to human review. There is no new field: *the confidence is the mark*.

The reason is measured: the model's dominant error is
[drifting to the neighbouring tooth](../../notebooks/exercise-point-transformer-teeth3ds.md),
so in a disagreement it is the less reliable side — but the report is not infallible
either. Resolving it silently would be the failure that
[ADR 003](../../docs/architecture/003-verification-fault-tolerance.md) flags as the
worst: silent and irreversible, over a clinical value.

**It never mutates the input snapshot** and it preserves the `acquisition_id`, which is
the identity of the visit. That is why re-running fusion **replaces** instead of
appending, and does not inflate the patient history with visits that never happened.

## Traceability

Each observation's `Provenance` **keeps the report's own** (`source_file`, `modality`,
`agent`): the value still comes from the PDF, and losing that would break traceability
back to the original. Only `confidence` is rewritten.

Who performed the fusion is recorded in the **snapshot's** `Provenance`, which is the
value this agent does derive.

## Tests

```bash
uv run pytest packages/fusion-agents
```

56 tests, 97% coverage. For registration they cover the **unit level of ADR §2.7**: a
**known** transform is applied to a cloud and recovering it is required — the reference
truth is exact because the test fabricates it, so a failure belongs to the algorithm and
not to the data. For semantic anchoring: the weakest-link rule, the FDI conflict, the
configurable gate, non-mutation, time-series idempotency and the *fail-loud* contract —
including that **quarantine must not leak clinical data**: it stores the
`acquisition_id` and the traceback, never the pH.
