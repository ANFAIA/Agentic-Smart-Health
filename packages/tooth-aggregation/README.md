# `tooth-aggregation` — point → tooth aggregation

A point cloud segmentation model returns one label **per point**. The
`segmentation-agent` has to deliver **teeth**: instances, each one with its FDI code.
This package is that jump, and only that jump.

Extracted from [`notebooks/exercise-point-transformer-teeth3ds.ipynb`](../../notebooks/exercise-point-transformer-teeth3ds.md)
(section A4), where it was measured over the full Teeth3DS+ dataset.

## No `torch`, on purpose

The model *forward* pass (GPU, `torch` + `pyg`) belongs to the **caller**, which passes
the already computed log-probabilities in here as an `ndarray`. Aggregation is pure
geometry and combinatorics: `numpy` + `scipy`.

This is not a style detail. The GPU stack lives in a separate venv
(`~/.venvs/dental-gpu`, see [`notebooks/README.md` §04](../../notebooks/README.md)) that
sits **outside** the `uv` workspace. A package that imported `torch` would neither
install here nor be testable in CI.

## Usage

```python
from tooth_aggregation import aggregate_teeth

# logprob (N, C) comes out of the model; points (N, 3) is the mesh.
dientes = aggregate_teeth(points, logprob, codes={1: 11, 2: 12, ...})

for d in dientes:
    print(d.fdi, d.size, d.confidence)
```

Individual pieces, if you need finer control: `connected_labels` (instances),
`merge_fragments` (merging), `assign_unique` (FDI uniqueness), `typical_spacing`.

## Two measured facts that shape how this is used

**The honest metric is per-TOOTH accuracy, not per-point.** On Teeth3DS+, per-point
accuracy *underestimates* per-tooth identification. The agent card must declare the
second one.

**`enforce_unique` ships disabled by default**, and not out of generic caution. The
"one FDI per arch" constraint presupposes *one instance = one tooth*; since detection
splits teeth into fragments, that premise is false and forcing a different code per
fragment **invents** errors. Measured over Teeth3DS+: enabling it **without merging
first makes** per-tooth accuracy **worse**; with merging almost everything is recovered,
but it still **did not beat plain majority voting**. Only enable it if you measure that
merging really does leave one instance per tooth in your case.

> The concrete figures live **only** in the
> [experiment card](../../notebooks/exercise-point-transformer-teeth3ds.md), which is the
> single source: they are re-measured on every run and duplicating them here would let
> them drift.

## Tests

```bash
uv run pytest packages/tooth-aggregation
```

Synthetic data (regular grids, so the thresholds can be reasoned about by hand): that
distant blobs do not merge, that neighbours of a different class do not either, that
merging is transitive, and that the Hungarian algorithm finds the **global optimum**
and not the greedy solution.
