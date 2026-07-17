# Graph-bloat note (feeds M4 reduction targets)

Why `graphed` exists: prior HEP task-graph systems recorded **one graph node per array operator**,
so the low-level graph grew with `ops × partitions × systematics` and graph *optimization* (not
execution) dominated wall time. This note quantifies the target M4 must hit.

## The mechanism (dask-awkward, `coffea_2023_postrelease` branch)

dask-awkward builds a high-level graph where each awkward call (`events.Jet.pt`, a mask, `ak.num`,
a histogram fill, …) is recorded as a layer, then lowered to a low-level graph with **one task per
(layer × partition)**. Two multipliers make this explode:

1. **Per-operator recording.** A single ADL query is ~20–60 operators (see the Required Operations
   Catalog); a realistic analysis (AGC ttbar) is several hundred operators per region.
2. **Systematics replication.** Each kinematic variation **re-runs the whole selection+observable
   subgraph**. With `S` systematic variations and `R` regions the operator count scales as
   `O(base_ops × R × S)`. The AGC with its full weight+JES/JER+b-tag set reaches **O(10⁴) nodes**.

dask's blockwise fusion that should collapse this was effectively **O(N²)** in the number of
layers, so as `N` grew the optimizer's own runtime came to dominate, and the residual low-level
graph was still large enough that the Python interpreter walking it was itself a cost (plan A.3
failures #3, #6, #7). The `master` (virtual-array) branch sidesteps the schedulable graph entirely.

## Order-of-magnitude estimate from the fixtures

Counting operators in our runnable fixtures (a proxy; the exact dask-awkward layer counts must be
measured against the two coffea-benchmark branches — a **tracked follow-up**, not run in this
network-free repo):

| Fixture | ~operators (base) | × regions | × variations | ~nodes (un-reduced) |
|---------|-------------------|-----------|--------------|---------------------|
| ADL q5 (dimuon+MET) | ~25 | 1 | 1 | ~25 |
| AGC ttbar (per region, nominal) | ~120 | 2 | 1 | ~240 |
| AGC ttbar (full systematics) | ~120 | 2 | 5 (here) … 50+ (real) | ~1.2k here, **O(10⁴) real** |

## What M4 must achieve (the reduction target)

`graphed` reduces **incrementally as the graph is built**, so the un-reduced O(10⁴) graph never
exists. The optimizer (equality saturation via `egg`, behind `RewriteEngine`) must collapse a run
to a **concise stage-graph whose node count scales with the number of stages, not with the
systematics count**. The binding M4 gates this note sets:

- the 10,000-node systematics graph reduces in **< 1 s** to `O(stage)` nodes;
- the AGC slice reduces to a stage-graph whose node count is **independent of the variation count**;
- a CI benchmark **fails on super-linear** reduction time across sizes {1k, 2k, 4k, 8k} — the guard
  against re-introducing the O(N²) blow-up.

## Follow-up (Phase 2)

Measure the actual dask-awkward low-level node counts for ADL q1–q8 and the AGC slice on both the
`coffea_2023_postrelease` (dask-awkward) and `master` (virtual-array) branches, and record the
measured explosion factor here to replace the operator-count proxy above.
