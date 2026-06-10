# Frozen acceptance suite — M12 (graphed-numpy): reductions, creation, random, monoids

dask.array user-facing parity, tier P1 (`dask-array-parity-plan.md` in the superproject).
Traceability:

| Test file | Verifies | Plan item |
|---|---|---|
| `test_reductions_match_numpy.py` | every reduction kind × axis (None/0/1) × keepdims, nan-variants, `ddof`, any/all, scans — all evaluating EXACTLY as numpy; forms track the partitioned axis (global → scalar, axis 0 → concrete, axis ≥ 1 → `(None, …)`); text reductions fail at record time | P1.4 |
| `test_creation.py` | `zeros/ones/full/empty/arange/linspace` record deterministic named sources (identical creations intern to ONE node); `*_like` record fusible ops preserving the partitioned axis; `empty` is zeros (determinism); `from_array(chunks=)` is metadata-only; creation programs serialize byte-identically | P1.5 |
| `test_random_and_monoids.py` | seeded random sources: same seed → identical values AND identical IR bytes, successive draws distinct; `monoid(kind)` (chunk/combine/empty/finalize — the M7 process/combine/empty quadruple): any chunking and any tree shape agrees with numpy whole-array, `empty()` is a true identity, `ddof` supported, non-tree-reducible kinds refused | P1.4/P1.5 |

Reference semantics are numpy's own throughout. The M2/M5/M11 frozen suites stay authoritative
for the surfaces they pin.
