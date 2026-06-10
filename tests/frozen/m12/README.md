# Frozen acceptance suite — M12 (graphed): axis-aware reduction + scan infrastructure

dask.array user-facing parity, tier P1 (`dask-array-parity-plan.md` in the superproject), under
the M11 factorization: the base `Array` gains only protected infrastructure (`_reduction`,
`_scan`, `_norm_axis`); the numpy method/function idiom over it lives in `graphed_numpy`.
Traceability:

| Test file | Verifies | Plan item |
|---|---|---|
| `test_reduction_surface.py` | the structural rule: `axis None/0` records a **boundary reduction node**, `axis>=1` records a **fusible op**, scans always fusible; negative-axis normalization against the form's `ndim` (and refusal without it); `keepdims`/`ddof` params recorded only when non-default so defaults intern; nan-variant kinds are their own canonical ops; NO reduction methods leak onto the base Array | P1.4 |

`m12_toy.py` is the suite's permissive backend whose `ReduceArray` proxy is the minimal idiomatic
surface over the infrastructure; `recorded()` reads (kind, name, params) back from the serialized
IR — the pins are on what is RECORDED, not on any backend's evaluation (that is graphed-numpy's
M12 suite).
