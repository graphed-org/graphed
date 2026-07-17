# Frozen acceptance suite — M13 (graphed): common indexing surface

dask.array user-facing parity, tier P2 (`dask-array-parity-plan.md` in the superproject), under
the M11 factorization. Traceability:

| Test file | Verifies | Plan item |
|---|---|---|
| `test_getitem_surface.py` | `a[slice]` and `a[int]` on the base proxy (common to numpy AND awkward): both record **boundary reduction nodes** (they consume/restructure the partitioned axis); present-only slice params so equal slices intern; mask/field keys unchanged from M3; idiom-specific keys (tuples, bools, non-int slice fields) refused; no manipulation method leaks onto the base Array | P2.6 |

`m13_toy.py` mirrors the m12 helper. The numpy manipulation idiom (tuple subscripts, `reshape`,
`take`, `where`, `concatenate`, `unique`, `histogram`, …) is pinned in graphed-numpy's M13 suite.
