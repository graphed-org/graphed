# Frozen acceptance suite — M17 (graphed-awkward): structure-op parity

dask-awkward parity plan, milestone M17 (P1). Traceability:

| Test file | Verifies | Plan item |
|---|---|---|
| `test_structure_ops.py` | the missing structure tier — `sort/ravel/run_lengths`, `mask/is_none/singletons/pad_none(clip)`, `fill_none∘pad_none`, `unflatten`, `to_regular/from_regular`, `full_like/nan_to_num/isclose`, `argcombinations/argcartesian`, `without_field`, `values_astype`, multi-output `broadcast_arrays`/`unzip` as tuples of deferred arrays, the M17 `fields`-subset getitem evaluated, eager `to_list` sugar — each with a record-time typetracer form and EXACT (`ak.array_equal`) agreement with ak.* on raw data; structure ops stay fusible; projection through structure-only touches stays honest | P1 |

Reuses `m16_helpers` (the suite directory is on the pythonpath).
