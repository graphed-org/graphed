# Frozen acceptance suite — M15 (graphed-numpy): rectilinear parquet I/O

dask-awkward parity plan, milestone M15.3 (`dask-awkward-parity-plan.md` in the superproject) —
the numpy specialization of the `graphed.parquet` base, RECTILINEAR ONLY by user direction.
Traceability:

| Test file | Verifies | Plan item |
|---|---|---|
| `test_parquet_io.py` | multi-file `from_parquet` records a record source (schema → `NumpyForm` fields) materializing the concatenation; M5 field-touch projection works on parquet sources; `columns=` filter; **jagged/nested columns refused at construction naming the column and pointing at graphed-awkward**; blind+eager partition reads tile the dataset with column projection; `to_parquet` via the compiled IR with deterministic part names; disabled/enabled write consistency (R15.4); the writer's read list wired from the projection; multi-source and non-1-D outputs rejected; in-memory record sources write by steps; IR determinism | M15.3 |

This amends the R16.7 deferral: parquet I/O (alone) was pulled into the MVP for graphed-numpy by
user decision on 2026-06-10; the rest of P3.9/P4 stays Phase 2.
