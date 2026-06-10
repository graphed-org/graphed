# Frozen acceptance suite — M15 (graphed-awkward): partitioned parquet I/O

dask-awkward parity plan, milestone M15.2 (`dask-awkward-parity-plan.md` in the superproject),
specializing the `graphed.parquet` common base. Traceability:

| Test file | Verifies | Plan item |
|---|---|---|
| `test_parquet_io.py` | multi-file/glob `from_parquet` materializing the concatenation; the form comes from the ARROW SCHEMA ALONE (witness: a garbage second file constructs + types, only materialization fails); `columns=` filtering; the M3 single-file shape still holds; blind and eager partition reads tile the dataset and project columns; `to_parquet` writes per partition through the COMPILED IR with deterministic key-ordered part names; the compute-disabled task graph run later equals the enabled run bit-for-bit (R15.4); the writer's read list is wired from the projection (cannot drift, R15.3); multi-source arrays rejected loudly (R15.9); in-memory sources write by steps | M15.2 |

pyarrow is required here in practice (awkward's own parquet path uses it); the suite still
`importorskip`s for matrix cells without wheels.
