# Frozen acceptance suite — M15 (graphed): the backend-agnostic parquet base

dask-awkward parity plan, milestone M15.1 (`dask-awkward-parity-plan.md` in the superproject).
Traceability:

| Test file | Verifies | Plan item |
|---|---|---|
| `test_parquet_base.py` | deterministic discovery (file/dir/glob sorted; explicit list order kept); metadata-only row counts/schema; **blind partitioning opens NO file** (witnessed with nonexistent paths — R7.9/R15.3); eager partitions cover each file contiguously and blind resolution reproduces them; writer-side part-index derivation from the partition alone (R15.9 — no global map pickled per task, including the non-invertible n=5/steps=3 case); deferred sources are lazy (counting-loader witness: nothing read at record, one read at materialize) with the file list in the source identity; the deferred write plan: compute-disabled is a task graph whose later run equals the enabled run (R15.4), the sequential reference runner is key-ordered; missing pyarrow names the `graphed[parquet]` extra | M15.1 |

`m15_toy.py` supplies the permissive backend + the counting loader. pyarrow is an optional
dependency: the suite `importorskip`s it (CI installs it via dev extras where wheels exist).
