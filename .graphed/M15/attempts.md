# M15 attempts — graphed (parquet I/O common base, dask-awkward parity plan M15.1)

## Iteration 0 — TEST_AUTHORING/TEST_SANITY — 2026-06-10 (freeze-M15-0)

- frozen suite authored (tests/frozen/m15: 10 tests); verified NON-VACUOUS (collection fails on
  the missing graphed.parquet module). pyarrow enters as the optional `graphed[parquet]` extra +
  a dev dependency; the suite importorskips it.

## Iteration 0 — IMPLEMENTING/REVIEW — 2026-06-10

- graphed/parquet.py: deterministic discovery (dir/glob sorted, explicit order kept);
  metadata-only num_rows/schema; blind/eager make_partitions on the first-class blind Partition
  (R7.9 — blind partitioning witnessed against NONEXISTENT paths); resolve_partition;
  writer-side part-index derivation (R15.9: per-task state bounded by files; exact step
  reconstruction — entry_start alone is not invertible, n=5/steps=3); lazy deferred_source with
  the file list in the source identity; write_plan (R15.4: compute-disabled task graph whose
  later run IS the enabled mode) + dependency-free SequentialRunner (key-ordered; any R7
  executor accepts the same plan).
- mypy boundary: pyarrow's inline types drag sphinx sources (3.12-only `type` statements) into
  the 3.11-pinned strict run; pyarrow.* is followed-skip like numpy in graphed-numpy (own
  sources stay --strict).
- gates: frozen_tests 174/174 PASS · coverage 93% (>=90, branch) · ruff+format clean ·
  mypy --strict clean · determinism green · sphinx -W clean.
