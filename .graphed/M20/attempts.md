# M20 attempts — graphed (the partitioned-write base; user-directed refactor of parity P3.6)

## Iteration 0 — TEST_AUTHORING/TEST_SANITY/IMPLEMENTING — 2026-06-10 (freeze-M20-0)

- USER DIRECTION: deferred partitioned writing gets a format-agnostic COMMON BASE in graphed,
  specialized by clients (parquet in the backends, ROOT in the reader fork) — not borrowed
  parquet tools.
- frozen suite tests/frozen/m20 (7 tests); NON-VACUOUS (collection fails on the missing module).
- graphed/write.py: write_plan + key-ordered SequentialRunner + _LocalResources (moved from
  parquet.py), file_bases over GENERIC keys (uri strings or (uri, tree) pairs), blind_part_index
  (no-I/O, non-blind refused), exact step_of reconstruction, suffix-explicit part_path.
  graphed/parquet.py becomes the parquet SPECIALIZATION: derive_part_index/part_path wrappers;
  write_plan/SequentialRunner/file_bases are ALIASES of the base — pinned identical by the m20
  suite, so the frozen m15 suites (here and in both backends) stay green unchanged.
- gates: frozen_tests 183/183 PASS · coverage 94% (>=90, branch) · ruff+format clean ·
  mypy --strict clean · determinism green · sphinx -W clean. Backend regressions green
  (graphed-numpy 334, graphed-awkward 221).
