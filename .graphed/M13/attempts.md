# M13 attempts — graphed (dask.array parity P2: common indexing surface)

## Iteration 0 — TEST_AUTHORING/TEST_SANITY — 2026-06-10 (freeze-M13-0)

- frozen suite authored under the M11 factorization from the start; verified NON-VACUOUS
  (4/6 fail against pre-M13 code; the 2 passing tests pin already-frozen M3 behavior and the
  factorization invariant).

## Iteration 0 — IMPLEMENTING/REVIEW — 2026-06-10

- base `__getitem__` gains the COMMON keys only: `slice` (present-only start/stop/step int
  params; equal slices intern) and `int` — both consume/restructure the partitioned axis, so
  both record BOUNDARY reduction nodes per the M12 structural rule. bools, non-int slice
  fields, and tuple subscripts (the numpy idiom -> graphed-numpy) are refused.
- gates: frozen_tests 158/158 PASS · coverage 94% (>=90, branch) · ruff+format clean ·
  mypy --strict clean · determinism green · sphinx -W clean.
