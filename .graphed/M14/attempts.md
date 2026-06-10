# M14 attempts — graphed (dask.array parity P3.8: multi-input blockwise externals)

## Iteration 0 — TEST_AUTHORING/TEST_SANITY — 2026-06-10 (freeze-M14-0)

- frozen suite authored under the M11 factorization; NON-VACUOUS (collection fails on the
  missing `graphed.apply`).

## Iteration 0 — IMPLEMENTING/REVIEW — 2026-06-10

- `graphed.apply(fn, *arrays, name=)`: one multi-input External "map" node via the existing
  `record_external` infrastructure; idiom-neutral function-over-arrays surface; single-input
  apply interns with the M2 `Array.map`; session/arity validation. The numpy signature-aware
  `apply_gufunc` lands in graphed-numpy's M14. P3.9 and P4 stay Phase 2 (user decision).
- gates: frozen_tests 164/164 PASS · coverage 94% (>=90, branch) · ruff+format clean ·
  mypy --strict clean · determinism green · sphinx -W clean.

## Iteration 1 — IMPLEMENTING — 2026-06-10

- Gap exposed by graphed-numpy's gufunc work: `record_external` did not wrap backend `op_form`
  errors into provenance-located GraphedTypeError the way `record_op` does, so record-time
  binding errors surfaced raw. Fixed (same wrap as record_op); all gates re-run green.
