# M12 attempts — graphed (dask.array parity P1: reduction/scan infrastructure)

## Iteration 0 — TEST_AUTHORING/TEST_SANITY — 2026-06-10 (freeze-M12-1)

- First authored numpy-flavored (freeze-M12-0, methods on the base Array); superseded pre-push by
  the human-directed M11 factorization (see .graphed/M11/attempts.md): the suite was re-authored
  to pin the protected `_reduction`/`_scan`/`_norm_axis` infrastructure through a minimal backend
  proxy subclass (`m12_toy.ReduceArray`), plus the pin that NO reduction method leaks onto the
  base Array. Verified NON-VACUOUS (suite fails with the infrastructure stashed).
- One helper defect found during the first pass and fixed pre-freeze (recorded): `recorded()`
  selected the node by output marking, but `mark_output` accumulates across `serialized_ir` calls
  within one session; it now selects by the array's node id (strictly more precise).

## Iteration 0 — IMPLEMENTING/REVIEW — 2026-06-10

- `_norm_axis` (negative axes resolved against the form's `ndim`; refused without it),
  `_reduction` (axis None/0 -> boundary reduction node, axis>=1 -> fusible op; keepdims/ddof
  recorded only when non-default), `_scan` (always fusible) — protected infrastructure only.
- gates: frozen_tests 152/152 PASS · coverage 94% (>=90, branch) · ruff+format clean ·
  mypy --strict clean · determinism green · sphinx -W clean.
