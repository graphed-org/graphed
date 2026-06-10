# M11 attempts — graphed (dask.array parity P0: ufunc tier + array metadata + __array_function__)

## Iteration 0 — TEST_AUTHORING/TEST_SANITY — 2026-06-10

- frozen suite authored at `tests/frozen/m11/` (README + m11_toy + 3 test files, 68 tests);
  verified NON-VACUOUS against pre-M11 code: 66/68 fail for the right reasons (unmapped ufuncs
  return NotImplemented from `__array_ufunc__`; `.shape` records a field op instead of answering;
  `np.sum` falls into numpy's method-dispatch and dies on a recorded field op).
- freeze: freeze-M11-0.

## Iteration 0 — IMPLEMENTING/REVIEW — 2026-06-10

- `_UFUNC_TO_OP` extended to the full single-output ufunc tier (~85 names, aliases canonicalized);
  new dunders (`__floordiv__ __xor__ __lshift__ __rshift__ __pos__` + reflected `** & | ^ << >>`);
  `.shape/.dtype/.ndim` properties delegating to the form with M3 field-recording fallback;
  `__array_function__` + `_ARRAY_FUNCTIONS` table (np.sum -> canonical sum reduction).
- gates: frozen_tests 123/123 PASS · coverage 93% (>=90, branch) · ruff+format clean ·
  mypy --strict clean · determinism green (frozen byte-equality test) · sphinx -W clean.
- downstream regression: graphed-numpy, graphed-awkward, graphed-debug, graphed-exec-local,
  graphed-checkpoint, graphed-preserve, graphed-corpus, graphed-orchestrator suites all pass;
  uproot fork graphed tests 59/59 pass.
