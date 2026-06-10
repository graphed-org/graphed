# M17 attempts — graphed (record-subset getitem, dask-awkward parity plan)

## Iteration 0 — TEST_AUTHORING/TEST_SANITY/IMPLEMENTING — 2026-06-10 (freeze-M17-0)

- frozen suite tests/frozen/m17 (3 tests); NON-VACUOUS (list keys were refused pre-M17).
- base __getitem__ gains list-of-strings: ONE fusible "fields" op, canonical comma-joined params,
  order significant, equal subsets intern; tuples/mixed/empty stay refused.
- gates: frozen_tests 177/177 PASS · coverage 93% · ruff+format · mypy --strict · sphinx -W all
  clean.
