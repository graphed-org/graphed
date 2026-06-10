# Frozen acceptance suite — M16 (graphed-awkward): P0 foundation

dask-awkward parity plan, milestone M16 (`dask-awkward-parity-plan.md`). Traceability:

| Test file | Verifies | Plan item |
|---|---|---|
| `test_ufunc_tier.py` | the full M11 canonical ufunc tier on JAGGED arrays (~35 unary + 20 binary cases incl. scalar/reflected), typetracer inference at record time, materialization equal to the ufunc on raw awkward | M16.1 |
| `test_structural_rule.py` | the M12 structural rule on awkward reducers: axis>=1 (per-event) records a FUSIBLE op, axis None/0 a BOUNDARY reduction — both directions witnessed per kind; softmax always partition-local; two-array reducers follow the rule | M16.2 |
| `test_reducers.py` | the 13 missing reducers (`mean/std/var/min/max/prod/count_nonzero/ptp/moment/softmax/corr/covar/linear_fit`) + ddof, evaluated exactly as ak.* on raw data; projection tracks through them | M16.3 |
