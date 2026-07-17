# Frozen acceptance suite — M14 (graphed-numpy): apply_gufunc

dask.array user-facing parity, tier P3.8 (`dask-array-parity-plan.md`; P3.9 + P4 are Phase 2 by
user decision). Traceability:

| Test file | Verifies | Plan item |
|---|---|---|
| `test_apply_gufunc.py` | gufunc signatures give opaque callables record-time forms: core-dim binding against operand forms (mismatch/unbound/rank/arity errors at record time), declared output dtype, whole-array `vectorize=False` evaluation matching the callable applied directly, the PayloadDescriptor carrying the signature as `io_schema`, and the M5 on-fail projection policy through gufunc nodes | P3.8 |
