# Frozen acceptance suite — M11 (graphed-numpy): shape-aware forms + full elementwise tier

dask.array user-facing parity, tier P0 (`dask-array-parity-plan.md` in the superproject).
Traceability:

| Test file | Verifies | Plan item |
|---|---|---|
| `test_elementwise_full.py` | every canonical op the frontend records evaluates EXACTLY as the corresponding numpy ufunc (~40 unary + ~30 binary cases, array/array, array/scalar incl. reflected, comparisons, new operator dunders, `np.sum` via `__array_function__`) | P0.2 |
| `test_form_inference.py` | `NumpyForm` carries `(shape, dtype)` with `None` axis 0; record-time inference via zero-length meta arrays gives numpy's own promotion/broadcasting; ill-typed and unbroadcastable programs raise `GraphedTypeError` AT RECORD TIME; `.shape/.dtype/.ndim` answer without recording; the M2 `vector[<dtype>]` describe pin holds | P0.1 |
| `test_projection_and_determinism.py` | M5 column projection tracks through the new ops; two independent recordings of the same new-op program serialize to byte-identical reduced IR (M4/M8 determinism) | P0 gate |

Reference semantics are numpy's own — every assertion compares the deferred result against numpy
applied directly (no hand-written expected values to drift). The M2/M5 frozen suites remain
authoritative for the original 4-op surface and projection behavior they pin.
