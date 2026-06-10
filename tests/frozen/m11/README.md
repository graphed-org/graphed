# Frozen acceptance suite — M11 (graphed): full elementwise tier + array metadata + `__array_function__`

dask.array user-facing parity, tier P0 (`dask-array-parity-plan.md` in the superproject).
Traceability:

| Test file | Verifies | Plan item |
|---|---|---|
| `test_ufunc_surface.py` | every single-output numpy ufunc records one canonical backend-agnostic op via `__array_ufunc__`; aliases (`degrees`/`rad2deg`, …) intern to the SAME node; new operator dunders (`// ^ << >> +x`, reflected `** & \| ^`); repeated application interns to zero new nodes | P0.2 |
| `test_array_meta.py` | `.shape/.dtype/.ndim` delegate to the form when it carries metadata (no graph growth), and fall back to M3 field recording when it does not | P0.1 |
| `test_array_function.py` | `__array_function__` routes `np.sum` to the canonical `sum` reduction (interning with the method form); unsupported numpy functions raise `TypeError` | P0.3 |

`m11_toy.py` is the suite's permissive backend + a metadata-carrying form. graphed itself must
remain numpy-free (plan §A.4): all dispatch is by `__name__`. The M2/M3 frozen suites stay
authoritative for the op surface they already pin.
