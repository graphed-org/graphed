# Frozen acceptance suite — M11 (graphed): full elementwise tier + backend-idiom factorization

dask.array user-facing parity, tier P0 (`dask-array-parity-plan.md` in the superproject), under
the design rule that `graphed.Array` stays **backend-idiom-neutral**: it carries only what numpy
and awkward semantics share (operators, `__array_ufunc__`, field access) plus protected
infrastructure; idiomatic surfaces live in the backend packages (`graphed_numpy.NumpyArray`
methods/properties; `graphed_awkward.gak` functions). Traceability:

| Test file | Verifies | Plan item |
|---|---|---|
| `test_ufunc_surface.py` | every single-output numpy ufunc records one canonical backend-agnostic op via `__array_ufunc__` (ufunc application is common to BOTH idioms); aliases (`degrees`/`rad2deg`, …) intern to the SAME node; new operator dunders (`// ^ << >> +x`, reflected `** & \| ^`); repeated application interns to zero new nodes | P0.2 |
| `test_array_factory.py` | the `array_type` factory: a backend's proxy subclass is returned by every Session builder; the base Array has NO numpy-idiomatic members (`sum`/`shape`/`__array_function__` must not leak in — the factorization pin); `_form_meta` infrastructure delegates to metadata-carrying forms and falls back to M3 field recording | P0.1/P0.3 |

`m11_toy.py` is the suite's permissive backend + a metadata-carrying form + an `IdiomBackend`
whose proxy subclass is built only from the protected infrastructure. graphed itself must remain
numpy-free (plan §A.4): all dispatch is by `__name__`. The numpy idiom (`.shape`, `.sum()`,
`__array_function__`) is pinned in graphed-numpy's M11 suite instead. The M2/M3 frozen suites stay
authoritative for the op surface they already pin.
