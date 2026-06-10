# Frozen acceptance suite — M14 (graphed): multi-input blockwise externals

dask.array user-facing parity, tier P3.8 (`dask-array-parity-plan.md` in the superproject; P3.9
and P4 are Phase 2 by user decision). Traceability:

| Test file | Verifies | Plan item |
|---|---|---|
| `test_apply.py` | `graphed.apply(fn, *arrays)`: ONE External node with N inputs carrying the backend's PayloadDescriptor (opaque callables stay flagged preservation risks); materialization calls `fn` on all inputs; single-input `apply` interns with the M2 `map`; session/arity validation; the signature-aware gufunc idiom stays OUT of the base Array (graphed-numpy's M14) | P3.8 |

`m14_toy.py` carries data through toy sources so materialization is observable.
