# Frozen acceptance suite — M12 (graphed): axis-aware reductions + scans

dask.array user-facing parity, tier P1 (`dask-array-parity-plan.md` in the superproject).
Traceability:

| Test file | Verifies | Plan item |
|---|---|---|
| `test_reduction_surface.py` | reduction methods (`sum/prod/mean/std/var/min/max/any/all/argmin/argmax`) + numpy-function forms + nan-variants; the structural rule: `axis None/0` records a **boundary reduction node**, `axis>=1` records a **fusible op**, scans always fusible; negative-axis normalization against the form's `ndim`; `keepdims`/`ddof` params recorded only when non-default so defaults intern; unsupported kwargs raise | P1.4 |

`m12_toy.py` is the suite's permissive backend; `recorded()` reads the (kind, name, params) of a
node back from the serialized IR — the pins are on what is RECORDED, not on any backend's
evaluation (that is graphed-numpy's M12 suite).
