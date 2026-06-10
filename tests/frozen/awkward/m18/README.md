# Frozen acceptance suite — M18 (graphed-awkward): behaviors

dask-awkward parity plan, milestone M18 (P2). Traceability:

| Test file | Verifies | Plan item |
|---|---|---|
| `test_behaviors.py` | `gak.with_name` + `AwkwardBackend(behavior=...)`: behavior properties (`.pt`, `.mass` — vector's `Momentum4D`, the coffea pattern) work through PLAIN attribute access with typetracer forms at record time and exact evaluation; the buffer projection reports exactly the leaves a property reads (`pt → {px, py}`); plain fields stay fields; unknown attributes fail at record time without a behavior; `with_parameter`/`without_parameters`; named records survive structure ops | P2 |

`vector` is a dev/test dependency (`importorskip`); `attrs` (high-level metadata dicts) are not
recorded in the MVP — Phase 2 alongside the deferred `__awkward_function__` dispatch.
