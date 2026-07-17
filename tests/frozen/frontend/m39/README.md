# M39 frozen suite — graphed (repartition frontend + shuffle_plan builder)

Milestone **M39**. graphed owns the backend-agnostic frontend: the neutral `repartition` verb, the
`Array.repartition` physical rebalance, and the multi-source/multi-stage `shuffle_plan` builder + the
generic engine over `ShuffleBackend`. These run over a **toy** `ShuffleBackend` (graphed imports no
numpy/awkward, §A.4); the real two-backend witnesses live in graphed-awkward/graphed-numpy and the
executed backend-independence + block-count witnesses in graphed-exec-local. **Frozen — read-only.**

## Files → plan clause / theme

| File | Covers | Plan clause |
|---|---|---|
| `shuffle_backends.py` | toy `ShuffleBackend` + `Backend` + `ListSource` (the M2 `backends.py` analogue; §A.4-clean) | §3.3 |
| `test_repartition_api.py` | `graphed.repartition(by=)` records a hash `Exchange`; `Array.repartition(target_bytes=/n=)` records coalesce/count; idiom-neutral proxy; params drive structural identity | §3.1 |
| `test_shuffle_plan_builder.py` | `graphed.shuffle_plan(...)` emits a multi-stage `DurablePlanV2` (map_write → gather dependency edge); single-source `aggregate_plan` unchanged (control) | §3.2, §4.4 |

## Pinned surface (test-author decisions)

- `graphed.repartition(array, *, by=None, n=None, target_bytes=None)` (module verb) and
  `Array.repartition(*, n=None, target_bytes=None)` (method) — each records an `Exchange` node:
  `by=` → `scheme in {hash,range}`, `key=<field>`; `target_bytes=` → `scheme="coalesce",
  target_bytes=…`; `n=` → `parts=n`.
- `graphed.shuffle_plan(output, *, reduce, combine, empty, backend=None, steps_per_file=1) ->
  DurablePlanV2` (mirrors `aggregate_plan`). Its stage `kind`s include `"map_write"` and one of
  `{"gather_join","gather","reduce"}` with a dependency edge on the map stage.

## Non-vacuity

Pre-implementation: `graphed.repartition`/`Array.repartition`/`graphed.shuffle_plan`/`DurablePlanV2`
are absent → right-reason failures. The `aggregate_plan` two-source rejection is a control asserting
the single-source builder is not silently widened.
