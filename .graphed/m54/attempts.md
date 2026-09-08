# m54 implementer attempts (recorded retroactively; the journal is `graphed-workdir/systematics-vary-worklog.md`)

## Target
Frozen `tests/frozen/awkward/m54` (51 tests, tag `freeze-m54`, test-author commit 047e794).

## Iterations
1. `graphed.BoundMethod` from `Array.__getattr__` via the backend's `attribute_kind`; one `method`
   op with JSON `args`/`kwargs`; tuple results one node per `index`; `Varied` via `expand`.
2. Review folds D1–D3 / E1–E2: CLOSED property side of `attribute_kind` (data descriptor or
   `cached_property`), dict keys sorted at every nesting level, the `{"$": i}` marker shape refused,
   `Session.record_op` refuses inputs from another Session.
3. Attrs fold: every behavior re-wrap passes `attrs=` (typetracer wrap, `_ops.apply`, `with_name`).
Landed as graphed#22 (1598395); pool witness in graphed-executors#12.
