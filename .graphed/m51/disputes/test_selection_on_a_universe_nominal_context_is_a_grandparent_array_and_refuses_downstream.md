# Test Dispute — `test_selection_on_a_universe_nominal_context_is_a_grandparent_array_and_refuses_downstream`

File: `tests/frozen/awkward/m51/test_selection_bridge.py`
Anchor: `vary-m51-C` (§9.1 case-2 universe/nominal → grandparent Array + (2a) refuse downstream)

## The test (lines 117-133), the offending assertion
```python
mask = _event_mask(events)  # = gak.num(events.Jet) >= 2  -> a DEFERRED graphed.Array
sel = events[mask]
projected = graphed.selection(graphed.nominal(sel))  # -> a DEFERRED graphed.Array (grandparent mask)
assert not isinstance(projected, graphed.Varied)  # PASSES (projected is a graphed.Array)
assert as_list(projected) == as_list(mask)  # <-- ERRORS: ak.to_list on a DEFERRED array
assert graphed.selection(graphed.nominal(events)) is None  # PASSES
with pytest.raises(GraphedError):
    ga.to_parquet(sel.Jet, str(tmp_path / "nope"), select=projected)  # PASSES (6.4b row-space refuse)
```

`as_list` (the fixture helper, `m51_write_fixtures.py`) is `ak.to_list(value)` — verbatim m50's
`as_list`, whose m50 docstring reads *"Materialized values compared elementwise; `ak.to_list` is
exact for both idioms."*

## The contradiction (measured)
`ak.to_list` cannot list a DEFERRED graphed array. graphed.Array carries no `__array__`/real
`tolist` (measured: `type(mask).__mro__ == [Array, object]`, `'__array__' not in dir`); its
`__getattr__` fabricates a `tolist` attribute that is itself an Array (not callable), so
`ak.to_list(mask)` raises `TypeError: 'Array' object is not callable` — INDEPENDENT of any
implementation of `graphed.selection`, because the RHS `as_list(mask)` fails on `mask` alone.
Deferred graphed arrays are non-materializing BY DESIGN (`Array.__iter__` raises "materialize
first").

The spec clause it contradicts: §9.1 binds case-2 to return *"an unvaried `Array` ... living in the
GRANDparent's row space"* — i.e. a deferred `graphed.Array` (not a materialized `ak.Array`), which
is required so it is passable to `to_parquet(select=…)` for the downstream (2a/6.4b) refuse the same
test exercises on the next line. So `projected` is necessarily deferred, and `mask` is deferred.

The established idiom (m48/m49/m50 frozen suites, EVERY conversion): `as_list` is called only on
`session.materialize(...)` results, e.g. m50 `test_selection_bridge`-adjacent:
`as_list(session.materialize(graphed.nominal(projected)))`. This test omits the `session.materialize`
wrapper on both operands — the sole such omission in the m51 suite (every other `as_list` call is on
a `read_varied` result, which returns already-materialized `ak.Array`s).

## Proposed correction (one line, behavior-preserving, matches the m50 idiom)
```python
assert as_list(session.materialize(projected)) == as_list(session.materialize(mask))
```
(`session = graphed.nominal(sel).<session>` or capture `session` from `events_context()` at the top,
which the test already binds as `_session`.) With this wrapper the assertion is measured to PASS
against the implemented `graphed.selection` (projected materializes to the grandparent mask, equal to
`mask`). All other assertions in this test already pass with the implemented feature.

## Impact / requested action
The `graphed.selection` case-2 FEATURE is implemented correctly (verified: `projected.node_id ==
mask.node_id`; `graphed.selection(graphed.nominal(events)) is None`; the downstream `to_parquet`
6.4b row-space refuse raises `GraphedError`). Only this one assertion's operand-materialization is
defective. Recommend a `m51-freeze-fixup` re-freeze wrapping the two operands in
`session.materialize(...)` (the decomposition §4 anticipates `m51-freeze-fixup`). I am NOT modifying
`tests/frozen/**`; the rest of the suite (34 other reds) is being implemented in parallel and this
test will pass once the operands are materialized.
