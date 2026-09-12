# Test Dispute — `tests/frozen/awkward/m57/test_records.py::test_ambient_entries_keep_their_slots_kinds_and_order_across_a_value_free_join`

## The test

```python
values = m57_ambient_values(base.session, graphed.weight(projected))
```

where `projected = graphed.universe(base.ctx, "hf_up")`. `m57_ambient_values` calls
`graphed.labels(weight)`, and at a projection the ambient is the ONE adopted composed member — a
plain `Array`, not a `Varied`.

## The clause it contradicts

§2.3: "a row-space change adopts one composed container". The accessor contract (pre-m57, m48) then
refuses `graphed.labels` on a plain `Array`: `python/graphed/systematics/accessors.py` raises "a
plain Array carries no variations; graphed.labels takes a Varied, an event context, a result mapping
or a histogram", pinned by `tests/extra/frontend/m48/test_accessor_input_shapes.py`.

## The measurement

```
weight(projected) type: Array node 23
labels refuses: a plain Array carries no variations; graphed.labels takes a Varied, an …
```

## Proposed correction

Read the nominal with the value instrument, which takes a bare member:

```python
values = m57_values(base.session, graphed.weight(projected))
...
assert m57_ambient_values(base.session, graphed.weight(joined))["nominal"] == values
```

Every assertion then passes (measured, `scratchpad/p21.py`): slots `[2] == [2]`, kinds
`['factor'] == ['factor']`, `pu2` in the families, nominal unchanged.

The alternative — letting `graphed.labels` answer `("nominal",)` for a plain `Array` — relaxes an
m48 refusal outside this milestone's scope and contradicts the extra test above; it is the owner's
call, not a workaround I will take.

STOPPED on this test.
