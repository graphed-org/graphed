# Test Dispute — `tests/frozen/awkward/m57/test_records.py::test_ambient_entries_carry_the_families_kind_and_links_of_the_parents`

## The test

Its `projected=True` leg registers `lf` at `child = graphed.universe(base.ctx, "hf_up")` with the
central `graphed.reindex_to(base.sf, child)` — `base.sf` being the central of the `hf` factor, the
factor that OWNS `hf_up`.

## The clause it contradicts

§2.3: "except the factor that OWNS `L` (a family of its rider is a coordinate of `L`'s point …):
its member at `L` is the universe projected into, not the central, and naming it there is refused".

Another frozen test of this same suite demands that refusal on the SAME registration shape —
`tests/frozen/awkward/m57/test_projection.py::test_naming_the_factor_that_owns_the_label_is_refused`
registers `m57_weight(target, "mf", owner, members)` with `owner = _at(base.sf, projected)` at
`projected = graphed.universe(source, "hf_up")` inside `pytest.raises(GraphedError)`. The two
registrations differ only in the family NAME, which no clause makes a discriminator.

## The measurement

```
graphed.errors.GraphedError: graphed.vary('lf'): its central names the weight factor that the
universe 'hf_up' this context is projected into is OF, whose member there is that universe rather
than the central; register this family on the factor at the parent, before the projection, or read a
graphed.weight() handle AT this context (`w = graphed.weight(ctx)`) and register this family on that
```

## Proposed correction

Name a factor that does NOT own the projected universe, so the leg exercises the join at a row-space
change it is about. With `_at(base.pu, child)` as the central, every assertion of the test passes in
both legs (measured, `scratchpad/p18.py`):

```
projected False families ['hf', 'lf', 'pu'] kinds ['factor'] link ok True
projected True  families ['hf', 'lf', 'pu'] kinds ['factor'] link ok True
```

STOPPED on this test: the code is right and no change can satisfy both it and the refusal test.
