# Test Dispute — `tests/frozen/awkward/m57/test_explain.py::test_explain_reports_the_links_the_lineage_took`

## The test

Its `projected=True` leg registers `lf` at `child = graphed.universe(base.ctx, "hf_up")` with the
central `graphed.reindex_to(base.sf, child)` — the central of the `hf` factor, which OWNS `hf_up`.
Same registration shape as
`test_records.py::test_ambient_entries_carry_the_families_kind_and_links_of_the_parents`.

## The clause it contradicts

§2.3: "except the factor that OWNS `L` …: its member at `L` is the universe projected into, not the
central, and naming it there is refused". `test_projection.py::test_naming_the_factor_that_owns_the_label_is_refused`
demands that refusal for the same central at the same projection (family `mf` instead of `lf`).

## The measurement

```
graphed.errors.GraphedError: graphed.vary('lf'): its central names the weight factor that the
universe 'hf_up' this context is projected into is OF, whose member there is that universe rather
than the central; …
```

## Proposed correction

Register on the factor that does not own the universe — `graphed.reindex_to(base.pu, child)` with
`base.hf_members` reindexed as the members. Every assertion then passes in both legs (measured,
`scratchpad/p18.py`):

```
projected False  explain: lf links (('mask', None),)        | pu links () | ops True
projected True   explain: lf links (('project', 'hf_up'),)  | pu links () | ops True
```

STOPPED on this test.
