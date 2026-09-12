# Test Dispute — `tests/frozen/awkward/m57/test_projection.py::test_at_the_nominal_projection_a_re_derived_central_joins_while_a_weight_label_stays_a_factor`

The identity assertion and the second half (the weight-label factor) PASS. The first oracle is the
dispute.

## The test

```python
joined = m57_weight(flat, "lf", re_derived, probe)      # probe = LF_TABLE members, family "lf"
ops = [m57_factor("pu", _at(base.pu, flat), {}),
       m57_factor("hf", re_derived, probe)]             # <- keys lf's members under "hf"
```

`m57_factor(name, nominal, members)` keys the members BY FAMILY (`m57_by_label(name, members)`), so
`m57_factor("hf", re_derived, probe)` declares `probe["up"]` at `hf_up` and the bare nominal at
`lf_up`. The context is `graphed.nominal(base.ctx)`, which carries no `hf_*` labels, so the oracle is
evaluated only at `('nominal', 'lf_up', 'lf_down')` and answers the NOMINAL at `lf_up`/`lf_down` —
i.e. it demands that the join put `lf`'s members nowhere.

## The clause it contradicts

§2.1's joins-a-factor outcome: "the family's members join that factor's container as its values in
their universes … and the composition at `name_t` multiplies the member ONCE with the other
operations (the absolute-weight idiom: `hf_up = SF(up_hf)`, `lf_up = SF(up_lf)`…)". The test's own
name and docstring say the re-derived central JOINS.

## The measurement (`scratchpad/p15.py`)

```
oracle keyed 'hf': differs
oracle keyed 'lf': MATCHES
labels ('nominal', 'lf_up', 'lf_down')
second half: MATCHES
```

## Proposed correction

`m57_factor("lf", re_derived, probe)`. With that one key the oracle matches the code exactly, over
every label the ambient carries.

STOPPED on this test.
