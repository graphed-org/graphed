# Test Dispute — `tests/frozen/awkward/m57/test_placements.py::test_a_placement_at_a_point_carrying_the_overlays_coordinate_keeps_its_member`

Self-contradictory at `prefix_length=1`: the test's own ops oracle and its last assertion demand two
different values for `mu_up`.

## The test

At `prefix_length=1` the handle is read over `pu` ALONE, so the `mu` overlay is anchored before the
`hf` factor, which the test itself encodes:

```python
ops = [m57_factor("pu", pu, pu_members), mu_op, joined]   # prefix_length == 1
assert m57_ambient_values(session, weight) == m57_oracle_values(session, ops, graphed.labels(weight))
assert m57_values(session, graphed.member_of(weight, "mu_up")) == m57_values(
    session, m57_at(m57_scaled(handle, MU)["up"], "mu_up"))
```

The second assertion is the overlay's member UNMULTIPLIED, which is the final value only when no
factor follows the overlay.

## The clause it contradicts

§2.1: "The overlay is placed right after the last factor it was read over and after any overlay
already anchored there, so a factor registered after the read or after the overlay multiplies its
result (`w = weight(ctx)` over `pu, hf`, then `trig`, then `mu` on `w`:
`mu_up = 1.05 · pu · SF · trig`)". At `prefix_length=1` the `hf` factor is registered after the read,
so it multiplies the overlay's result.

## The measurement (`scratchpad/p16.py`)

```
prefix 1 points ok: True
   ops oracle: MATCHES
   last assert: differs got [3.75, 27.5, 5.0] want [1.875, 3.4375, 2.5]
   oracle mu_up: [3.75, 27.5, 5.0]
prefix 2 points ok: True
   ops oracle: MATCHES
   last assert: MATCHES
   oracle mu_up: [3.75, 27.5, 5.0]
```

The ops oracle (which the code matches at both prefix lengths) says `mu_up = 3.75`; the last
assertion demands `1.875 = 3.75 / SF`, the overlay's member before `hf` multiplies it.

## Proposed correction

Compare the overlay's own member against what the composition makes of it — multiply by the factors
registered after it, or restrict the last assertion to `prefix_length == 2`, where the overlay is the
last operation and the two readings coincide.

STOPPED on this test.
