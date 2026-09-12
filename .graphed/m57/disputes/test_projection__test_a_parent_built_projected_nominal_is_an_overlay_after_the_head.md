# Test Dispute — `tests/frozen/awkward/m57/test_projection.py::test_a_parent_built_projected_nominal_is_an_overlay_after_the_head`

The test's body (both `extra_read` legs) PASSES. Its CONTROL is the dispute.

## The control

```python
child = base.ctx[base.met.pt > MET_CUT]
below = graphed.universe(child, "hf_up")
crossed = _at(parent_member, below)          # parent_member = universe(weight(base.ctx), "hf_up")
ops = [m57_factor("head", graphed.weight(below), {}), m57_factor("mu", crossed, m57_scaled(crossed, MU))]
```

It expects `crossed` to be a NEW factor, so the oracle multiplies the head by the head
(`nominal = 3.75 * 3.75 = 14.0625`).

## The clause it contradicts

§2.3: "the adoption itself records the adopted member as the child's read, so a family whose nominal
is that node … is an overlay after the head with or without a read in between (built above a mask
that lies between, that expression is a member of a different ambient, **not the adopted node**, and
stays a new factor as today)".

The parenthetical's premise is false. Adoption at a row-space change IS the parent's composition
re-indexed across the mask, and the IR interns, so the expression built above the mask and
re-indexed down is the adopted node — one node, one context.

## The measurement (`scratchpad/p14.py`)

```
parent_member 23 ctx 4518310208
adopted       33 below id 4518311776
crossed       33 ctx 4518311776
below._reads  [((33,), (0, 1), (0, 0))]
```

`crossed.node_id == adopted.node_id == 33`, both in `below`, and `below._reads` records 33 as the
child's own read. No field distinguishes them, so §2.1's "decided BY NODE" cannot answer "new
factor" here; deciding otherwise would multiply the head into itself — the squaring m57 removes.

## Proposed correction

Make the control's central a genuine re-derivation, which is what "a member of a different ambient"
needs: `crossed = _at(parent_member, below) * 1.0`. Measured (`scratchpad/p22.py`):

```
as frozen:          crossed=33 adopted=33 -> differs got [3.75, 5.0] want [14.0625, 25.0]
re-derived (*1.0):  crossed=35 adopted=33 -> MATCHES the new-factor oracle
```

STOPPED on this test.
