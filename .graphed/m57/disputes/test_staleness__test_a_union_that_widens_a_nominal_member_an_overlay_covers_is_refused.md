# Test Dispute — `tests/frozen/awkward/m57/test_staleness.py::test_a_union_that_widens_a_nominal_member_an_overlay_covers_is_refused`

The refusal, its message and its family names all PASS. The mint assertion is the dispute.

## The test

```python
before = m57_node_count(session)
with pytest.raises(GraphedError) as caught:
    m57_weight(with_overlay, "lf", widening, m57_table_members(sjets, LF_TABLE))
...
assert m57_node_count(session) == before
```

Python evaluates `m57_table_members(sjets, LF_TABLE)` — the test's own argument expression — before
the call it guards, inside the `raises` block. Those nodes are the fixture's, not the refusal's.

## The clause it contradicts

§2.3: "Nothing is composed or re-indexed to DECIDE: a central re-indexed for the comparison or an
ambient composed for it would mint nodes on programs that name nothing". That binds the REFUSAL, and
the refusal honours it.

## The measurement (`scratchpad/p17.py`)

```
refused: graphed.vary('lf'): its central names the weight factor registered here, but joining would
         add the universes ['jes_down' …
before=56 after building the members argument=74 after the refused call=74
the argument mints 18; the refusal itself mints 0
```

## Proposed correction

Hoist the argument above the baseline, so the instrument measures the call:

```python
members = m57_table_members(sjets, LF_TABLE)
before = m57_node_count(session)
with pytest.raises(GraphedError) as caught:
    m57_weight(with_overlay, "lf", widening, members)
```

STOPPED on this test.
