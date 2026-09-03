# Test Dispute — `test_written_superset_is_the_union_and_each_universe_is_its_eager_row_set`

File: `tests/frozen/awkward/m51/test_superset_rows.py`
Anchor: `vary-m51-A` (§6.4a superset row rule against an independent reference)

## The offending assertion (line 56)
```python
assert superset_size > max(len(set(rows)) for rows in per_universe.values())
```
where (lines 43-54) `per_universe[label] == sorted(as_list(eager_universe_met(label)))` (pinned by the
line-47 assertion, which PASSES against the implemented feature) and `superset_size ==
len(set(as_list(eager_superset_met())))`. So line 56 is a claim about the FIXTURE's eager references
ALONE — `eager_superset_met` vs `eager_universe_met` — independent of any implementation.

## The contradiction (measured, from the fixture's own independent references)
The fixture's level-0 mask migrates under a per-jet JES shift that scales `pt` MONOTONICALLY:
`eager_evt_mask(label) = any(_scaled_jets(label).pt > 30.0, axis=1)`, with `jes_up = pt*1.05`,
`nominal = pt`, `jes_down = pt*0.95`. For positive `pt`, `pt*1.05 >= pt >= pt*0.95`, so the event
masks are STRICTLY NESTED:

    {jes_down} ⊆ {nominal} ⊆ {jes_up}

and the level-0 OR (§6.4a's superset) therefore EQUALS `{jes_up}`. Measured on the committed fixture
(`seed=51`, `N_EVENTS=60`), purely from `m51_write_fixtures`:

    nominal   rows 43  distinct met 43
    jes_up    rows 44  distinct met 44
    jes_down  rows 41  distinct met 41
    superset  rows 44  distinct met 44   (== jes_up)

So `superset_size (44) > max(universe distinct met) = jes_up (44)` is `44 > 44` — FALSE. No universe
can be strictly smaller than the union when one universe's mask (`jes_up`) already CONTAINS the union;
`jes_up` scales every jet up, so any event another universe selects, `jes_up` also selects. The
assertion is unsatisfiable for THIS fixture, for ANY correct implementation.

## Why the feature is correct (measured)
The three substantive assertions PASS against the implemented `to_parquet(select=)`/`read_varied`:
  * line 47: each universe's reconstructed `met_pt` == its INDEPENDENT eager row set, for all three
    universes (jes_up included);
  * line 51: the written union == the eager superset;
  * line 55: `superset_size == len(written_union)` (44 == 44).
The superset written is exactly the §6.4a OR (44 rows). Writing MORE rows than the OR would be wrong
and would break line 51. So the superset must be 44, and line 56 cannot pass.

## The spec clause it contradicts
§6.4a binds the superset to the level-0 OR of the per-label masks — nothing more. For nested masks
the OR equals the largest universe; §6.4a does not (and cannot) guarantee the OR strictly exceeds
every universe. Line 56 asserts a property §6.4a does not provide, using a fixture whose own
construction guarantees nesting.

## Proposed correction (make the masks non-nested, or drop the strict-inequality)
Either (preferred) choose per-universe shifts that MIGRATE events in BOTH directions across the
threshold so no universe's mask contains the union — e.g. vary a per-event quantity up in one
universe and down in another on DIFFERENT events (a shift that is not a global monotone scale), so
`{jes_up}` and `{jes_down}` each catch an event the other misses and the union strictly exceeds both;
or (minimal) delete line 56, since lines 47/51/55 already prove "a genuine OR" (the three universes'
row sets differ: 43/44/41, and the union equals the eager superset). The strict-inequality as written
is a stronger claim than the anchor's intent and is structurally impossible for a monotone shift.

## Impact / requested action
The §6.4a superset FEATURE is implemented correctly (verified against independent eager references,
lines 47/51/55 green). Only line 56's structural assumption (no single universe contains the union)
is violated by the fixture's monotone-scale shift. Recommend a `m51-freeze-fixup` re-freeze that
either re-designs `superset_inputs`/`eager_*` to produce non-nested masks or drops line 56. I am NOT
modifying `tests/frozen/**`.
