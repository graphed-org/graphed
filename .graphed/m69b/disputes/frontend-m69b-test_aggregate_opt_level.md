# Test Dispute — `test_aggregate_opt_level.py` (two merged-output legs + README)

File: `tests/frozen/frontend/m69b/test_aggregate_opt_level.py`, `tests/frozen/frontend/m69b/README.md` (graphed#228)

Status: CLOSED (owner ruling below)

## The pins
- `test_opt0_runs_and_reduce_receives_one_value_per_marked_output`: at `opt_level=1` the merged pair
  `events.w, events.w * 1.0` gives `[total]`.
- `test_opt0_writes_the_defaults_parts`: `arities == {1: [2, 2], 0: [3, 3]}`.

## The contradiction
#228: `reduce` handed fewer values than outputs cannot tell which positions merged, so the natural
`dict(zip(names, values))` pairs names with the wrong values, silently. The ruled contract is one
value per output passed, in order, at every opt level; both pins assert the merged count.

## Correction
`[total]` → `[total, total]`; `{1: [2, 2], 0: [3, 3]}` → `{1: [3, 3], 0: [3, 3]}`; the README rows
and its replay sentence say the same. No other assertion changes.

## Ruling (owner, 2026-11-20)
"#228 option A with refreeze authorized". Amendment under `--allow-refreeze tests/frozen/frontend/m69b`,
tagged `freeze-m69b-fixup4` on the merge commit.
