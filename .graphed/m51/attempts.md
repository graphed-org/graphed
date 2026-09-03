# m51 graphed implementer — iteration log

Freeze: m51-freeze (ab5c71e). Scope C1-C4 + D1. `git diff m51-freeze -- tests/frozen/` MUST stay empty.

## Iteration 0 — baseline
awkward/m51: 35 failed / 2 passed. numpy/m51: 1 failed / 2 passed. (36 red + 4 green.)
Design validated by probes + full round-trip prototype (see m51-graphed-impl-worklog.md).

## Iteration 1 — C1 (06141f0)
graphed.selection bridge + numpy refusal. numpy/m51 3/3 green; awkward selection root-None green.
Filed dispute: test_selection_on_a_universe_nominal...refuses_downstream (as_list on deferred arrays;
needs session.materialize per m48-m50 idiom). Feature correct.

## Iteration 2 — C2 (write path: select= API + record-time + superset + augmentation + node-id unpack + exec-time)
All 17 record-time/exec-time/disposition/single-read tests green. Remaining awkward reds all need
read_varied (C4) + the disputed test. mypy+ruff clean.

## Iteration 3 — C4 (parquet-KV manifest + read_varied)
`_MANIFEST_KEY=b"graphed.variations"`, `_build_manifest` (driver-derived, mirrors the columns
`__call__` writes; `levels` in the bound order), `_write_augmented` merges the manifest through the
PUBLIC arrow route (`pq.read_table` → `Table.replace_schema_metadata` → `pq.write_table`, no
`awkward._connect`), and `read_varied(path)` reconstructs each universe (label order from the on-disk
vary-column order, nominal forced first; XOR value inverse; level-0 row mask + level->=1 object masks
decoded at superset granularity then restricted to kept rows). Added tests/extra guard
`test_no_private_awkward_kv`. mypy+ruff clean.

Frozen suite now 38 green / 2 red — the 2 reds are the two FILED DISPUTES, each provable a
fixture-level defect from the fixture's own independent eager references (NOT an implementation bug):
  * test_selection_on_a_universe_nominal...refuses_downstream — `ak.to_list` on deferred arrays;
  * test_written_superset_is_the_union... — line 56 `superset_size > max(universe)` is `44 > 44`:
    the JES-shift masks are strictly nested (jes_up ⊇ nominal ⊇ jes_down), so the level-0 OR EQUALS
    jes_up; lines 47/51/55 (the substantive superset assertions) all PASS.
`git diff m51-freeze -- tests/frozen/` empty.
