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
