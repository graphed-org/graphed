# m68c graphed implementer — iteration log

Freeze: freeze-m68c (9325a90). `git diff freeze-m68c -- tests/frozen/` must stay empty.

## Iteration 1 — plan-graphed.md §3/§4 as one pass, split into the §7 commits
- core: `DurablePlanV2.services` (omitted from bytes when empty, outside `task_id`), `DurablePlanV2.value`,
  `OpSpec.live` (compare=False; `resolve()` returns it); services.py `bind_services`/`require_bound`/
  `resolve_services` take a V2 plan (overloads); `evaluate_ir` dispatches `exchange`/`join` through
  `eval_stage` and takes `outputs=`/`given=` (cone cut at given nodes; `ir_cone`).
- shuffle: `_MapWrite`/`_Gather`/`_Fold`, `split`/`pick`, one `_build` for both builders; build refusals;
  `awkward.join.take` zero-length index → `block[:0]`, empty carrier `.simplified`.
  Deviation from §3.3 "move `_attribute`": frozen `debug/m50` calls `_PartitionReduce(...)._attribute`,
  so the method stays as a one-line delegate to the new module function `attribute_failures`; the
  tests/extra call site needs no change. `_Gather` evaluates from the barrier's INPUTS
  (`given={input: concat(side)}`) so the barrier dispatch itself goes through `evaluate_ir`'s
  attribution hook; same semantics as `given={barrier: eval_stage(...)}`.
  `_GatherReduce`/`partition_block` are no longer emitted by any builder; deleting them and their
  tests/extra/frontend/m39 tests trips the precommit integrity scan (`assertion_removed`), so both
  stay, docstrings corrected; removal is the owner's call.
- runners: `SequentialRunner.run(v2)` (`_run_stages`), `run_shuffle_resumable` calls `require_bound`.
- tests/extra/awkward/m68c: `pick`, partial fold refusal, a join side reading a non-partitioned source
  (each refused by a hand mutant: disabling the check → DID NOT RAISE / KeyError).
- docs: frontend design (executed example with `SequentialRunner().run(...).value`, runner list, the
  3-stage shuffle_plan sentence), awkward design (`target_bytes` → `run_repartition_by_size`),
  checkpoint design (last-stage hashes + `plan.value`), changelog 0.0.7 section.
Gates (tree before the commit split): `COV=1 ./scripts/run-tests.sh` rc=0; `coverage_gate.py` 4 files
<90%, all pre-existing ML plugin externals (jax/pytorch/tensorflow/xgboost, frameworks not installed),
every changed file 96-100%; `diff-cover --compare-branch=d0ad16b` 100% (240 lines); ruff/format/mypy
clean; sphinx -W clean; doc example executed from the rst, output matches; join/shuffle plan bytes
identical across two processes (PYTHONHASHSEED 1, 2); `git diff freeze-m68c -- tests/frozen` empty;
precommit --fast ok.

## Iteration 2 — 275c802 (builder `OpSpec.live`) reverted
- `live=` made `pickle.dumps(plan)` carry the user callable by reference (lambda: dumps fails;
  __main__: fresh-process load fails). Reverted; docstring cleanup kept. Executors broadcasts V2
  stage processes by value instead. `test_a_plan_with_an_unimportable_reduce_runs_in_a_fresh_process`
  (lambda, __main__) passes here, fails against 275c802's shuffle.py.
