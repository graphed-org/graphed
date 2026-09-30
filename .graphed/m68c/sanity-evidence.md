# m68c TEST_SANITY evidence (graphed, test author)

Base: graphed-org/graphed main d0ad16b (branch m68c). Venv `/Users/lgray/vibe-coding/cloud/.venv-m68c`.
Regenerate the table: run each dir with `--junitxml` and `python .graphed/m68c/probes/reasons.py <xml>...`:
`python -m pytest tests/frozen/core/m68c --junitxml=core.xml`, same for `tests/frozen/awkward/m68c`.

## First exception per test on d0ad16b (every test fails; 50 of 50)
Each matches plan-graphed.md §6's stated reason. The two core tests that construct
`DurablePlanV2(services=…)` fail at that keyword (§6 *Bytes round-trip*; the unbound core leg).

```
  sequential_v2::test_a_completed_run_returns_the_last_stages_value_in_stage_order -> AttributeError: 'DurablePlanV2' object has no attribute 'services'
  sequential_v2::test_an_unbound_plan_is_refused_before_any_process_call_and_a_bound_one_runs -> TypeError: DurablePlanV2.__init__() got an unexpected keyword argument 'services'
  sequential_v2::test_a_control_cancelled_before_run_stops_with_no_process_call -> AttributeError: 'DurablePlanV2' object has no attribute 'services'
  sequential_v2::test_a_cancel_inside_the_first_stage_stops_before_the_next_stage -> AttributeError: 'DurablePlanV2' object has no attribute 'services'
  v2_plan_services::test_services_round_trip_and_stay_out_of_the_bytes_when_empty_and_out_of_task_ids -> TypeError: DurablePlanV2.__init__() got an unexpected keyword argument 'services'
  v2_plan_services::test_binding_keeps_bytes_and_task_ids_and_reaches_the_bindable_stage_only -> AttributeError: 'DurablePlanV2' object has no attribute 'process'
  attribution::test_a_raising_external_before_a_join_points_at_its_recording_line -> AttributeError: 'DurablePlanV2' object has no attribute 'services'
  build_refusals::test_target_bytes_repartition_is_refused_by_shuffle_plan -> Failed: DID NOT RAISE TypeError
  build_refusals::test_a_post_join_operation_reading_a_source_directly_is_refused -> Failed: DID NOT RAISE TypeError
  build_refusals::test_a_chunk_partial_is_refused[_join_sum_before] -> Failed: DID NOT RAISE GraphedError
  build_refusals::test_a_chunk_partial_is_refused[_join_sum_after] -> Failed: DID NOT RAISE GraphedError
  build_refusals::test_a_chunk_partial_is_refused[_join_sum_unfolded] -> Failed: DID NOT RAISE GraphedError
  build_refusals::test_a_chunk_partial_is_refused[_shuffle_sum_before] -> Failed: DID NOT RAISE GraphedError
  build_refusals::test_a_chunk_partial_is_refused[_shuffle_sum_after] -> Failed: DID NOT RAISE GraphedError
  build_refusals::test_grouped_right_and_outer_joins_are_refused[right] -> Failed: DID NOT RAISE TypeError
  build_refusals::test_grouped_right_and_outer_joins_are_refused[outer] -> Failed: DID NOT RAISE TypeError
  builders_services::test_join_plan_and_shuffle_plan_carry_the_named_services_deterministically -> AttributeError: 'DurablePlanV2' object has no attribute 'services'
  evaluate_ir_full::test_evaluate_ir_over_the_full_ir_equals_materialize[_join_then_op] -> graphed.errors.GraphedError: evaluate_ir: unknown node kind 'exchange'
  evaluate_ir_full::test_evaluate_ir_over_the_full_ir_equals_materialize[_grouped_left] -> graphed.errors.GraphedError: evaluate_ir: unknown node kind 'exchange'
  evaluate_ir_full::test_evaluate_ir_over_the_full_ir_equals_materialize[_repartition_then_op] -> graphed.errors.GraphedError: evaluate_ir: unknown node kind 'exchange'
  grouped_join::test_grouped_join_plan_matches_materialize[inner-True] -> TypeError: join_plan() got an unexpected keyword argument 'reduce'
  grouped_join::test_grouped_join_plan_matches_materialize[inner-False] -> AttributeError: 'DurablePlanV2' object has no attribute 'services'
  grouped_join::test_grouped_join_plan_matches_materialize[left-True] -> TypeError: join_plan() got an unexpected keyword argument 'reduce'
  grouped_join::test_grouped_join_plan_matches_materialize[left-False] -> AttributeError: 'DurablePlanV2' object has no attribute 'services'
  per_dest_join_type::test_per_dest_joins_concatenate_to_the_whole_join[False-inner] -> TypeError: IndexedOptionArray cannot contain a union-type (unless categorical), option-type, or indexed 'content' (IndexedArray); try IndexedOptionArr
  per_dest_join_type::test_per_dest_joins_concatenate_to_the_whole_join[False-left] -> TypeError: IndexedOptionArray cannot contain a union-type (unless categorical), option-type, or indexed 'content' (IndexedArray); try IndexedOptionArr
  per_dest_join_type::test_per_dest_joins_concatenate_to_the_whole_join[False-right] -> TypeError: IndexedOptionArray cannot contain a union-type (unless categorical), option-type, or indexed 'content' (IndexedArray); try IndexedOptionArr
  per_dest_join_type::test_per_dest_joins_concatenate_to_the_whole_join[False-outer] -> TypeError: IndexedOptionArray cannot contain a union-type (unless categorical), option-type, or indexed 'content' (IndexedArray); try IndexedOptionArr
  per_dest_join_type::test_per_dest_joins_concatenate_to_the_whole_join[True-inner] -> TypeError: IndexedOptionArray cannot contain a union-type (unless categorical), option-type, or indexed 'content' (IndexedArray); try IndexedOptionArr
  per_dest_join_type::test_per_dest_joins_concatenate_to_the_whole_join[True-left] -> TypeError: IndexedOptionArray cannot contain a union-type (unless categorical), option-type, or indexed 'content' (IndexedArray); try IndexedOptionArr
  resolve_services::test_resolve_services_reaches_the_gathers_reduce -> AttributeError: 'DurablePlanV2' object has no attribute 'process'
  resumable_live::test_resumable_bound_join_plan_equals_the_sequential_value[True] -> TypeError: join_plan() got an unexpected keyword argument 'reduce'
  resumable_live::test_resumable_bound_join_plan_equals_the_sequential_value[False] -> AttributeError: 'DurablePlanV2' object has no attribute 'process'
  runs_no_service::test_join_plan_with_reduce_folds_the_post_join_value_per_dest[inner] -> TypeError: join_plan() got an unexpected keyword argument 'reduce'
  runs_no_service::test_join_plan_with_reduce_folds_the_post_join_value_per_dest[left] -> TypeError: join_plan() got an unexpected keyword argument 'reduce'
  runs_no_service::test_join_plan_with_reduce_folds_the_post_join_value_per_dest[right] -> TypeError: join_plan() got an unexpected keyword argument 'reduce'
  runs_no_service::test_join_plan_with_reduce_folds_the_post_join_value_per_dest[outer] -> TypeError: join_plan() got an unexpected keyword argument 'reduce'
  runs_no_service::test_join_plan_without_reduce_returns_the_per_dest_blocks[inner] -> AttributeError: 'DurablePlanV2' object has no attribute 'services'
  runs_no_service::test_join_plan_without_reduce_returns_the_per_dest_blocks[left] -> AttributeError: 'DurablePlanV2' object has no attribute 'services'
  runs_no_service::test_join_plan_without_reduce_returns_the_per_dest_blocks[right] -> AttributeError: 'DurablePlanV2' object has no attribute 'services'
  runs_no_service::test_join_plan_without_reduce_returns_the_per_dest_blocks[outer] -> AttributeError: 'DurablePlanV2' object has no attribute 'services'
  runs_no_service::test_shuffle_plan_by_key_routes_each_key_to_one_dest_and_runs_the_post_op -> AttributeError: 'DurablePlanV2' object has no attribute 'services'
  runs_no_service::test_shuffle_plan_by_count_sends_each_map_block_whole_to_one_dest -> AttributeError: 'DurablePlanV2' object has no attribute 'services'
  service_live::test_a_bound_server_call_runs_in_a_v2_plan[_before_repartition] -> AttributeError: 'DurablePlanV2' object has no attribute 'process'
  service_live::test_a_bound_server_call_runs_in_a_v2_plan[_before_join] -> TypeError: join_plan() got an unexpected keyword argument 'reduce'
  service_live::test_a_bound_server_call_runs_in_a_v2_plan[_after_join] -> AttributeError: 'DurablePlanV2' object has no attribute 'process'
  unbound_refusal::test_an_unbound_v2_plan_is_refused_before_any_process_call[_join-_sequential] -> AttributeError: 'DurablePlanV2' object has no attribute 'services'
  unbound_refusal::test_an_unbound_v2_plan_is_refused_before_any_process_call[_join-_resumable] -> TypeError: partition_block() takes 2 positional arguments but 3 were given
  unbound_refusal::test_an_unbound_v2_plan_is_refused_before_any_process_call[_shuffle-_sequential] -> AttributeError: 'DurablePlanV2' object has no attribute 'services'
  unbound_refusal::test_an_unbound_v2_plan_is_refused_before_any_process_call[_shuffle-_resumable] -> TypeError: keep() takes 1 positional argument but 3 were given
```

## Determinism
Two consecutive runs of both dirs give identical per-test outcomes and messages (junit diff empty).
`python -m pytest tests/frozen/core` as one process: the m1 `awkward not in sys.modules` guard still
passes (only the 6 m68c tests fail), so core/m68c is awkward-free.

## Lint / types
`ruff check` clean on both dirs. `mypy` (repo config) clean. Under a strict config with the
`tests.frozen.*` code override removed, the only remaining errors are the not-yet-existing API
(`services=`/`reduce=` keywords, `DurablePlanV2` passed where `Plan` is typed, `.value`,
`.services`) and one comparison-overlap that follows from `resolve_services`' V1 typing.

## Pass-ability and discrimination (measured where the fix is local)
- Per-dest join type under §3.3's `take` fix (`probes/takefix_plugin.py`, `-p takefix_plugin`): 6/6
  pass. Under the r1 `.simplified`-only mutant (`probes/simplified_plugin.py`): 5/6 fail (outer passes,
  as g3 predicts). On main: 6/6 TypeError.
- `probes/gather_sim.py` runs §3.3's map/gather by hand over the harness data (read per partition,
  `pack_key`, `partition` into 2, wire round trip, per-dest `concat` + `eval_stage("join")`, post op,
  pickle) with the `take` fix: `.type` and rows equal `materialize` for flat × 4 `how` and grouped
  inner/left; per-dest lengths show dest 1 empty for inner/left and right-only for right/outer.
- Attribution premise: the compiled correspondence carries the External key `(n, None)` with the
  test's `record_external` line (probed via `compile_ir(...).correspondence.frames`).
- Mutants each test refuses (by construction; measured against a stand-in in `sanity-G tests-1.md`):
  - broadcast honoured in V2: dest-1 right-only check (refused by the left and outer cases only).
  - post-barrier ops dropped: post-op rows differ from the bare join, and the test asserts that.
  - whole-dataset read instead of per partition: `READS` counts; a whole-file read per task keeps
    the uri, so `READS` passes and the row count in `.type` refuses it.
  - count route that splits a block across dests, or a zero-row filler of a different type:
    `[0, 4, 0]` plus the per-block type check. (`task.key` without `% parts` is not refused: with
    2 map tasks and 3 dests the two agree.)
  - `services` = every declared spec: the no-service join must give `()`.
  - `plan.value` returning a list, or folding a non-reduce last stage: `(3, 6)` tuple.
  - control checked only on entry, or not reset: the in-stage cancel test and `state is RUNNING`.
  - `require_bound` after the first task: `READS == []`, no journal entry.
