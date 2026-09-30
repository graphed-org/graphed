# awkward/m68c — join and repartition plans run their IR and carry services (traceability)

Authority: lane plan `lanes/services-v2/plan-graphed.md` §3.2–§3.4 and §6, with the unit G exit
constraints (`exit-constraints.md`). Run: `python -m pytest tests/frozen/awkward/m68c -q`.

Harness `m68c_services_harness.py` (bare-module name in the pyproject mypy override): `Chunks`, an
in-memory `PartitionedSource` (no pyarrow) that records every `read_partition` in `READS`; `events`
(one file, every run hashes to dest 0 of 2, first half `x < 25`) and `lumi` (two files); `SFServer`/
`serve()`, a local HTTP stand-in counting `requests` (binds without `socket.getfqdn`); `SF_PLUGIN`, an External whose evaluation fetches
the scale factor from the bound endpoint on every call (`scaled`, `corrected`, `by_hand`); `FOLD`,
a reduce/combine/empty that keeps each gathered block in dest order; `ResolvingReduce`/`RESOLVED`.
Spies are module state because runners resolve cloudpickle copies of stage processes.

| test | plan clause | pins |
|---|---|---|
| `test_m68c_builders_services.py::test_join_plan_and_shuffle_plan_carry_the_named_services_deterministically` | §3.3 builders, §6 *Builders carry services*, *Determinism* | referenced specs only (not every declared one), `services=` adds by name order, byte-identical rebuild, services survive `from_bytes` |
| `test_m68c_evaluate_ir_full.py::test_evaluate_ir_over_the_full_ir_equals_materialize[*]` | §3.3 `evaluate_ir` kinds | outer join + post op, grouped left join, keyed repartition + post op equal `materialize` in `.type` and rows |
| `test_m68c_per_dest_join_type.py::test_per_dest_joins_concatenate_to_the_whole_join[*]` | §3.3 `take` fix, §6 *Per-dest join type* | some dest has a zero-row left and some a zero-row right side (run 7 left only); union `.type` and rows equal the whole join, flat × 4 `how`, grouped × {inner, left} |
| `test_m68c_runs_no_service.py::test_join_plan_with_reduce_folds_the_post_join_value_per_dest[*]` | §3.3/§3.4, §6 *Runs with no service* | stage kinds, recorded `broadcast=True`, each map task reads its partition once, counts (6, 3), dest 1 holds right-side rows only (hash routing both sides), post-join op runs; `.type` and rows equal `materialize` |
| `…::test_join_plan_without_reduce_returns_the_per_dest_blocks[*]` | §3.3 no-reduce value | value is a tuple of the per-dest blocks whose union is `materialize`'s |
| `…::test_join_plan_keeps_the_later_registered_source_on_the_left` | §3.3 builders (stage i reads join input i) | `join(lumi, events, how="left")`, lumi registered second: `.type` and rows equal `materialize` |
| `…::test_shuffle_plan_by_key_routes_each_key_to_one_dest_and_runs_the_post_op` | §3.3 hash route | per-dest key sets equal `backend.partition(…, "run", 2)`'s; post-repartition op runs |
| `…::test_shuffle_plan_by_count_sends_each_map_block_whole_to_one_dest` | §3.3 count route | `n=3` > 2 partitions, a cut empties partition 0: per-dest lengths `[0, 4, 0]`, every block keeps the type |
| `test_m68c_grouped_join.py::test_grouped_join_plan_matches_materialize[*]` | §3.3 grouped, §6 *Grouped join* | inner/left × with/without reduce, dest 1 zero-row left: `var * {…}` type and sublist multiset equal `materialize` |
| `test_m68c_service_live.py::test_a_bound_server_call_runs_in_a_v2_plan[*]` | §3.2/§3.3, §6 *Server call live* | before repartition, before a join (left), after a join: bound value equals the by-hand value; `requests > 0` |
| `test_m68c_unbound_refusal.py::test_an_unbound_v2_plan_is_refused_before_any_process_call[*]` | §3.2/§3.4, §6 *Unbound refusal* | join and shuffle plans × SequentialRunner and `run_shuffle_resumable`: `UnboundService` naming `sf`, zero reads, nothing journaled |
| `test_m68c_resumable_live.py::test_resumable_bound_join_plan_equals_the_sequential_value[*]` | §3.4, §6 *Resumable live* | `plan.value` over every last-stage hash equals the SequentialRunner value, with and without reduce; every task executed; `requests > 0` |
| `test_m68c_resolve_services.py::test_resolve_services_reaches_the_gathers_reduce` | §3.2/§3.3, §6 *resolve_services* | the gather's Resolvable `reduce` sees the value exactly once |
| `test_m68c_build_refusals.py::test_target_bytes_repartition_is_refused_by_shuffle_plan` | §3.3 refusals | TypeError naming `target_bytes`; `n=` builds |
| `…::test_a_post_join_operation_reading_a_source_directly_is_refused` | §3.3 refusals | TypeError; the same op over the joined array builds |
| `…::test_a_chunk_partial_is_refused[*]` | §3.3 chunk partials | `x / gak.sum(x)` before and after the barrier in both builders, and an unfolded `gak.sum` join output: GraphedError naming `'ak.sum'`; a folded `gak.sum` output builds in both |
| `…::test_grouped_right_and_outer_joins_are_refused[*]` | §3.3 refusals | TypeError naming `grouped`; grouped left builds |
| `test_m68c_attribution.py::test_a_raising_external_before_a_join_points_at_its_recording_line` | §3.3 attribution (M6) | `StageError` whose user frame is this file's `record_external` line |
