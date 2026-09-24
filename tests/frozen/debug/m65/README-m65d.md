# M65 frozen suite, unit D, graphed-debug slice (`freeze-m65d`, plan-D.md)

Frozen and read-only after `freeze-m65d`. A new file beside the earlier units' READMEs, which it does not
edit (plan.md decision 4). New API is read as module attributes (`gd.replay`) or keywords
(`aggregate_plan(store=)`), so before D1/D3 each test fails on its own line. D writes no `checkpoint/m65`
tests: capture is pinned here through `graphed.checkpoint.Store` / `FsspecStore` reads.

| Test (`test_m65d_replay.py`) | Contract | Fails |
|---|---|---|
| `test_capture_writes_inputs_and_outputs` (1) | D-2 capture, the path arm of the open rule, input kept for a failing task, no store opened without `store=` | a path root opened as an fsspec layout; input put after evaluating; a store touched on the default path |
| `test_capture_through_a_url_from_a_spawned_process` (2) | D-2 URL arm from spawned children; driver reads it back | a capture that opens every root as a `Store` path; a replay that does not find the children's journals |
| `test_steps_walk_the_opt_level_0_cone_with_user_frames` (3) | D-3 cone (no unrelated program), node order, D2 `describe` fallback, per-step values | placeholder step values; steps over the whole arena or the fused IR; `lower` crashing on `axis=None` |
| `test_steps_are_lazy_and_timed` (4) | D-4 lazy generator, per-node timing, `.value` cached | eager evaluation before the first yield; `seconds` hard-coded or attributed to the next node; `.value` recomputed |
| `test_value_and_diff_against_the_recorded_output` (5) | D-6 `recorded` vs `re-evaluated` reference | a wrong `input_source` or `reference`; a diff whose sides are not the task's partial |
| `test_inputs_come_from_the_store` (6) | D-2 input read from the store; re-read otherwise | a replay that re-reads a deleted file although the input was captured; a re-read that does not surface the missing file |
| `test_failing_task_raises_at_its_node` (7) | D-5 `StageError` at the failing node, M6 frames/forms, `diff()` raises; unreached key re-read | an error at the source or output node; frames other than M6 `_stage_error`'s; steps that yield past the failure |
| `test_diff_reports_a_difference` (8) | D-6 by-value diff against the recorded output | a diff against a fresh evaluation (both 249 rows); an elementwise `==` compare (`ValueError`) |
| `test_replay_refusals` (9) | API refusals | a replay that accepts a non-`aggregate_plan` plan, an unknown key or other outputs |
| `test_replay_binds_the_runs_external_evaluators` (10) | D-4 Externals from `process.externals` | a replay that binds the session's evaluators |
| `test_replay_reads_its_own_plans_capture_root` (11) | API: `replay` reads the plan's own capture root (path and URL); one run per root (review D r3 L1) | a replay that reads another plan's root for the same capture id |

Constraints folded from review D r1–r3:
- Test 4's map sleeps 0.06 s against a 0.05 s floor; its counter baseline is taken after the run; the map
  step is the one whose `node.kind == "external"`.
- Test 6 accepts a raw `FileNotFoundError` (review D r3 L4's resolution) or a `StageError` whose `__cause__`
  is one; it pins that the missing file surfaces.
- Test 1 reads a path capture through `graphed.checkpoint.Store` (review D r2/r3 L2).
- Test 11 keeps one run per root, as review D r3 L1 closes it; two plans sharing a root are not pinned.
- Reduce/combine/map functions are module-level so spawned children unpickle them.

Mutation check against a scratch prototype of plan-D (not committed): the prototype passes all 11; each
mutant fails at least its row's test — a replay reading the most recently captured root (11), every root
as `FsspecStore` (1), diff against a fresh evaluation (2, 5, 8, 11), session externals (10), eager
steps (4, 7), zero timings (4), placeholder step values (3), input put after evaluating (1, 7), and
an elementwise compare (8).
