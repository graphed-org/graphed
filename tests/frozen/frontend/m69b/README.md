# frontend/m69b — `aggregate_plan(opt_level=)` (traceability)

Spec: the m69b services plan's graphed PR, "`aggregate_plan(opt_level=)`", with its review exit items.
Awkward-free (numpy backend, the in-memory record source `Events`), so the free-threaded job can collect it.

**New API (a `TypeError`/`AttributeError` on d0ad16b is the expected pre-implementation failure):**
`graphed.aggregate_plan(..., opt_level=1)`: `1` is the optimized compile, `0` the 1:1 cone of the plan's
outputs, write arrays and metadata arrays, any other value refused (`ValueError`) naming `0` and `1`;
`GraphStore.cone(outputs=)`; `graphed.debug.replay` compiles once at the plan's level and hands `reduce` one
value per IR output.

Fixture (`m69b_opt_fixtures.record`): before any output, the session records, unmarked, a node over a column
no output reads (`z * 3.0`) and a node that raises on the data (`x[x > 100][0]`). The marked pair `w`,
`w * 1.0` is one the optimizer merges.

| Test | Clause | Witness | Fails on |
|---|---|---|---|
| `test_opt0_runs_and_reduce_receives_one_value_per_marked_output` | at `0` the plan runs, the reduce receives one value per marked output, the totals equal the default's, which receives one fewer | `sums` returns one total per value: `[5, 5]` at `0`, `[5]` by default | shipping the whole store (`compile_ir(optimize=False)`: the unread-column node raises); a default of `0` |
| `test_opt0_ships_the_lowered_cone_of_the_outputs` | the shipped IR's node count equals the outputs' `lower(opt_level=0)` cone; `compile_ir(session, *outputs, optimize=False)` holds more | node counts | shipping the whole store |
| `test_opt0_writes_the_defaults_parts` | `writes=` with a metadata array at `0` writes the default's parts | part files byte-equal; reduce arity `3` at `0` (two values, one path), `2` by default | a cone over `outputs` only (write and metadata arrays unslotted); the whole store; a default of `0` |
| `test_the_default_is_the_optimized_compile` | the default's IR bytes equal a plan built without the kwarg, and today's optimized compile | IR bytes | a default of `0` |
| `test_opt0_ir_and_pickled_plan_do_not_depend_on_the_hash_seed` | at `0` the IR and the pickled plan are byte-identical across two child interpreters with `PYTHONHASHSEED` 1 and 2 | each child (`check=True`) prints one line: IR digest, pickled-plan digest, `hash("graphed")`; the hashes differ, so the comparison is live | a hash-ordered structure (a `frozenset` of strings) in the shipped process |
| `test_replay_of_a_capturing_plan_equals_its_run[w-and-w-times-1-at-0]` | a `store=` plan at `0`, run, then `graphed.debug.replay(plan, 0, *outputs).diff()` is equal | `diff.reference == "recorded"`, `diff.equal` | replay recompiling optimized (refuses the plan); the whole store |
| `test_replay_of_a_capturing_plan_equals_its_run[w-twice-at-0]` | the same with the outputs `(w, w)` at `0` | as above | replay reducing one value per argument; replay recompiling optimized |
| `test_replay_of_a_capturing_plan_equals_its_run[w-and-w-times-1-by-default]` | the same with `(w, w * 1.0)` at the default | as above | replay reducing one value per argument (d0ad16b) |
| `test_a_failing_op_reports_the_compiled_level_and_its_own_line` | an op failing on the data, recorded after the unmarked nodes, raises a `StageError` whose `opt_level` is `1` by default and `0` at `0`, and whose `user_frame` names that op's line at both | `StageError.opt_level`, `user_frame` file and line | the literal `opt_level=1`; frames keyed by record id rather than shipped id (names an unmarked node's line); the whole store; a default of `0` |
| `test_cone_refuses_an_id_past_the_store` | `GraphStore.cone(outputs=)` refuses an out-of-range id with `ValueError("no node with id <n>")`; an in-range cone succeeds | the message names the first id past the store, listed after a valid one | an unvalidated DCE (pyo3 `PanicException`); validating only the first id; a cone refusing every id |
| `test_other_levels_are_refused_naming_0_and_1[2]`, `[-1]` | `opt_level=2` and `-1` refused naming `0` and `1` | `ValueError` whose message holds the numbers `0` and `1` | no validation (any other value compiled as the cone) |

Run: `python -m pytest tests/frozen/frontend/m69b -q`.
