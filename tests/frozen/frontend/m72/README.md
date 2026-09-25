# frontend/m72 — multi-output plans: part writes beside reductions, and collate (traceability)

Spec: the m72 brief (a)/(b) and `graphed-workdir/lanes/multiout/plan-A.md` §2 "Frozen-test contract".
Awkward-free (numpy backend subclass `CountingBackend`, toy picklable `json_codec`), so the
free-threaded job can collect it.

**New API (an `AttributeError`/`TypeError` on 0.0.6 is the expected pre-implementation failure):**
`graphed.write.PartWrite(array, destination, name, codec, metadata=None)`;
`graphed.aggregate_plan(..., writes=())`; `graphed.collate` (= `graphed.aggregate.collate`);
`graphed.refuse_chunk_partials(..., as_outputs=<collection of compiled ids>)`.

| Test | Contract item | Pins (witness) |
|---|---|---|
| `test_write_and_reduce_share_one_read_and_one_evaluation` | frontend 1 | reads == tasks (`FileSource.reads`), `EVALS["mul"]` == tasks, value and part contents right |
| `test_reduce_sees_todays_values_then_one_path_per_write` | frontend 2 | outputs `(a, b, a)` + writes `(a, b*1.0, 3x)` → `reduce` gets `[va, vb, p1, p2, p3]`; `b*1.0` (optimizer-merged) and `a` (an output) write the right values; `writes=()` gives `[va, vb]` |
| `test_part_metadata_holds_this_chunks_reduction` | frontend 3 | per-part KV = `str()` of that chunk's float32 sum (differs from `str(float())`); two chunks differ; a static value is `str()`-ed once at build; `metadata=None` → codec gets `None` |
| `test_part_path_is_destination_joined_with_name_of_the_tasks_partition` | frontend 4 | explicit ranges and blind tasks: path = `join(destination, name(task.partition))` (the unresolved blind partition), parent dirs created |
| `test_colliding_part_names_are_refused_before_any_task_runs` | frontend 5 | blind `0-0` naming and two writes to one name raise `ValueError` "write the same part" at build: zero reads, no codec call, no directory; explicit ranges build |
| `test_write_root_reduction_refused_metadata_reduction_allowed` | frontend 6 | a reduction write root raises `GraphedError` "a partitioned write has no combine step"; the same reduction as metadata and as an output builds and runs |
| `test_refusals` | frontend 7 | cross-session write array / metadata array: `TypeError` "one session"; `store=` + writes: `TypeError` "does not capture writes"; nothing: `ValueError` "at least one output"; `refuse_chunk_partials(as_outputs={id})` refuses only that id; `Varied` write array, bare and nested `Varied` metadata: `GraphedError` "does not accept a Varied" |
| `test_collate_value_equals_each_plan_run_alone` | frontend 8 | two different graphs: collated `SequentialRunner` value == `{name: run(sub).value}` == unseeded key-order tree fold of partials; zero-task sub-plan absent on both sides; all-zero collate → `{}` |
| `test_collate_routes_each_task_to_its_own_graph` | frontend 9 | each source reads only its own `(uri, tree)` (`mc-a` exists under two trees); tasks re-keyed `0..N-1` in mapping order then key order (hand-built plan with out-of-order keys) |
| `test_collate_ships_no_per_task_data` | frontend 10 | `len(pickle.dumps(process))` equal at 1 and 16 steps/file; tasks are plain `Task`s of the sub-plans' own partitions, byte-identical pickles |
| `test_collate_refusals` | frontend 11 | `{}`: "at least one plan"; `next_tasks` / `stop`: `TypeError` "fixed tasks"; shared `(uri, tree)`: `ValueError` "appears in plans" naming both; same uri other tree allowed; `open_once` is the OR |
| `test_collated_plan_with_writes_runs_across_processes` | frontend 12 | collated plan with writes pickles; spawn `ProcessPoolExecutor` partials folded by key == `SequentialRunner` value; parts byte-identical to the sequential run's |

Run: `python -m pytest tests/frozen/frontend/m72 -q`.
