# M65 frozen suite, unit C, graphed-debug slice (`freeze-m65c`, plan-C.md)

Frozen and read-only after `freeze-m65c`. A new file beside the A1 `README.md` and B's `README-m65b.md`,
which it does not edit (plan.md decision 4). New API is read as module attributes (`gd.RunRecorder`,
`gc.complete_events`, `gp.capture_environment`), so before C1 each test fails on its own line.

| Test (`test_m65c_run_report.py`) | Contract | Fails |
|---|---|---|
| `test_failed_run_keeps_its_stage_error` (1) | C-2, B-7 labels | a report without the raised `StageError`; wrong per-key states; blank or shared labels; an `error` not `"StageError: …"` |
| `test_report_round_trips_through_json` (2) | Report JSON v1 | a lossy `to_json`/`from_json` (list-vs-tuple `frames`); a `from_json` that accepts version 2 |
| `test_durations_are_per_task_worker_clock` (3) | C-3 durations, `wall_s` | receipt-stamp or `SUBMITTED`-based durations; a duration for a never-started key |
| `test_fold_takes_each_keys_latest_attempt` (3b) | C-3 lifecycle fold | a first-`STARTED` duration; a sticky error; a lifecycle-scoped label; keeping key 2's first run; no `started` arm; a label other than `""` with no taken `SUBMITTED` |
| `test_report_consumes_what_it_folds` (3c) | C-3 run boundary | a fold over every call since construction |
| `test_environment_digest` (4) | C-4 | an environment not from `capture_environment`; a dropped `container_digest` |
| `test_recorder_forwards_and_forces_the_full_stream` (5) | C-1, C-8 helper | missing forwarding of events, combines, profiles or the profiler factory; a forwarded `lean_events`/`worker_monitor_factory`; a recorder that forwards before it records; a truthy (not `True`) `complete_events` |
| `test_outcomes` (6) | Report JSON `outcome`, C-2 message | `EXHAUSTED` not `completed`; a cancelled run not `cancelled`; `report()` accepting none or both of `result=`/`error=` |

Decisions on the dispatch constraints:
- `RunRecorder(inner=…)` is the constructor spelling (C r7 L2); a recorder with no inner monitor is `RunRecorder()`.
- Test 3 sleeps 0.06 s against a 0.05 s floor (C r1 N1).
- Test 3b adds a key whose only event is a `STARTED`: `partition == ""` (C r1 N1 (5)).
- Test 5's raising inner is exercised through `on_task` only (C r2 N1 (1)).
- Tasks are compared by `state` string; `tasks` is read by iteration only, so a tuple or a list serves.

Mutation check against a scratch prototype (plan-C C-1..C-6, not committed): each row's "Fails" mutant fails
its test (forward-first, no-consume, first-lifecycle, receipt-stamp, sticky-error, lifecycle label, no
`started` arm, dropped digest).
