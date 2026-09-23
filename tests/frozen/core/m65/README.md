# M65 frozen suite — graphed-core slice

Traceability for the debug lane (m65). One section per sub-plan, appended; earlier rows are never
rewritten. Plans live in `graphed-workdir/lanes/debug/`.

## A1 — run control seam and SequentialRunner (`freeze-m65a1`, plan-A1.md)

| Test (`test_m65a1_control.py`) | Contract | Fails |
|---|---|---|
| `test_run_control_state_machine` | A-1 `RunState`, `RunControl` transitions, `apply`, `StopReason.CANCELLED` | a cancel that `resume()`/`pause()` undoes; `apply` accepting an unknown command |
| `test_wait_blocks_while_paused` | A-1 `wait(timeout)` | a non-blocking wait; a waiter not released by resume or cancel |
| `test_sequential_pause_holds_the_next_task` | A-2 PAUSED | a runner that starts the next task before `resume()` is called |
| `test_sequential_cancel_drains_and_folds_completed` | A-2 CANCELLED fold, counts, events, reset | dropping or double-folding a completed partial; starting work after the cancel; no reset |
| `test_sequential_failure_after_cancel_raises` | A-2 failure while draining; reset on raise | a CANCELLED result over a failed task; reset only on return |
| `test_cancel_no_check_saw_is_reset` | A-2 `stopped` only when a check saw it; reset on every exit | a reset keyed on a check having seen the cancel |
| `test_cancelled_control_does_no_work` | A-2 entry under CANCELLED | any event or task run; a non-empty value |
| `test_runner_attributes_are_public` | A-7 | a private or constructor-only `monitor`/`control` |
