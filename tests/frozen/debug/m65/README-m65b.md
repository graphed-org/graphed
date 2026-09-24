# M65 frozen suite, unit B, graphed-debug slice (`freeze-m65b`, plan-B.md)

Frozen and read-only after `freeze-m65b`. A new file beside the A1 `README.md`, which it does not edit
(plan.md decision 4). The module skips only through `pytest.importorskip`.

| Test (`test_m65b_network_monitor.py`) | Contract | Fails |
|---|---|---|
| `test_on_task_builds_no_message_on_the_calling_thread` | B-4 buffer append | an `on_task` that calls `_wire.task_message` on the emitting thread |
| `test_sender_ships_batches_and_reconnects_after_a_send_failure` | B-4 batch frames, drop-and-reconnect | a per-message sender (no multi-item batch frame); no reconnect after a failed send; a connect per drain |
| `test_server_accepts_a_batch_frame` | B-4 server side | a server that ignores `{"type": "batch"}` |
| `test_lean_stream_derives_started_inflight_label_and_start` | B-5 lean derivation | STARTED-based in-flight (0); no label join (`""`); a driver-clock start (key 0 `0.0`, key 2 `2.0`); a derivation over a real STARTED (key 3 `6.0`); double counting (`started 5`, `w1 3`); a STARTED-only per-worker count (`w0 0`); `n_entries` from the terminal (0); a per-worker in-flight that a terminal does not lower |
| `test_lean_inflight_is_an_open_set_under_retry` | B-5 open set, `min(workers seen, open)` | the count formula (1); a `max(0, …)` clamp (1); an uncapped open set (4); a lifecycle-scoped label (`""`) |
| `test_submitted_after_terminal_on_another_connection` | B-5 per-key lifecycle and row rule | an unconditional row write; a rank that never resets; a reset on a repeated SUBMITTED only; a label-only rule that waits for a terminal (`('submitted', 'L3')`) |
| `test_ingest_connections_counts_connections` | B-5 `snapshot()["ingest_connections"]` | a missing or miscounted field |
| `test_per_worker_factory_pushes_from_a_subprocess` | B-6 factory, spawn child, exit flush, `(url, lean)` cache | no factory; an unpicklable one; a child event lost at exit; a driver that connects without emitting; a worker monitor that drops `lean`; no per-process cache; a factory when `per_worker` is off |
| `test_worker_monitor_exit_flush_is_bounded` | B-6 `_WORKER_EXIT_S` | an exit hook that is plain `close()` (5 s connect wait) |

Decisions on the dispatch constraints:
- Test 10 (FIFO with batches) is dropped: m37 `test_network_monitor_streams_tasks_and_combines` already
  pins that property and runs against B's client (B r7 N1).
- Test 4b's fake connection (`m65b_dash_helpers.HeldConnect`) accepts `create_connection(*args, **kwargs)`
  and exposes `connected`, `send`, `recv`, `settimeout`, `ping`, `close` and `sock` (B r6 L2).
- Test 6 pins today's per-worker in-flight rule in lean mode: a STARTED raises it, a terminal lowers it,
  never below 0; a lean terminal derives no increment (B r6 L1).
- Test 6b submits six keys, so the open set (4) exceeds workers seen (2) and the cap is witnessed (B r4 L1).
- Test 7 names key 0's record at its first check, and at the end asserts `n_entries == 7` on every `w0`
  record (B r6 N1).
- Test 9 builds a lean factory as well, and asserts its monitor is lean and differs from the non-lean one
  (B r7 L4).

`m65b_dash_helpers.py`: ingest waits, the `tasks` row and per-worker record reads, the spawn-child entry
and the held fake connection. The fork control for test 9 (`probes/bu1_child_ctx.py`) needs B's code and
runs after implementation.
