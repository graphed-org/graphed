# m65 — implementer iterations

Debug lane (plans in `graphed-workdir/lanes/debug/`). One section per sub-plan.

## A1 (plan-A1.md, frozen `freeze-m65a1` = `68b5b99`)

Run: `python -m pytest tests/frozen/core/m65 tests/frozen/debug/m65 -q -p no:cacheprovider`.

### Iteration 1 — A1.1 core seam (16 failing → core 8/8)

`RunState`, `RunControl` (one `Condition`; `cancel` sticky, `reset` unconditional), `StopReason.CANCELLED`,
`SequentialRunner(monitor=, control=)` with public attributes: an entry check before any event, a
`control.wait()` before each task, reset of a CANCELLED control in `finally`. Decision on plan r1 L1:
the entry check is one of the checks, and a run entered cancelled reports `stopped=CANCELLED` even for
an empty plan (stated in `docs/core/design.rst`).

### Iteration 2 — A1.2 dashboard control (debug 8/8, core 8/8)

Server: `control=` registers `POST /api/control` only when on (tornado's own 404 otherwise); 415 on a
non-JSON content type; one `try` maps every malformed body to 400. Commands are written on the IOLoop
thread only (the POST handler), so `send_control` stays private (`_send_control`, plan r1 L2); `stop()`
closes the live ingest handlers through `loop.add_callback` before `loop.stop`. Any `hello` makes a
listener (the wire defines only the control hello). Client: `_connect` sends the hello and starts a
daemon reader; the sender drops a connection whose `connected` is false and, with a control, reopens
it on every tick. Error paths reachable only by race (recv non-timeout exception, write to a closed
handler, a failed idle reconnect) are `contextlib.suppress` blocks with no line of their own (plan r2
L1). Dashboard: `control=`, `.control`, attach refuses a missing attribute with `TypeError`.

### Iteration 3 — review r1 repair (M1, L1, L2, N1)

M1: only a `hello` with `control: true` makes a listener (`_server._ingest`); iteration 2's "any
hello" widening would have counted plan-B's lean-only hellos. No first-message clause (no sender
sends a late hello). N1: `_connect` closes the connection when the control hello fails to send. L1:
the m37 `FakeConn` models `connected`. L2: the docs' "every runner" is scoped to runners that take a
`control`. New `tests/extra/debug/m65/test_m65a1_control_hello.py` fails 2/3 hello cases with the old
predicate and the failed-hello case with the old `_connect`.

## B (plan-B.md, frozen `freeze-m65b` = `b5a2a71`)

### Iteration 1 — B1 core hooks, B2 dashboard (graphed B frozen 14/14 first run)

B1: `lean_events`/`worker_monitor_factory` helpers; `SequentialRunner` builds no event without a
monitor and in lean mode emits SUBMITTED plus an unlabelled terminal; a blind partition's label is
`uri:tree:step/n_steps`. B2: `NetworkMonitor.on_task`/`on_combine`/`on_profile` append to one bounded
deque (drop-oldest); the sender builds the wire messages every 50 ms and ships one `batch` frame per
drain, dropping a batch on a failed connect or send; the hello goes out with a control or `lean`.
`per_worker=True` returns `partial(_worker_monitor, url, lean)`: one monitor per process and
`(url, lean)` under a module lock, exit flush bounded by `_WORKER_EXIT_S` through `atexit.register`.
The server ingests batch and single frames, counts ingest connections, and keeps per-key state: the
lifecycle (phase classes seen; a repeated class starts a new one), the key-scoped label and open set;
in lean mode it derives starts and `min(workers seen, open keys)` in-flight, and a late event writes
only the label. The m37 client extras were rewritten for the deque and batch frames. Test 9's fork
control under py3.12 fails as the plan says (`impl/b-test9-fork-control.out`).
