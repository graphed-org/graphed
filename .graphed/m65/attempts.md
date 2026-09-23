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
