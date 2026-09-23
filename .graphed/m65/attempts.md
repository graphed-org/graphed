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
