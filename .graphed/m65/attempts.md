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

### Iteration 2 — review r1 repair (H1, N1)

H1: the server wrote one Perspective `update` per ingested item, so a burst of driver SUBMITTED held
the IOLoop past a fresh worker's bounded exit flush and per-worker push delivered nothing. `_ingest`
now merges a frame's task rows per key and writes one `update` per column set (an indexed update keeps
omitted columns, so sets are never padded together); the lock (now an `RLock`) spans the frame so a
reader waiting on a count sees the rows. New `tests/extra/debug/m65/test_m65b_frame_rows.py` asserts
three 100-row updates for two frames of 100 and 200 items; it fails on iteration 1's server (one update
per row) and on a single padded update (labels lost). N1: the core docs' per-worker bullet names peer
actors and submit backends, a `ThreadBackend`'s included.

### Iteration 3 — review r2 fold (L1, N2)

L1: iteration 2's column-set grouping was unneeded — on perspective-python 4.5.1 a mixed indexed
update keeps every column a row omits, and "labels lost" was wrong. `_write_rows` now makes one
`update` per frame; the frame-rows test asserts `[N, 2 * N]` and still fails on iteration 1's server
(`[1, 1, 1, …]`). N2: the core per-worker bullet no longer claims `ThreadExecutor` peer actors build
their own monitor.

## C (plan-C.md, frozen `freeze-m65c` = `56aa79a`)

### Iteration 1 — C1 RunRecorder/RunReport, C2 bundle reports (graphed C frozen 13/13 first run)

C1: `graphed.debug.report` — `RunRecorder(inner=None)` appends `(event, perf_counter())` and then forwards
(a raising inner monitor cannot cost the record); `report(*, result=, error=, container_digest=)` takes
the calls since the previous report and folds each key with the phase-class lifecycle rule (latest
lifecycle's state/worker/duration/error; latest taken `SUBMITTED` label, `""` without one).
`RunReport.to_json` emits lists only; `from_json` rebuilds `SourceFrame`s and tuples so `StageError.__eq__`
holds. `error` is `"<Type>: <str(error)>"`, so a `StageError` reads `"StageError: " + summary`, which
`inspect` renders (C r7 N1 (1)). `complete_events()` in core (+ `.pyi`, docs: three capabilities);
`capture_environment` public. C2: `attach_run_report` = `Store.put(canonical_bytes(report))` +
`record_done("run-report:<digest>", "", digest, stage="run-report")`; `Bundle.run_reports()` and the
`inspect` section read `completed()` entries whose stage is `run-report` (journal replay keeps first
insertion order, so a re-attach neither duplicates nor reorders). Doc examples in debug and preserve
`design.rst` executed and their printed output compared; sphinx -W ok.

## D (plan-D.md, frozen `freeze-m65d-fixup` = `440e43c`)

### Iteration 1 — D1 capture, D2 describe fallback, D3 replay (graphed D frozen 11/11 first run)

D1: `aggregate_plan(store=)` → `_PartitionReduce.store` (appended, default `None`); a capturing task puts
its chunk before evaluating and its partial after `reduce` through `_open_store(f"{pid}-{tid}")`
(`FsspecStore` for `://`, else `Store`) under `_capture_id` (`_sha256_hex` over a domain tag, the IR and
`_partition_bytes`), which replay reuses. The store-off path is one `is None` test and a `_evaluate`
call. D2: `AwkwardForm.describe` falls back to the scalar typetracer's dtype. D3: `iter_ir` in place
(`run_ir` = `dict(iter_ir(...))`); `graphed.debug.replaying` (`replay`, `Replay`, `Step`, `ReplayDiff`)
reads the input before the first step (a failed re-read raises raw), steps the union of the outputs'
opt_level=0 `lower` cones over the unfused IR's matching nodes, binds `process.externals`, raises
`_stage_error` at the failing cone node, and diffs by value. preserve/checkpoint/numpy import lazily
(worker attribution import check passes). Docs: checkpoint capture section (one run per root),
frontend `store=`, debug "Replaying a task" (example executed, output compared), improvements,
architecture's second coupling, api.rst.

Gates at `74ce4ee` (lane `impl/d-*-2.log`): D frozen 11/11; `COV=1 ./scripts/run-tests.sh` rc 0; per-file gate
fails only the 4 ML externals (jax/pytorch/tensorflow/xgboost, frameworks absent from the lane venv, as A1–C);
diff-cover vs `freeze-m65d-fixup` 100 % (124 lines) on the full run and on a frozen-only (debug + preserve)
run. `git diff freeze-m65d..HEAD -- python/graphed/checkpoint/` empty. Store-off cost: interleaved A/B
(`probes/d_default_ab.out`, base = `freeze-m65d-fixup` tree) puts head within base on every row;
`probes/D.before.out` / `D.after.out` are single runs under varying load.

### Iteration 2 — review r1 repair (M1, L1)

M1: replay bound the task's chunk to every source node and called a missing External evaluator as
`None`. `Replay.steps()` now binds only `process.source_name` and raises the run's own errors, built by
`execute._unbound_source` / `_unbound_external` (factored out of `evaluate_ir`'s raise branches, so run and
replay share one message); a step failure still surfaces as `StageError` with that `GraphedError` as
`__cause__`. `tests/extra/debug/m65/test_m65d_replay_binding.py` (second in-memory source; External with no
evaluator) asserts replay's cause equals the run's error; both fail at `e086bbc` (DID NOT RAISE; cause
`TypeError`) and pass after. L1: the store-off branch of `_PartitionReduce.__call__` calls `evaluate_ir`
inline; the reviewer's `di1/call_overhead.py`, 3 interleaved base/head pairs: head min 1518.8–1524.2 ns vs
base 1512.1–1585.0 ns. N1 dropped: the integrity scan refuses removing the `assert` in `_decode`, and the
defect needs both `python -O` and a corrupt blob. Gates at `af76e14`: D frozen 11/11; `COV=1
./scripts/run-tests.sh` rc 0; per-file gate fails only the 4 ML externals; diff-cover vs `lane/debug-c` 100 %
(138 lines) full run, 98 % frozen-only (missing: the two new raises, covered by the extra test).
