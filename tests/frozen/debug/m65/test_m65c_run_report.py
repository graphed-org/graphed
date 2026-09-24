"""M65 C frozen suite (graphed-debug slice): ``RunRecorder`` folds a run's events into a ``RunReport``
(plan-C C-1..C-4, C-8's ``complete_events`` helper, the report JSON). New API is read through module
attributes so each test fails on its own line before the implementation exists."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed.core as gc
import graphed.debug as gd
import graphed.preserve as gp
from graphed import Session
from graphed.aggregate import aggregate_plan
from graphed.awkward import AwkwardBackend, from_parquet, gak
from graphed.core import ExecResult, Partition, Plan, RunControl, StopReason, Task, TaskEvent, TaskPhase
from graphed.debug import SourceFrame, StageError

REPORT_ARGS = "RunRecorder.report needs exactly one of result= or error="

ERR = StageError(
    op="map",
    frames=(SourceFrame("analysis.py", 12, "build", "ev.x.map(f)"),),
    input_forms=("var * float64",),
    partition="u::1/4",
    cause_type="IndexError",
    cause_message="boom",
    opt_level=0,
)


class Inner:
    def __init__(self) -> None:
        self.events: list[TaskEvent] = []
        self.combines: list[int] = []
        self.profiles: list[tuple[str, bytes]] = []

    def on_task(self, event: TaskEvent) -> None:
        self.events.append(event)

    def on_profile(self, worker: str, payload: bytes) -> None:
        self.profiles.append((worker, payload))

    def on_combine(self, leaves_done: int) -> None:
        self.combines.append(leaves_done)

    def worker_profiler_factory(self) -> Any:
        return _profiler_factory


def _profiler_factory() -> None:
    return None


class LeanPushInner(Inner):
    lean_events = True

    def worker_monitor_factory(self) -> Any:
        return Inner


class RaisingInner(Inner):
    def on_task(self, event: TaskEvent) -> None:
        raise RuntimeError("dashboard gone")


class TruthyComplete(Inner):
    complete_events = 1


def _add(a: float, b: float) -> float:
    return a + b


def _zero() -> float:
    return 0.0


def _ok(p: Partition, _r: object) -> float:
    return 1.0


def _raise_on_1(p: Partition, _r: object) -> float:
    if p.blind_step == 1:
        raise ERR
    return 1.0


def _plan(fn: Any, n: int = 4) -> Plan[float]:
    tasks = [Task(k, Partition.blind("u", "t", k, n)) for k in range(n)]
    return Plan(process=fn, combine=_add, empty=_zero, tasks=tasks)


def _failed() -> tuple[Any, StageError]:
    rec = gd.RunRecorder()
    with pytest.raises(StageError) as info:
        gc.SequentialRunner(monitor=rec).run(_plan(_raise_on_1))
    return rec.report(error=info.value), info.value


def _finished() -> Any:
    rec = gd.RunRecorder()
    return rec.report(result=gc.SequentialRunner(monitor=rec).run(_plan(_ok)))


def _states(report: Any) -> list[str]:
    return [t.state for t in report.tasks]


def _ev(
    phase: TaskPhase, key: int, t: float, *, worker: str = "w0", label: str = "", error: str | None = None
) -> TaskEvent:
    return TaskEvent(phase, key, worker, t, label, 1, error=error)


def _bad_chunk(a: Any) -> Any:
    if float(ak.min(a)) >= 250 and float(ak.max(a)) < 500:
        return a[10**6]
    return a


def test_failed_run_keeps_its_stage_error(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = str(tmp_path / "d.parquet")
    ak.to_parquet(ak.Array({"x": np.arange(1000.0)}), path)
    s = Session(AwkwardBackend())
    y = gak.sum(from_parquet(s, "events", path).x.map(_bad_chunk, name="bad"), axis=None)
    plan = aggregate_plan(y, reduce=lambda v: float(v[0]), combine=_add, empty=_zero, steps_per_file=4)
    rec = gd.RunRecorder()
    with pytest.raises(StageError) as info:
        gc.SequentialRunner(monitor=rec).run(plan)
    err = info.value
    r = rec.report(error=err)
    assert r.outcome == "failed"
    assert r.failure == err
    assert r.failed_keys == (1,)
    assert _states(r) == ["finished", "errored", "submitted", "submitted"]
    labels = [t.partition for t in r.tasks]
    assert len(set(labels)) == 4 and all(labels)
    assert r.error.startswith("StageError: ")


def test_report_round_trips_through_json() -> None:
    failed, _ = _failed()
    finished = _finished()
    for r in (failed, finished):
        assert gd.RunReport.from_json(json.loads(json.dumps(r.to_json()))) == r
    assert failed.failure is not None and finished.failure is None
    newer = dict(finished.to_json())
    newer["version"] = 2
    with pytest.raises(ValueError, match="2"):
        gd.RunReport.from_json(newer)


def test_durations_are_per_task_worker_clock() -> None:
    control = RunControl()

    def sleepy(p: Partition, _r: object) -> float:
        time.sleep(0.06)
        if p.blind_step == 1:
            control.cancel()
        return 1.0

    rec = gd.RunRecorder()
    res = gc.SequentialRunner(monitor=rec, control=control).run(_plan(sleepy))
    r = rec.report(result=res)
    by_key = {t.key: t for t in r.tasks}
    assert [by_key[k].state for k in range(4)] == ["finished", "finished", "submitted", "submitted"]
    assert by_key[0].duration_s >= 0.05 and by_key[1].duration_s >= 0.05
    assert by_key[2].duration_s is None and by_key[3].duration_s is None
    assert r.wall_s >= by_key[0].duration_s + by_key[1].duration_s

    rec = gd.RunRecorder()
    rec.on_task(_ev(TaskPhase.SUBMITTED, 0, 0.0, label="L0"))
    rec.on_task(_ev(TaskPhase.STARTED, 0, 5000.0))
    rec.on_task(_ev(TaskPhase.FINISHED, 0, 5000.25))
    r = rec.report(result=ExecResult(1.0, 1, 0, StopReason.EXHAUSTED))
    (task,) = r.tasks
    assert task.duration_s == 0.25
    assert r.wall_s < 1.0


def test_fold_takes_each_keys_latest_attempt() -> None:
    rec = gd.RunRecorder()
    for e in (
        _ev(TaskPhase.SUBMITTED, 0, 1.0, label="L0"),
        _ev(TaskPhase.STARTED, 0, 10.0),
        _ev(TaskPhase.ERRORED, 0, 10.5, error="E: a"),
        _ev(TaskPhase.STARTED, 0, 11.0),
        _ev(TaskPhase.FINISHED, 0, 11.125),
        _ev(TaskPhase.SUBMITTED, 1, 1.0, label="L1"),
        _ev(TaskPhase.STARTED, 1, 20.0),
        _ev(TaskPhase.ERRORED, 1, 20.5, error="E: b"),
        _ev(TaskPhase.STARTED, 1, 21.0, worker="w1"),
        _ev(TaskPhase.ERRORED, 1, 21.25, worker="w1", error="E: c"),
        _ev(TaskPhase.SUBMITTED, 2, 1.0, label="L2"),
        _ev(TaskPhase.STARTED, 2, 40.0),
        _ev(TaskPhase.FINISHED, 2, 40.5),
        _ev(TaskPhase.SUBMITTED, 2, 41.0, label="L2b"),
        _ev(TaskPhase.SUBMITTED, 3, 1.0, label="L3"),
        _ev(TaskPhase.STARTED, 3, 30.0),
    ):
        rec.on_task(e)
    r = rec.report(result=ExecResult(0.0, 4, 3, StopReason.EXHAUSTED))
    t = {x.key: x for x in r.tasks}
    assert [x.key for x in r.tasks] == [0, 1, 2, 3]
    assert (t[0].state, t[0].duration_s, t[0].error) == ("finished", 0.125, None)
    assert (t[1].state, t[1].duration_s, t[1].error) == ("errored", 0.25, "E: c")
    assert (t[1].partition, t[1].worker) == ("L1", "w1")
    assert (t[2].state, t[2].partition, t[2].duration_s) == ("submitted", "L2b", None)
    assert (t[3].state, t[3].partition, t[3].duration_s) == ("started", "L3", None)
    assert r.failed_keys == (1,)

    empty = gd.RunRecorder().report(result=ExecResult(0.0, 0, 0, StopReason.EXHAUSTED))
    assert empty.wall_s == 0.0
    assert list(empty.tasks) == []

    rec = gd.RunRecorder()
    rec.on_task(_ev(TaskPhase.STARTED, 7, 1.0))
    (orphan,) = rec.report(result=ExecResult(0.0, 0, 0, StopReason.EXHAUSTED)).tasks
    assert (orphan.key, orphan.partition, orphan.state) == (7, "", "started")


def test_report_consumes_what_it_folds() -> None:
    rec = gd.RunRecorder()
    runner = gc.SequentialRunner(monitor=rec)
    r1 = rec.report(result=runner.run(_plan(_ok)))
    time.sleep(0.3)
    res2 = runner.run(_plan(_ok, 2))
    r2 = rec.report(result=res2)
    assert len(r1.tasks) == 4
    assert [t.key for t in r2.tasks] == [0, 1]
    assert _states(r2) == ["finished", "finished"]
    assert r2.wall_s < 0.3
    r3 = rec.report(result=res2)
    assert list(r3.tasks) == []
    assert r3.wall_s == 0.0


def test_environment_digest() -> None:
    rec = gd.RunRecorder()
    res = gc.SequentialRunner(monitor=rec).run(_plan(_ok))
    r = rec.report(result=res)
    assert r.environment == gp.capture_environment()
    assert r.environment_digest == gp.fingerprint(gp.capture_environment())
    tagged = rec.report(result=res, container_digest="sha256:abc")
    assert tagged.environment == gp.capture_environment("sha256:abc")
    assert tagged.environment_digest == gp.fingerprint(gp.capture_environment("sha256:abc"))
    assert tagged.environment_digest != r.environment_digest


def test_recorder_forwards_and_forces_the_full_stream() -> None:
    inner = Inner()
    rec = gd.RunRecorder(inner=inner)
    gc.SequentialRunner(monitor=rec).run(_plan(_ok))
    rec.on_combine(3)
    rec.on_profile("w", b"x")
    assert rec.worker_profiler_factory() is _profiler_factory
    assert [(e.phase, e.key) for e in inner.events] == [
        *[(TaskPhase.SUBMITTED, k) for k in range(4)],
        *[(ph, k) for k in range(4) for ph in (TaskPhase.STARTED, TaskPhase.FINISHED)],
    ]
    assert inner.combines == [3]
    assert inner.profiles == [("w", b"x")]
    assert _states(rec.report(result=ExecResult(4.0, 4, 3, StopReason.EXHAUSTED))) == ["finished"] * 4

    lean = LeanPushInner()
    rec = gd.RunRecorder(inner=lean)
    assert gc.lean_events(rec) is False
    assert gc.worker_monitor_factory(rec) is None
    gc.SequentialRunner(monitor=rec).run(_plan(_ok))
    r = rec.report(result=ExecResult(4.0, 4, 3, StopReason.EXHAUSTED))
    assert all(t.duration_s is not None for t in r.tasks)
    assert sum(e.phase is TaskPhase.STARTED for e in lean.events) == 4

    rec = gd.RunRecorder(inner=RaisingInner())
    res = gc.SequentialRunner(monitor=rec).run(_plan(_ok))
    assert res.n_partitions == 4
    assert _states(rec.report(result=res)) == ["finished"] * 4

    assert gc.complete_events(gd.RunRecorder()) is True
    assert gc.complete_events(None) is False
    assert gc.complete_events(TruthyComplete()) is False
    assert gc.complete_events(Inner()) is False


def test_outcomes() -> None:
    rec = gd.RunRecorder()
    res = gc.SequentialRunner(monitor=rec).run(_plan(_ok))
    assert res.stopped is None
    r = rec.report(result=res)
    assert (r.outcome, r.n_partitions) == ("completed", 4)
    exhausted = gd.RunRecorder().report(result=ExecResult(4.0, 4, 3, StopReason.EXHAUSTED))
    assert exhausted.outcome == "completed"

    control = RunControl()

    def cancel_after_1(p: Partition, _r: object) -> float:
        if p.blind_step == 1:
            control.cancel()
        return 1.0

    rec = gd.RunRecorder()
    res = gc.SequentialRunner(monitor=rec, control=control).run(_plan(cancel_after_1))
    assert res.stopped is StopReason.CANCELLED
    r = rec.report(result=res)
    assert (r.outcome, r.n_partitions) == ("cancelled", 2)
    assert _states(r) == ["finished", "finished", "submitted", "submitted"]

    failed, err = _failed()
    assert failed.outcome == "failed"
    assert failed.failure == err
    with pytest.raises(ValueError, match=re.escape(REPORT_ARGS)):
        rec.report()
    with pytest.raises(ValueError, match=re.escape(REPORT_ARGS)):
        rec.report(result=res, error=err)
