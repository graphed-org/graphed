"""M65 A1 frozen suite (graphed-core slice): ``RunControl`` and the runner contract on
``SequentialRunner``. Plain-Python process functions only; this subtree never imports awkward."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

import graphed.core as gc
from graphed.core import Partition, Plan, Task, TaskEvent, TaskPhase

N = 6


class Recorder:
    def __init__(self) -> None:
        self.events: list[TaskEvent] = []

    def on_task(self, event: TaskEvent) -> None:
        self.events.append(event)

    def on_profile(self, worker: str, payload: bytes) -> None:
        return None

    def on_combine(self, leaves_done: int) -> None:
        return None

    def worker_profiler_factory(self) -> None:
        return None

    def keys(self, phase: TaskPhase) -> list[int]:
        return [e.key for e in self.events if e.phase is phase]

    def t(self, phase: TaskPhase, key: int) -> float:
        (t,) = [e.t for e in self.events if e.phase is phase and e.key == key]
        return t


def _one_hot_plan(n: int = N, hooks: dict[int, Any] | None = None) -> Plan[tuple[int, ...]]:
    """Task ``k``'s partial is the one-hot tuple at ``k``; ``hooks[k]()`` runs inside task ``k``."""
    hooks = hooks or {}

    def process(p: Partition, _reader: object) -> tuple[int, ...]:
        k = p.entry_start
        if k in hooks:
            hooks[k]()
        return tuple(int(i == k) for i in range(n))

    def combine(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(x + y for x, y in zip(a, b, strict=True))

    tasks = [Task(k, Partition(f"f{k}.root", "Events", k, k + 1)) for k in range(n)]
    return Plan(process=process, combine=combine, empty=lambda: (0,) * n, tasks=tasks)


def test_run_control_state_machine() -> None:
    RunState = gc.RunState
    c = gc.RunControl()
    assert c.state is RunState.RUNNING
    c.pause()
    assert c.state is RunState.PAUSED
    c.resume()
    assert c.state is RunState.RUNNING
    c.cancel()
    assert c.state is RunState.CANCELLED
    c.pause()
    assert c.state is RunState.CANCELLED
    c.resume()
    assert c.state is RunState.CANCELLED
    c.reset()
    assert c.state is RunState.RUNNING

    a = gc.RunControl()
    a.apply("pause")
    assert a.state is RunState.PAUSED
    a.apply("resume")
    assert a.state is RunState.RUNNING
    a.apply("cancel")
    assert a.state is RunState.CANCELLED
    a.apply("resume")
    assert a.state is RunState.CANCELLED
    with pytest.raises(ValueError, match="'stop'"):
        a.apply("stop")
    assert [s.value for s in RunState] == ["running", "paused", "cancelled"]
    assert gc.StopReason.CANCELLED.value == "cancelled"


def _wait_in_thread(c: Any) -> tuple[threading.Thread, list[Any]]:
    out: list[Any] = []
    th = threading.Thread(target=lambda: out.append(c.wait()), daemon=True)
    th.start()
    return th, out


def test_wait_blocks_while_paused() -> None:
    RunState = gc.RunState
    c = gc.RunControl()
    t0 = time.perf_counter()
    assert c.wait(timeout=5) is RunState.RUNNING
    assert time.perf_counter() - t0 < 1.0

    c.pause()
    t0 = time.perf_counter()
    assert c.wait(timeout=0.2) is RunState.PAUSED
    assert time.perf_counter() - t0 >= 0.1

    th, out = _wait_in_thread(c)
    time.sleep(0.1)
    assert th.is_alive()
    c.resume()
    th.join(timeout=5)
    assert out == [RunState.RUNNING]

    c.pause()
    th, out = _wait_in_thread(c)
    time.sleep(0.1)
    assert th.is_alive()
    c.cancel()
    th.join(timeout=5)
    assert out == [RunState.CANCELLED]
    t0 = time.perf_counter()
    assert c.wait(timeout=5) is RunState.CANCELLED
    assert time.perf_counter() - t0 < 1.0


def test_sequential_pause_holds_the_next_task() -> None:
    n = 4
    baseline = gc.SequentialRunner().run(_one_hot_plan(n))
    control = gc.RunControl()
    resumed_at: list[float] = []

    def resume() -> None:
        resumed_at.append(time.perf_counter())
        control.resume()

    def pause_then_arm_resume() -> None:
        control.pause()
        threading.Timer(0.3, resume).start()

    rec = Recorder()
    res = gc.SequentialRunner(monitor=rec, control=control).run(_one_hot_plan(n, {1: pause_then_arm_resume}))
    assert len(resumed_at) == 1
    assert rec.t(TaskPhase.STARTED, 2) >= resumed_at[0]
    assert (res.value, res.n_partitions, res.n_combines) == (baseline.value, baseline.n_partitions, baseline.n_combines)
    assert res.value == (1,) * n
    assert res.stopped is None
    assert control.state is gc.RunState.RUNNING


def test_sequential_cancel_drains_and_folds_completed() -> None:
    control = gc.RunControl()
    rec = Recorder()
    res = gc.SequentialRunner(monitor=rec, control=control).run(_one_hot_plan(hooks={2: control.cancel}))
    assert res.stopped is gc.StopReason.CANCELLED
    assert res.value == (1, 1, 1, 0, 0, 0)
    assert res.n_partitions == 3
    assert res.n_combines == 2
    assert rec.keys(TaskPhase.SUBMITTED) == list(range(N))
    assert rec.keys(TaskPhase.STARTED) == [0, 1, 2]
    assert rec.keys(TaskPhase.FINISHED) == [0, 1, 2]
    assert control.state is gc.RunState.RUNNING


def test_sequential_failure_after_cancel_raises() -> None:
    control = gc.RunControl()

    def cancel_then_fail() -> None:
        control.cancel()
        raise RuntimeError("m65 fail 2")

    runner = gc.SequentialRunner(control=control)
    with pytest.raises(RuntimeError, match="m65 fail 2"):
        runner.run(_one_hot_plan(hooks={2: cancel_then_fail}))
    assert control.state is gc.RunState.RUNNING


def test_cancel_no_check_saw_is_reset() -> None:
    control = gc.RunControl()
    runner = gc.SequentialRunner(control=control)
    res = runner.run(_one_hot_plan(hooks={N - 1: control.cancel}))
    assert res.value == (1,) * N
    assert res.n_partitions == N
    assert res.stopped is None
    assert control.state is gc.RunState.RUNNING

    again = runner.run(_one_hot_plan())
    assert again.value == (1,) * N
    assert again.n_partitions == N
    assert again.stopped is None


def test_cancelled_control_does_no_work() -> None:
    control = gc.RunControl()
    control.cancel()
    rec = Recorder()
    ran: list[int] = []
    plan = _one_hot_plan(hooks={k: (lambda k=k: ran.append(k)) for k in range(N)})
    res = gc.SequentialRunner(monitor=rec, control=control).run(plan)
    assert rec.events == []
    assert ran == []
    assert res.value == plan.empty()
    assert res.n_partitions == 0
    assert res.stopped is gc.StopReason.CANCELLED
    assert control.state is gc.RunState.RUNNING


def test_runner_attributes_are_public() -> None:
    bare = gc.SequentialRunner()
    assert bare.monitor is None
    assert bare.control is None

    rec, control = Recorder(), gc.RunControl()
    runner = gc.SequentialRunner(monitor=rec, control=control)
    assert runner.monitor is rec
    assert runner.control is control

    late = Recorder()
    bare.monitor = late
    bare.run(_one_hot_plan(2))
    assert late.keys(TaskPhase.FINISHED) == [0, 1]
