"""``MonitorBase``, the guarded run hooks, and ``ProgressBar`` on the in-process runner."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from typing import Any

import pytest

from graphed.core import (
    DurablePlanV2,
    MonitorBase,
    Partition,
    Plan,
    RunControl,
    SequentialRunner,
    StopReason,
    Task,
    TaskEvent,
    TaskPhase,
    complete_events,
    lean_events,
    observed_run,
    worker_monitor_factory,
    worker_profiler_factory,
)
from graphed.debug import ProgressBar, RunRecorder


def _tasks(n: int) -> list[Task]:
    return [Task(i, Partition("f.root", "Events", i, i + 1)) for i in range(n)]


def _plan(n: int, process: Callable[[Partition, object], int] | None = None) -> Plan[int]:
    return Plan(
        process=process or (lambda part, _: part.entry_start + 1),
        combine=lambda a, b: a + b,
        empty=lambda: 0,
        tasks=_tasks(n),
    )


def _event(phase: TaskPhase, key: int = 0) -> TaskEvent:
    return TaskEvent(phase, key, "w", 0.0)


class OnlyOnTask:
    def __init__(self) -> None:
        self.phases: list[TaskPhase] = []

    def on_task(self, event: TaskEvent) -> None:
        self.phases.append(event.phase)


class Hooks(MonitorBase):
    def __init__(self) -> None:
        self.calls: list[object] = []

    def on_run_start(self, n_tasks: int) -> None:
        self.calls.append(n_tasks)

    def on_run_end(self) -> None:
        self.calls.append("end")


class RaisingHooks(MonitorBase):
    def on_run_start(self, n_tasks: int) -> None:
        raise RuntimeError("start")

    def on_run_end(self) -> None:
        raise RuntimeError("end")


class RaisingFactory(MonitorBase):
    def worker_profiler_factory(self) -> Any:
        raise RuntimeError("factory")


class WithFactory(MonitorBase):
    def worker_profiler_factory(self) -> Any:
        return object


@pytest.fixture(params=["tqdm", "text"])
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.param == "text":
        monkeypatch.setitem(sys.modules, "tqdm.auto", None)


class OverridesOnTask(OnlyOnTask, MonitorBase):
    pass


@pytest.mark.parametrize("cls", [OnlyOnTask, OverridesOnTask])
def test_a_monitor_with_only_on_task_runs(cls: Any) -> None:
    monitor = cls()
    assert SequentialRunner(monitor=monitor).run(_plan(2)).value == 3
    assert monitor.phases.count(TaskPhase.FINISHED) == 2


@pytest.mark.parametrize(
    "monitor",
    [None, OnlyOnTask(), MonitorBase(), RaisingFactory()],
    ids=lambda m: type(m).__name__,
)
def test_no_factory_from_a_monitor_without_one(monitor: Any) -> None:
    assert worker_profiler_factory(monitor) is None


def test_a_raising_worker_monitor_factory_gives_no_factory() -> None:
    class Raising(MonitorBase):
        def worker_monitor_factory(self) -> Any:
            raise RuntimeError("factory")

    assert worker_monitor_factory(Raising()) is None


def test_a_monitor_factory_comes_through() -> None:
    assert worker_profiler_factory(WithFactory()) is object


def test_run_hooks_bracket_a_run() -> None:
    hooks = Hooks()
    SequentialRunner(monitor=hooks).run(_plan(3))
    assert hooks.calls == [3, "end"]


def test_run_end_is_reported_when_the_run_raises() -> None:
    def boom(part: Partition, _: object) -> int:
        raise ValueError("bad")

    hooks = Hooks()
    with pytest.raises(ValueError, match="bad"):
        SequentialRunner(monitor=hooks).run(_plan(1, boom))
    assert hooks.calls == [1, "end"]


def test_a_staged_plan_reports_no_tasks() -> None:
    hooks = Hooks()
    with observed_run(hooks, DurablePlanV2(ir=b"")):
        pass
    assert hooks.calls == [0, "end"]


@pytest.mark.parametrize(
    "monitor", [None, OnlyOnTask(), RaisingHooks()], ids=lambda m: type(m).__name__
)
def test_run_hooks_cannot_break_a_run(monitor: Any) -> None:
    assert SequentialRunner(monitor=monitor).run(_plan(2)).value == 3


def test_run_hook_failure_does_not_hide_the_run_error() -> None:
    with pytest.raises(ValueError, match="bad"), observed_run(RaisingHooks(), _plan(1)):
        raise ValueError("bad")


def test_bar_reaches_total(backend: None, capsys: pytest.CaptureFixture[str]) -> None:
    result = SequentialRunner(monitor=ProgressBar()).run(_plan(3))
    assert result.value == SequentialRunner().run(_plan(3)).value
    assert "3/3" in capsys.readouterr().err


def test_a_cancelled_run_closes_the_bar(backend: None, capsys: pytest.CaptureFixture[str]) -> None:
    control = RunControl()

    def cancel_after_first(part: Partition, _: object) -> int:
        control.cancel()
        return 1

    bar = ProgressBar()
    result = SequentialRunner(monitor=bar, control=control).run(_plan(3, cancel_after_first))
    assert result.stopped is StopReason.CANCELLED
    err = capsys.readouterr().err
    assert "1/3" in err
    assert "3/3" not in err
    assert err.endswith("\n")


def test_a_failed_task_does_not_tick_and_the_bar_closes(
    backend: None, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(part: Partition, _: object) -> int:
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        SequentialRunner(monitor=ProgressBar()).run(_plan(2, boom))
    err = capsys.readouterr().err
    assert "1/2" not in err
    assert err.endswith("\n")


def test_one_bar_watches_runs_in_turn(backend: None, capsys: pytest.CaptureFixture[str]) -> None:
    runner = SequentialRunner(monitor=ProgressBar())
    runner.run(_plan(2))
    runner.run(_plan(3))
    err = capsys.readouterr().err
    assert "2/2" in err
    assert "3/3" in err


def test_total_follows_tasks_beyond_the_plans_own(
    backend: None, capsys: pytest.CaptureFixture[str]
) -> None:
    bar = ProgressBar()
    bar.on_run_start(1)
    for _ in range(3):
        bar.on_task(_event(TaskPhase.SUBMITTED))
    for key in range(3):
        bar.on_task(_event(TaskPhase.STARTED, key))
        bar.on_task(_event(TaskPhase.FINISHED, key))
    bar.on_run_end()
    assert "3/3" in capsys.readouterr().err


def test_growing_the_total_does_not_redraw(
    backend: None, capsys: pytest.CaptureFixture[str]
) -> None:
    bar = ProgressBar()
    bar.on_run_start(0)
    capsys.readouterr()
    for _ in range(100):
        bar.on_task(_event(TaskPhase.SUBMITTED))
    assert capsys.readouterr().err == ""
    bar.on_run_end()


def test_a_retried_task_ticks_once(backend: None, capsys: pytest.CaptureFixture[str]) -> None:
    bar = ProgressBar()
    bar.on_run_start(2)
    bar.on_task(_event(TaskPhase.ERRORED, 0))
    bar.on_task(_event(TaskPhase.FINISHED, 0))
    bar.on_task(_event(TaskPhase.FINISHED, 1))
    bar.on_run_end()
    assert "2/2" in capsys.readouterr().err


def test_an_event_after_the_run_is_ignored(
    backend: None, capsys: pytest.CaptureFixture[str]
) -> None:
    bar = ProgressBar()
    bar.on_run_start(1)
    bar.on_run_end()
    capsys.readouterr()
    bar.on_task(_event(TaskPhase.FINISHED))
    bar.on_run_end()
    assert capsys.readouterr().err == ""


def test_text_line_is_throttled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "tqdm.auto", None)
    bar = ProgressBar()
    bar.on_run_start(1000)
    for key in range(1000):
        bar.on_task(_event(TaskPhase.FINISHED, key))
    bar.on_run_end()
    err = capsys.readouterr().err
    assert err.count("\r") < 10
    assert err.endswith("\rtasks: 1000/1000\n")


def test_run_recorder_forwards_the_hooks_to_its_inner_monitor() -> None:
    hooks = Hooks()
    SequentialRunner(monitor=RunRecorder(hooks)).run(_plan(2))
    assert hooks.calls == [2, "end"]


def test_run_recorder_without_inner_monitor_runs() -> None:
    recorder = RunRecorder()
    assert SequentialRunner(monitor=recorder).run(_plan(2)).value == 3
    assert recorder.worker_profiler_factory() is None


def test_bar_asks_for_every_terminal_event_and_nothing_else() -> None:
    assert lean_events(ProgressBar())
    assert complete_events(ProgressBar())


def test_importing_debug_leaves_tqdm_unimported() -> None:
    code = "import sys, graphed.debug; print(any(m.startswith('tqdm') for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
