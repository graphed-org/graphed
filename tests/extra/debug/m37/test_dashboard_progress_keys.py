"""The dashboard's overall bar counts task keys, so a retry or a rerun shows each task once."""

from __future__ import annotations

import operator
import time
from typing import Any

import pytest

from graphed.core.execution import TaskEvent, TaskPhase
from graphed.debug import DashboardServer
from graphed.debug.dashboard import _wire

S, ST, F, E = TaskPhase.SUBMITTED, TaskPhase.STARTED, TaskPhase.FINISHED, TaskPhase.ERRORED


def _run(keys: range, worker: str = "w0") -> list[tuple[TaskPhase, int, str]]:
    return [
        *((S, k, "") for k in keys),
        *((ST, k, worker) for k in keys),
        *((F, k, worker) for k in keys),
    ]


RETRY_AFTER_A_LOST_WORKER = [
    *((S, k, "") for k in range(4)),
    *((ST, k, "w0") for k in range(4)),
    *((S, k, "") for k in range(4)),
    *((ST, k, "w1") for k in range(4)),
    *((F, k, "w1") for k in (1, 3, 0, 2)),
]
RETRY_WITH_FINISHES_BEFORE_IT = [
    *((S, k, "") for k in range(4)),
    *((ST, k, "w0") for k in range(4)),
    *((F, k, "w0") for k in (0, 2)),
    *_run(range(4), "w1"),
]
SKIP_FAILED_RERUN = [
    *((S, k, "") for k in range(6)),
    *((S, k, "") for k in range(6)),
    *((ST, k, "w0") for k in range(6)),
    *((F, k, "w0") for k in (0, 3, 5, 1)),
    (E, 2, "w0"),
    (F, 4, "w0"),
]


def _feed(server: DashboardServer, events: list[tuple[TaskPhase, int, str]]) -> dict[str, Any]:
    for phase, key, worker in events:
        server._ingest_task(_wire.task_message(TaskEvent(phase, key, worker, 0.0)))
    return server.progress()


def _bar(progress: dict[str, Any]) -> tuple[int, int, int, int, int]:
    o = progress["overall"]
    return progress["total"], o["submitted"], o["finished"], o["errored"], o["inflight"]


@pytest.mark.parametrize(
    ("events", "bar"),
    [
        (RETRY_AFTER_A_LOST_WORKER, (4, 4, 4, 0, 0)),
        (RETRY_WITH_FINISHES_BEFORE_IT, (4, 4, 4, 0, 0)),
        (SKIP_FAILED_RERUN, (6, 6, 5, 1, 0)),
    ],
    ids=["lost-worker-retry", "lost-worker-retry-ipc", "skip-failed-rerun"],
)
def test_a_rerun_inside_one_run_counts_each_task_once(
    events: list[tuple[TaskPhase, int, str]], bar: tuple[int, ...]
) -> None:
    server = DashboardServer()
    assert _bar(_feed(server, events)) == bar
    stats = server.snapshot()["stats"]
    assert (stats["submitted"], stats["inflight"]) == (2 * bar[0], bar[4])


def _bracket(server: DashboardServer, *, start: bool) -> None:
    server._ingest_one(_wire.run_message(start=start), conn=None, cid=0)


def test_a_second_run_of_one_plan_reshows_its_tasks() -> None:
    server = DashboardServer()
    assert _bar(_feed(server, _run(range(4)))) == (4, 4, 4, 0, 0)
    resubmitted = _feed(server, [(S, k, "") for k in range(4)])
    assert _bar(resubmitted) == (4, 4, 0, 0, 0)
    assert _bar(_feed(server, [(ST, k, "w0") for k in range(2)])) == (4, 4, 0, 0, 2)
    assert _bar(_feed(server, _run(range(4)))) == (4, 4, 4, 0, 0)


def test_a_run_start_shows_another_plan_on_the_same_keys_alone() -> None:
    server = DashboardServer()
    _bracket(server, start=True)
    assert _bar(_feed(server, _run(range(4)))) == (4, 4, 4, 0, 0)
    _bracket(server, start=False)
    _bracket(server, start=True)
    assert _bar(_feed(server, _run(range(2)))) == (2, 2, 2, 0, 0)
    assert server.snapshot()["stats"]["finished"] == 6


@pytest.mark.parametrize("lean", [False, True], ids=["full", "lean"])
def test_a_run_that_ends_with_tasks_unfinished_shows_none_in_flight(lean: bool) -> None:
    server = DashboardServer()
    if lean:
        server._ingest_one(_wire.hello_message(lean=True), conn=None, cid=0)
    _bracket(server, start=True)
    events = [*((S, k, "") for k in range(4)), (ST, 0, "w0"), (F, 0, "w0"), (ST, 1, "w0")]
    assert _bar(_feed(server, events)) == (4, 4, 1, 0, 1)
    _bracket(server, start=False)
    assert _bar(server.progress()) == (4, 4, 1, 0, 0)
    assert server.snapshot()["stats"]["inflight"] == 0
    _bracket(server, start=True)
    assert _bar(_feed(server, [(S, 0, ""), (ST, 0, "w0")])) == (1, 1, 0, 0, 1)


def test_a_network_monitor_forwards_each_run_so_the_bar_shows_the_last_plan() -> None:
    pytest.importorskip("perspective")
    pytest.importorskip("tornado")
    pytest.importorskip("websocket")
    from graphed.core import Partition, Plan, SequentialRunner, Task  # noqa: PLC0415
    from graphed.debug import NetworkMonitor  # noqa: PLC0415

    def plan(n: int) -> Plan[int]:
        tasks = tuple(Task(i, Partition(f"mem://{i}", "", 0, 1)) for i in range(n))
        return Plan(process=lambda p, r: 1, combine=operator.add, empty=int, tasks=tasks)

    server = DashboardServer().start()
    try:
        mon = NetworkMonitor(server.ingest_url)
        with SequentialRunner(monitor=mon) as runner:
            assert runner.run(plan(4)).value == 4
            assert runner.run(plan(2)).value == 2
        mon.close()
        deadline = time.monotonic() + 5.0
        while server.snapshot()["stats"]["finished"] < 6 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.snapshot()["stats"]["finished"] == 6
        assert _bar(server.progress()) == (2, 2, 2, 0, 0)
    finally:
        server.stop()


def test_a_start_that_arrives_after_its_finish_leaves_the_task_finished() -> None:
    server = DashboardServer()
    assert _bar(_feed(server, [(S, 0, ""), (F, 0, "w0"), (ST, 0, "w0")])) == (1, 1, 1, 0, 0)


def test_a_lean_run_bounds_inflight_by_the_workers_seen() -> None:
    server = DashboardServer()
    server._ingest_one(_wire.hello_message(lean=True), conn=None, cid=0)
    events = [*((S, k, "") for k in range(4)), (F, 0, "w0"), (S, 0, ""), (S, 9, "")]
    assert _bar(_feed(server, events)) == (5, 5, 0, 0, 1)


def test_an_unbracketed_run_after_a_bracketed_one_shows_its_tasks_in_flight() -> None:
    server = DashboardServer()
    _bracket(server, start=True)
    _feed(server, _run(range(1)))
    _bracket(server, start=False)
    events = [*((S, k, "") for k in range(3)), *((ST, k, "w0") for k in range(3))]
    assert _bar(_feed(server, events)) == (3, 3, 0, 0, 3)


def test_a_run_message_without_a_boolean_start_is_ignored() -> None:
    server = DashboardServer()
    _feed(server, [(S, 0, ""), (ST, 0, "w0")])
    server._ingest_one({"type": "run"}, conn=None, cid=0)
    assert _bar(server.progress()) == (1, 1, 0, 0, 1)
