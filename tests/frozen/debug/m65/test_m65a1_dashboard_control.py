"""M65 A1 frozen suite (graphed-debug slice): browser-to-run control over the dashboard's
``POST /api/control`` route and the ``/ingest`` websocket, into a ``RunControl``."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

pytest.importorskip("perspective")
pytest.importorskip("tornado")
pytest.importorskip("websocket")

from m65a1_control_helpers import Relay, get_json, post, post_cmd, until

import graphed.core as gc
from graphed.core import Partition, Plan, Task, TaskEvent, TaskPhase
from graphed.debug import Dashboard, DashboardServer, NetworkMonitor

STATE_WORD = {"pause": "paused", "resume": "running", "cancel": "cancelled"}


def _listeners(server: Any) -> int:
    n: int = server.snapshot()["control_listeners"]
    return n


def test_post_control_reaches_a_listening_monitor() -> None:
    server = DashboardServer(control=True).start()
    try:
        assert server.snapshot()["control"] == "running"
        before = set(threading.enumerate())
        ctl = gc.RunControl()
        mon = NetworkMonitor(server.ingest_url, control=ctl).start()
        try:
            until(lambda: _listeners(server) == 1)
            assert all(t.daemon for t in set(threading.enumerate()) - before)
            for cmd, state in (("pause", gc.RunState.PAUSED), ("resume", gc.RunState.RUNNING), ("cancel", gc.RunState.CANCELLED)):
                assert post_cmd(server.url, cmd) == (200, {"cmd": cmd, "delivered": 1})
                until(lambda state=state: ctl.state is state)
                assert server.snapshot()["control"] == STATE_WORD[cmd]
                assert server.progress()["control"] == STATE_WORD[cmd]
                assert get_json(server.url + "api/progress.json")["control"] == STATE_WORD[cmd]
        finally:
            mon.close()
    finally:
        server.stop()


def test_resume_reaches_a_monitor_idle_past_its_recv_timeout() -> None:
    server = DashboardServer(control=True).start()
    try:
        ctl = gc.RunControl()
        mon = NetworkMonitor(server.ingest_url, control=ctl).start()
        try:
            until(lambda: _listeners(server) == 1)
            assert post_cmd(server.url, "pause") == (200, {"cmd": "pause", "delivered": 1})
            until(lambda: ctl.state is gc.RunState.PAUSED)
            time.sleep(6.0)
            assert post_cmd(server.url, "resume") == (200, {"cmd": "resume", "delivered": 1})
            until(lambda: ctl.state is gc.RunState.RUNNING)
        finally:
            mon.close()
    finally:
        server.stop()


def test_control_route_refusals() -> None:
    server = DashboardServer(control=True).start()
    try:
        route = server.url + "api/control"
        assert post_cmd(server.url, "stop")[0] == 400
        assert post(route, b"not json")[0] == 400
        assert post(route, b"{}")[0] == 400
        assert post(route, b'{"cmd": "pause"}', content_type="text/plain")[0] == 415
        assert server.snapshot()["control"] == "running"
    finally:
        server.stop()

    off = DashboardServer().start()
    try:
        assert post_cmd(off.url, "pause")[0] == 404
        assert off.snapshot()["control"] is None
        assert get_json(off.url + "api/progress.json")["control"] is None
    finally:
        off.stop()


def test_monitor_without_control_is_not_a_listener() -> None:
    server = DashboardServer(control=True).start()
    try:
        mon = NetworkMonitor(server.ingest_url).start()
        try:
            mon.on_task(TaskEvent(TaskPhase.SUBMITTED, 0, "", 0.0, "f0.root:Events:0-1", 1))
            until(lambda: server.snapshot()["stats"]["submitted"] == 1)
            assert _listeners(server) == 0
            assert post_cmd(server.url, "pause") == (200, {"cmd": "pause", "delivered": 0})
        finally:
            mon.close()
    finally:
        server.stop()


class _Both:
    monitor: object = None
    control: object = "sentinel"


class _MonitorOnly:
    monitor: object = None


class _Neither:
    pass


def test_dashboard_control_is_opt_in() -> None:
    with Dashboard() as dash:
        with pytest.raises(RuntimeError, match="control=True"):
            _ = dash.control
        ex = _Both()
        assert dash.attach(ex) is ex
        assert ex.monitor is dash.monitor
        assert ex.control == "sentinel"
        with pytest.raises(TypeError, match="_Neither"):
            dash.attach(_Neither())

    with Dashboard(control=True) as dash:
        assert isinstance(dash.control, gc.RunControl)
        ex = _Both()
        dash.attach(ex)
        assert ex.monitor is dash.monitor
        assert ex.control is dash.control
        with pytest.raises(TypeError, match="_MonitorOnly"):
            dash.attach(_MonitorOnly())


def _one_hot_plan(n: int, hooks: dict[int, Any]) -> Plan[tuple[int, ...]]:
    def process(p: Partition, _reader: object) -> tuple[int, ...]:
        k = p.entry_start
        if k in hooks:
            hooks[k]()
        return tuple(int(i == k) for i in range(n))

    def combine(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(x + y for x, y in zip(a, b, strict=True))

    tasks = [Task(k, Partition(f"f{k}.root", "Events", k, k + 1)) for k in range(n)]
    return Plan(process=process, combine=combine, empty=lambda: (0,) * n, tasks=tasks)


def _blocking_task() -> tuple[threading.Event, threading.Event, Any]:
    entered, release = threading.Event(), threading.Event()

    def hook() -> None:
        entered.set()
        release.wait(10)

    return entered, release, hook


def test_dashboard_steers_a_sequential_run() -> None:
    n = 20
    with Dashboard(control=True) as dash:
        runner = gc.SequentialRunner()
        assert dash.attach(runner) is runner
        until(lambda: _listeners(dash.server) == 1)

        entered, release, hook = _blocking_task()
        resumed_at: list[float] = []

        def pause_then_resume() -> None:
            entered.wait(10)
            post_cmd(dash.url, "pause")
            until(lambda: dash.control.state is gc.RunState.PAUSED)
            release.set()
            time.sleep(0.3)
            resumed_at.append(time.perf_counter())
            post_cmd(dash.url, "resume")

        th = threading.Thread(target=pause_then_resume, daemon=True)
        th.start()
        res = runner.run(_one_hot_plan(n, {3: hook}))
        th.join(10)
        assert res.value == (1,) * n
        assert res.stopped is None
        dash.wait_for(finished=n, timeout=10)
        (t_start_4,) = [t["t_start"] for w in dash.server.progress()["workers"] for t in w["tasks"] if t["key"] == 4]
        assert len(resumed_at) == 1
        assert t_start_4 > resumed_at[0]

        entered, release, hook = _blocking_task()

        def cancel() -> None:
            entered.wait(10)
            post_cmd(dash.url, "cancel")
            until(lambda: dash.control.state is gc.RunState.CANCELLED)
            release.set()

        th = threading.Thread(target=cancel, daemon=True)
        th.start()
        res = runner.run(_one_hot_plan(n, {3: hook}))
        th.join(10)
        assert res.stopped is gc.StopReason.CANCELLED
        assert res.n_partitions == 4
        assert res.value == (1,) * 4 + (0,) * (n - 4)
        assert dash.control.state is gc.RunState.RUNNING


def test_control_monitor_reconnects_while_idle() -> None:
    relay = Relay()
    first = DashboardServer(control=True).start()
    relay.target = first.port
    ctl = gc.RunControl()
    mon = NetworkMonitor(f"ws://127.0.0.1:{relay.port}/ingest", control=ctl).start()
    try:
        until(lambda: _listeners(first) == 1)
        first.stop()
        second = DashboardServer(control=True).start()
        relay.target = second.port
        try:
            until(lambda: _listeners(second) == 1)
            assert post_cmd(second.url, "pause") == (200, {"cmd": "pause", "delivered": 1})
            until(lambda: ctl.state is gc.RunState.PAUSED)
        finally:
            second.stop()
    finally:
        mon.close()
        relay.close()
