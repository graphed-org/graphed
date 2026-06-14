"""M37 frozen suite (graphed-debug slice): the FINOS-Perspective dashboard + websocket network
transport. The server runs its IOLoop in a daemon thread and the client's sender in another, so
everything here executes in-process (coverage sees it). ``importorskip`` lets the free-threaded leg
(no perspective wheel) skip the whole module cleanly."""

from __future__ import annotations

import time
import urllib.request

import pytest

pytest.importorskip("perspective")
pytest.importorskip("tornado")
pytest.importorskip("websocket")

from graphed_core import TaskEvent, TaskPhase

from graphed_debug import (
    Dashboard,
    DashboardServer,
    NetworkMonitor,
    _sampler,
)
from graphed_debug.dashboard import _wire


def _ev(phase: TaskPhase, key: int, worker: str = "w0", error: str | None = None) -> TaskEvent:
    return TaskEvent(phase, key, worker, float(key), f"f{key}.root:Events:0-{key}", key, error=error)


def _poll(server: DashboardServer, pred, timeout: float = 10.0) -> dict:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = server.snapshot()
        if pred(snap):
            return snap
        time.sleep(0.02)
    raise AssertionError(f"predicate not satisfied within {timeout}s; last={server.snapshot()}")


def _spin_profile(spins: int = 400000) -> bytes:
    p = _sampler.make_worker_profiler()
    p.start()
    s = 0.0
    for i in range(spins):
        s += (i % 5) ** 0.5
    payload = p.stop()
    assert payload is not None
    return payload


# ---- pure wire / adapter (no server) -----------------------------------


def test_wire_messages() -> None:
    msg = _wire.task_message(_ev(TaskPhase.STARTED, 1, "w"))
    assert msg["type"] == "task" and msg["phase"] == "started" and msg["error"] == ""
    assert _wire.combine_message(4) == {"type": "combine", "leaves_done": 4}
    assert _wire.profile_message("w", "B64") == {"type": "profile", "worker": "w", "session_b64": "B64"}
    assert set(_wire.task_row(msg)) == set(_wire.TASKS_SCHEMA)


def test_profile_rows_adapter() -> None:
    session = _sampler.session_from_bytes(_spin_profile())
    rows = _sampler.profile_rows(session, "w0")
    assert rows
    for r in rows:
        assert set(r) == {"function", "location", "worker", "self_us", "total_us"}
        assert r["total_us"] >= r["self_us"] >= 0
        assert r["worker"] == "w0"


def test_worker_profiler_factory_gate() -> None:
    assert (
        NetworkMonitor("ws://x/ingest", profile=True).worker_profiler_factory()
        is _sampler.make_worker_profiler
    )
    assert NetworkMonitor("ws://x/ingest", profile=False).worker_profiler_factory() is None


def test_sampler_edge_cases() -> None:
    p = _sampler.make_worker_profiler()
    assert p.flush() is None and p.stop() is None  # never started -> nothing to take

    class _Frame:
        function = "f"
        time = 0.01
        children = ()  # immutable empty -> no mutable-default lint

        def code_position_short(self) -> None:
            return None

    class _BadFrame(_Frame):
        def code_position_short(self) -> None:
            raise RuntimeError("boom")

    assert _sampler._location(_Frame()) == ""  # no position -> empty
    assert _sampler._location(_BadFrame()) == ""  # a raising accessor -> empty

    class _EmptySession:
        def root_frame(self) -> None:
            return None

    assert _sampler.profile_rows(_EmptySession(), "w") == []


# ---- server + network transport ----------------------------------------


def test_server_routes_and_lifecycle() -> None:
    server = DashboardServer().start()
    try:
        assert server.ingest_url.startswith("ws://") and "/ingest" in server.ingest_url
        with urllib.request.urlopen(server.url, timeout=5) as resp:
            assert resp.status == 200 and b"perspective" in resp.read().lower()
        assert server.snapshot()["stats"]["finished"] == 0
    finally:
        server.stop()


def test_network_monitor_streams_tasks_and_combines() -> None:
    server = DashboardServer().start()
    try:
        mon = NetworkMonitor(server.ingest_url).start()
        try:
            for k in range(5):
                mon.on_task(_ev(TaskPhase.SUBMITTED, k))
                mon.on_task(_ev(TaskPhase.STARTED, k, "w0"))
                mon.on_task(_ev(TaskPhase.FINISHED, k, "w0"))
            mon.on_combine(1)
            mon.on_combine(2)
            # combines are sent last over the (FIFO) websocket, so wait for them: that they landed
            # implies every earlier task message did too (no premature read on a slow runner).
            snap = _poll(server, lambda s: s["stats"]["combines"] >= 2 and s["stats"]["finished"] >= 5)
            assert snap["stats"]["submitted"] == 5
            assert snap["stats"]["started"] == 5
            assert snap["stats"]["finished"] == 5
            assert snap["stats"]["combines"] == 2
            assert snap["stats"]["inflight"] == 0
        finally:
            mon.close()
    finally:
        server.stop()


def test_errored_task_surfaces_in_last_error() -> None:
    server = DashboardServer().start()
    try:
        mon = NetworkMonitor(server.ingest_url).start()
        try:
            mon.on_task(_ev(TaskPhase.STARTED, 9, "w1"))
            mon.on_task(_ev(TaskPhase.ERRORED, 9, "w1", error="StageError in op 'divide' at analysis.py:42"))
            snap = _poll(server, lambda s: s["stats"]["errored"] >= 1)
            assert snap["last_error"]["key"] == 9
            assert "analysis.py:42" in snap["last_error"]["message"]
            assert snap["stats"]["inflight"] == 0
        finally:
            mon.close()
    finally:
        server.stop()


def test_profile_ingest_lands_rows() -> None:
    server = DashboardServer().start()
    try:
        mon = NetworkMonitor(server.ingest_url, profile=True).start()
        try:
            mon.on_profile("w0", _spin_profile())
            snap = _poll(server, lambda s: s["profile_rows"] > 0, timeout=10)
            assert snap["profile_rows"] > 0
        finally:
            mon.close()
    finally:
        server.stop()


def test_network_monitor_best_effort_without_a_server() -> None:
    mon = NetworkMonitor("ws://127.0.0.1:9/ingest").start()  # nothing listening
    mon.on_task(_ev(TaskPhase.STARTED, 0, "w"))  # must not raise or block
    mon.on_combine(1)
    time.sleep(0.3)  # the sender tries to connect, fails, drops
    mon.close()  # returns promptly


# ---- the Dashboard convenience -----------------------------------------


def test_dashboard_convenience_and_attach() -> None:
    with Dashboard() as dash:
        assert dash.url.startswith("http://")
        for k in range(3):
            dash.monitor.on_task(_ev(TaskPhase.SUBMITTED, k))
            dash.monitor.on_task(_ev(TaskPhase.FINISHED, k, "w0"))
        snap = dash.wait_for(finished=3, timeout=10)
        assert snap["stats"]["finished"] == 3

        class _Exec:
            monitor: object = None

            def run(self, plan: object) -> object:
                return None

        ex = _Exec()
        assert dash.attach(ex) is ex
        assert ex.monitor is dash.monitor


def test_wait_for_times_out_when_idle() -> None:
    with Dashboard() as dash, pytest.raises(TimeoutError):
        dash.wait_for(finished=5, timeout=0.3)


def test_optin_lifecycle() -> None:
    dash = Dashboard()
    with pytest.raises(RuntimeError):
        _ = dash.monitor  # not started yet -> no monitor
    dash.start()
    try:
        assert dash.monitor is not None
        assert dash.start() is dash  # idempotent
    finally:
        dash.stop()
