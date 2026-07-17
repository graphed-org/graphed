"""M37 frozen suite (graphed-debug slice): the FINOS-Perspective dashboard + websocket network
transport. The server runs its IOLoop in a daemon thread and the client's sender in another, so
everything here executes in-process (coverage sees it). ``importorskip`` lets the free-threaded leg
(no perspective wheel) skip the whole module cleanly."""

from __future__ import annotations

import json
import time
import urllib.request

import pytest

pytest.importorskip("perspective")
pytest.importorskip("tornado")
pytest.importorskip("websocket")

from graphed.core import TaskEvent, TaskPhase
from graphed.debug import (
    Dashboard,
    DashboardServer,
    NetworkMonitor,
    _sampler,
)
from graphed.debug.dashboard import _wire


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


def _spin_profile(duration: float = 0.2) -> bytes:
    """Run the off-thread sampler over a busy task thread long enough to land several samples (10ms
    interval), returning its JSON count-tree payload. Spins on wall-clock (not a fixed iteration
    count) so a fast machine can't finish before the first sample."""
    p = _sampler.make_worker_profiler()
    p.start()
    end = time.monotonic() + duration
    s = 0.0
    i = 0
    while time.monotonic() < end:
        s += (i % 5) ** 0.5
        i += 1
    payload = p.stop()
    assert payload is not None  # at least one sample landed in `duration` at a 10ms interval
    return payload


# ---- pure wire / sampler (no server) -----------------------------------


def test_wire_messages() -> None:
    msg = _wire.task_message(_ev(TaskPhase.STARTED, 1, "w"))
    assert msg["type"] == "task" and msg["phase"] == "started" and msg["error"] == ""
    assert _wire.combine_message(4) == {"type": "combine", "leaves_done": 4}
    assert _wire.profile_message("w", "B64") == {"type": "profile", "worker": "w", "tree_b64": "B64"}
    assert set(_wire.task_row(msg)) == set(_wire.TASKS_SCHEMA)


def test_sampler_tree_and_flamegraph() -> None:
    tree = _sampler.tree_from_bytes(_spin_profile())
    assert tree["name"] == "all" and tree["count"] > 0 and tree["children"]
    fg = _sampler.flamegraph(tree)
    assert fg["name"] == "all" and fg["value"] == tree["count"]

    # inclusive-count invariant (the flamegraph contract): a parent's value covers its children's sum.
    def check(node: dict) -> None:  # type: ignore[type-arg]
        assert node["value"] >= sum(c["value"] for c in node["children"])
        for c in node["children"]:
            check(c)

    check(fg)


def test_worker_profiler_factory_gate() -> None:
    assert (
        NetworkMonitor("ws://x/ingest", profile=True).worker_profiler_factory()
        is _sampler.make_worker_profiler
    )
    assert NetworkMonitor("ws://x/ingest", profile=False).worker_profiler_factory() is None


def test_sampler_edge_cases() -> None:
    p = _sampler.make_worker_profiler()
    assert p.flush() is None and p.stop() is None  # never started -> nothing sampled

    # merge sums inclusive counts per identifier; fresh parses (as the server feeds them) -> additive.
    payload = _spin_profile()
    n = _sampler.tree_from_bytes(payload)["count"]
    acc = _sampler._new_node()
    _sampler.merge_into(acc, _sampler.tree_from_bytes(payload))
    _sampler.merge_into(acc, _sampler.tree_from_bytes(payload))
    assert acc["count"] == 2 * n

    # an empty tree -> a zero-value root with no children (the server's initial state)
    empty = _sampler.flamegraph(_sampler._new_node())
    assert empty["value"] == 0 and empty["children"] == []


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


def test_progress_aggregates_overall_and_per_worker() -> None:
    server = DashboardServer().start()
    try:
        mon = NetworkMonitor(server.ingest_url).start()
        try:
            # 6 tasks split across two workers; leave one in-flight on w1 (started, not finished)
            for k in range(6):
                mon.on_task(_ev(TaskPhase.SUBMITTED, k))
            for k in range(3):
                mon.on_task(_ev(TaskPhase.STARTED, k, "w0"))
                mon.on_task(_ev(TaskPhase.FINISHED, k, "w0"))
            for k in range(3, 6):
                mon.on_task(_ev(TaskPhase.STARTED, k, "w1"))
            for k in range(3, 5):
                mon.on_task(_ev(TaskPhase.FINISHED, k, "w1"))
            snap = _poll(server, lambda s: s["stats"]["finished"] >= 5)
            assert snap["stats"]["inflight"] == 1

            p = server.progress()
            assert p["total"] == 6
            assert p["overall"]["submitted"] == 6 and p["overall"]["finished"] == 5
            assert p["overall"]["inflight"] == 1
            byw = {w["worker"]: w for w in p["workers"]}
            assert sorted(byw) == ["w0", "w1"]  # SUBMITTED (driver, worker="") never makes a row
            assert byw["w0"]["started"] == 3 and byw["w0"]["finished"] == 3 and byw["w0"]["inflight"] == 0
            assert byw["w1"]["started"] == 3 and byw["w1"]["finished"] == 2 and byw["w1"]["inflight"] == 1

            # per-task records (one hoverable cell each): w1 ran keys 3,4,5 -> 2 finished + 1 in-flight
            tasks = byw["w1"]["tasks"]
            assert [t["key"] for t in tasks] == [3, 4, 5]  # sorted by start time, then key
            assert [t["state"] for t in tasks] == ["finished", "finished", "started"]
            for t in tasks:
                assert t["partition"] and t["n_entries"] == t["key"]  # carried from the STARTED event
            assert tasks[0]["t_end"] is not None  # a finished task records its end time (-> duration)
            assert tasks[2]["t_end"] is None  # the in-flight one has no end yet
        finally:
            mon.close()
    finally:
        server.stop()


def test_progress_route_serves_json() -> None:
    server = DashboardServer().start()
    try:
        with urllib.request.urlopen(server.url + "api/progress.json", timeout=5) as resp:
            assert resp.status == 200
            p = json.loads(resp.read())
        # before any task: a well-formed empty progress doc (never a 404 / malformed body)
        assert p["total"] == 0 and p["workers"] == [] and p["overall"]["finished"] == 0
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


def test_profile_ingest_lands_flamegraph() -> None:
    server = DashboardServer().start()
    try:
        mon = NetworkMonitor(server.ingest_url, profile=True).start()
        try:
            mon.on_profile("w0", _spin_profile())
            snap = _poll(server, lambda s: s["profile_samples"] > 0, timeout=10)
            assert snap["profile_samples"] > 0
            fg = server.flamegraph()
            # the merged flamegraph's root value == the total samples ingested, with real frames under it
            assert fg["value"] == snap["profile_samples"] and fg["children"]
        finally:
            mon.close()
    finally:
        server.stop()


def test_profile_ingest_ignores_malformed_payload() -> None:
    server = DashboardServer()  # no start(): _ingest_profile is pure data, no IOLoop needed
    server._ingest_profile({"tree_b64": "@@@not-base64@@@"})  # undecodable -> swallowed, never raises
    server._ingest_profile({})  # missing key -> swallowed too
    assert server.snapshot()["profile_samples"] == 0
    assert server.flamegraph()["value"] == 0


def test_flamegraph_route_serves_json() -> None:
    server = DashboardServer().start()
    try:
        with urllib.request.urlopen(server.url + "api/flamegraph.json", timeout=5) as resp:
            assert resp.status == 200
            fg = json.loads(resp.read())
        # before any profile arrives: a well-formed empty flamegraph (never a 404 / malformed body)
        assert fg["name"] == "all" and fg["value"] == 0 and fg["children"] == []
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
