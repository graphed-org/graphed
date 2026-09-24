"""M65 B frozen suite (graphed-debug slice): the buffered, batched ``NetworkMonitor``, lean derivation
in the ``DashboardServer`` and the per-worker monitor factory."""

from __future__ import annotations

import atexit
import json
import multiprocessing
import pickle
import socket
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("perspective")
pytest.importorskip("tornado")
pytest.importorskip("websocket")

import websocket
from m65b_dash_helpers import HeldConnect, child_emit, row, until, wait_stat, worker

import graphed.debug.dashboard._wire as wire
from graphed.core import TaskEvent, TaskPhase
from graphed.debug import DashboardServer, NetworkMonitor

FIN, SUB, STA, ERR = TaskPhase.FINISHED, TaskPhase.SUBMITTED, TaskPhase.STARTED, TaskPhase.ERRORED


def _lean(k: int, w: str, t: float, phase: TaskPhase = FIN) -> TaskEvent:
    return TaskEvent(phase, k, w, t, "", 0)


def _sub(k: int, label: str, n: int, t: float = 0.0) -> TaskEvent:
    return TaskEvent(SUB, k, "driver", t, label, n)


def test_on_task_builds_no_message_on_the_calling_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    idents: list[int] = []
    real = wire.task_message

    def recording(ev: TaskEvent) -> dict[str, Any]:
        idents.append(threading.get_ident())
        return real(ev)

    monkeypatch.setattr(wire, "task_message", recording)
    server = DashboardServer().start()
    try:
        mon = NetworkMonitor(server.ingest_url).start()
        try:
            for k in range(50):
                mon.on_task(TaskEvent(FIN, k, "w0", float(k), f"f{k}", k))
        finally:
            mon.close()
        wait_stat(server, "finished", 50)
        assert server.snapshot()["stats"]["finished"] == 50
    finally:
        server.stop()
    assert idents
    assert threading.get_ident() not in idents


def test_sender_ships_batches_and_reconnects_after_a_send_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = HeldConnect()
    monkeypatch.setattr(websocket, "create_connection", fake)
    mon = NetworkMonitor("ws://127.0.0.1:9/ingest").start()
    try:

        def ev(k: int) -> None:
            mon.on_task(TaskEvent(FIN, k, "w0", float(k), f"f{k}", k))

        ev(0)
        until(lambda: len(fake.calls) >= 1)
        for k in range(1, 5):
            ev(k)
        fake.release(0)
        assert fake.failed.wait(5)
        for k in range(5, 8):
            ev(k)
        until(lambda: len(fake.calls) >= 2)
        for k in range(8, 12):
            ev(k)
        fake.release(1)
        until(lambda: any(n == 1 for n, _ in fake.frames))
        time.sleep(0.3)
    finally:
        for h in fake.holds:
            h.set()
        mon.close()
    assert len(fake.calls) == 2
    batches = [f for n, f in fake.frames if n == 1 and f.get("type") == "batch"]
    assert any(len(f["items"]) > 1 for f in batches), fake.frames


def test_server_accepts_a_batch_frame() -> None:
    server = DashboardServer().start()
    try:
        conn = websocket.create_connection(server.ingest_url, timeout=5)
        try:
            items = [wire.task_message(TaskEvent(FIN, k, "w0", float(k), f"f{k}", k)) for k in range(3)]
            conn.send(json.dumps({"type": "batch", "items": items}))
            wait_stat(server, "finished", 3)
        finally:
            conn.close()
        assert server.snapshot()["stats"]["finished"] == 3
    finally:
        server.stop()


def _records(server: Any, name: str) -> dict[int, dict[str, Any]]:
    return {t["key"]: t for t in worker(server, name)["tasks"]}


def test_lean_stream_derives_started_inflight_label_and_start() -> None:
    assert NetworkMonitor("ws://127.0.0.1:9/ingest").lean_events is False
    server = DashboardServer().start()
    try:
        mon = NetworkMonitor(server.ingest_url, lean=True)
        assert mon.lean_events is True
        mon.start()
        try:
            for k in range(4):
                mon.on_task(_sub(k, f"L{k}", 10 + k, float(k)))
            mon.on_task(_lean(0, "w0", 5.0))
            mon.on_task(_lean(1, "w0", 7.0))
            mon.on_task(_lean(2, "w1", 6.0))
            wait_stat(server, "finished", 3)
            stats = server.snapshot()["stats"]
            assert (stats["submitted"], stats["started"], stats["finished"], stats["inflight"]) == (
                4,
                3,
                3,
                1,
            )
            assert (worker(server, "w0")["started"], worker(server, "w1")["started"]) == (2, 1)
            assert worker(server, "w0")["inflight"] == 0
            w0, w1 = _records(server, "w0"), _records(server, "w1")
            assert (w0[0]["t_start"], w0[1]["t_start"], w1[2]["t_start"]) == (5.0, 5.0, 6.0)
            assert (w0[0]["partition"], w0[1]["partition"], w1[2]["partition"]) == ("L0", "L1", "L2")
            assert (w0[0]["n_entries"], w0[1]["n_entries"], w1[2]["n_entries"]) == (10, 11, 12)
            for k in range(3):
                assert row(server, k) == ("finished", f"L{k}", 10 + k)

            mon.on_task(TaskEvent(STA, 3, "w1", 6.5, "L3", 13))
            wait_stat(server, "started", 4)
            assert server.snapshot()["stats"]["inflight"] == 1
            assert worker(server, "w1")["inflight"] == 1
            mon.on_task(_lean(3, "w1", 8.0))
            wait_stat(server, "finished", 4)
            stats = server.snapshot()["stats"]
            assert (stats["started"], stats["inflight"]) == (4, 0)
            assert (worker(server, "w1")["started"], worker(server, "w1")["inflight"]) == (2, 0)
            rec3 = _records(server, "w1")[3]
            assert (rec3["t_start"], rec3["partition"], rec3["n_entries"]) == (6.5, "L3", 13)
        finally:
            mon.close()
    finally:
        server.stop()


def test_lean_inflight_is_an_open_set_under_retry() -> None:
    server = DashboardServer().start()
    try:
        mon = NetworkMonitor(server.ingest_url, lean=True).start()
        try:
            for k in range(6):
                mon.on_task(_sub(k, f"L{k}", 10 + k))
            for i, w in enumerate(("w0", "w1", "w0", "w1")):
                mon.on_task(TaskEvent(ERR, 0, w, 1.0 + i, "", 0, error="m65b retry"))
            mon.on_task(_lean(1, "w1", 9.0))
            wait_stat(server, "finished", 1)
            wait_stat(server, "errored", 4)
            assert server.snapshot()["stats"]["inflight"] == 2
            w1_key0 = [t for t in worker(server, "w1")["tasks"] if t["key"] == 0]
            assert w1_key0
            assert all(t["partition"] == "L0" for t in w1_key0)
        finally:
            mon.close()
    finally:
        server.stop()


def test_submitted_after_terminal_on_another_connection() -> None:
    server = DashboardServer().start()
    a = NetworkMonitor(server.ingest_url, lean=True).start()
    b = NetworkMonitor(server.ingest_url).start()
    try:

        def fin(n: int) -> None:
            a.on_task(_lean(0, "w0", 5.0))
            wait_stat(server, "finished", n)

        def sub(label: str, n: int, key: int = 0) -> None:
            b.on_task(_sub(key, label, 7))
            wait_stat(server, "submitted", n)

        fin(1)
        sub("L0", 1)
        assert row(server, 0) == ("finished", "L0", 7)
        (rec,) = [t for t in worker(server, "w0")["tasks"] if t["key"] == 0]
        assert (rec["partition"], rec["n_entries"]) == ("L0", 7)
        sub("L1", 2)
        assert row(server, 0) == ("submitted", "L1", 7)
        fin(2)
        fin(3)
        sub("L2", 3)
        assert row(server, 0) == ("finished", "L2", 7)
        a.on_task(TaskEvent(STA, 3, "w0", 6.0, "", 0))
        wait_stat(server, "started", 4)
        sub("L3", 4, key=3)
        assert row(server, 3) == ("started", "L3", 7)
        assert all(t["n_entries"] == 7 for t in worker(server, "w0")["tasks"])
    finally:
        a.close()
        b.close()
        server.stop()


def test_ingest_connections_counts_connections() -> None:
    server = DashboardServer().start()
    try:
        mons = [NetworkMonitor(server.ingest_url).start() for _ in range(2)]
        try:
            for k, m in enumerate(mons):
                m.on_task(TaskEvent(FIN, k, "w0", float(k), f"f{k}", k))
            wait_stat(server, "finished", 2)
            assert server.snapshot()["ingest_connections"] == 2
        finally:
            for m in mons:
                m.close()
    finally:
        server.stop()


def _capture_atexit(monkeypatch: pytest.MonkeyPatch) -> list[Callable[[], Any]]:
    hooks: list[Callable[[], Any]] = []

    def register(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Callable[..., Any]:
        hooks.append(lambda: fn(*args, **kwargs))
        return fn

    monkeypatch.setattr(atexit, "register", register)
    return hooks


def test_per_worker_factory_pushes_from_a_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    server = DashboardServer().start()
    try:
        driver = NetworkMonitor(server.ingest_url, per_worker=True)
        factory = driver.worker_monitor_factory()
        assert factory is not None
        shipped = pickle.loads(pickle.dumps(factory))
        child = multiprocessing.get_context("spawn").Process(target=child_emit, args=(shipped,))
        child.start()
        child.join(60)
        assert child.exitcode == 0
        wait_stat(server, "finished", 1, timeout=10)
        assert server.snapshot()["ingest_connections"] == 1
        (rec,) = worker(server, "child")["tasks"]
        assert rec["key"] == 0

        hooks = _capture_atexit(monkeypatch)
        built = factory()
        assert isinstance(built, NetworkMonitor)
        assert built.lean_events is False
        assert factory() is built
        lean_factory = NetworkMonitor(server.ingest_url, per_worker=True, lean=True).worker_monitor_factory()
        assert lean_factory is not None
        lean_built = lean_factory()
        assert lean_built.lean_events is True
        assert lean_built is not built
        assert NetworkMonitor(server.ingest_url).worker_monitor_factory() is None
        for h in reversed(hooks):
            h()
    finally:
        server.stop()


def test_worker_monitor_exit_flush_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    lsock = socket.create_server(("127.0.0.1", 0))
    try:
        port = lsock.getsockname()[1]
        hooks = _capture_atexit(monkeypatch)
        factory = NetworkMonitor(f"ws://127.0.0.1:{port}/ingest", per_worker=True).worker_monitor_factory()
        assert factory is not None
        factory().on_task(TaskEvent(FIN, 0, "w0", 1.0, ""))
        time.sleep(0.1)
        assert hooks
        t0 = time.perf_counter()
        for h in reversed(hooks):
            h()
        assert time.perf_counter() - t0 < 1.0
    finally:
        lsock.close()
