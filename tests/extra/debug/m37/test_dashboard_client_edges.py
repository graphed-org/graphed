"""Coverage ratchet: ``dashboard/_client.py`` lifecycle/reconnect branches the frozen suite skips.

The frozen M37 suite always calls ``NetworkMonitor.start()`` before emitting, so lazy autostart
and idempotent ``start()`` never run; it never closes with a non-empty buffer, so the flush-wait
loop never sleeps; and its "no server listening" case fails at ``create_connection``, never at
``conn.send`` on an already-open connection, so the mid-stream reconnect branch never runs either.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

websocket = pytest.importorskip("websocket")

from graphed.debug import NetworkMonitor  # noqa: E402
from graphed.debug.dashboard import _wire  # noqa: E402


def test_emit_before_start_lazily_autostarts_the_sender() -> None:
    mon = NetworkMonitor("ws://127.0.0.1:9/ingest")  # note: no .start() call
    assert mon._started is False
    mon.on_combine(1)  # _emit must autostart it
    assert mon._started is True
    assert mon._thread.is_alive()
    mon.close()


def test_start_is_idempotent() -> None:
    mon = NetworkMonitor("ws://127.0.0.1:9/ingest").start()
    thread = mon._thread
    mon.start()  # a second call must be a no-op, not re-Thread.start() (which would raise)
    assert mon._thread is thread
    assert thread.is_alive()
    mon.close()


def test_close_waits_out_the_flush_deadline_on_a_stalled_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    mon = NetworkMonitor("ws://127.0.0.1:9/ingest")  # never started -> nothing drains the buffer
    mon._buf.append((1, 1))

    # `_client.py` does `import time` too, so patching this (same) module object also redirects it.
    times = iter([100.0, 101.0, 200.0])  # deadline=105; 1st loop check passes, 2nd is past it
    monkeypatch.setattr(time, "monotonic", lambda: next(times))
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)

    mon.close()
    assert slept == [0.02]  # the flush-wait loop ran exactly one iteration before its deadline hit


def test_sender_reconnects_after_a_send_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    connects: list[str] = []
    sent: list[str] = []
    failed = threading.Event()

    class FakeConn:
        def __init__(self, n: int) -> None:
            self.n = n
            self.closed = False
            self.connected = True

        def send(self, msg: str) -> None:
            if self.n == 1:
                failed.set()
                raise ConnectionError("boom")  # simulates a broken mid-stream socket
            sent.append(msg)

        def close(self) -> None:
            self.closed = True
            self.connected = False

    conns: list[FakeConn] = []

    def fake_create_connection(url: str, timeout: float = 5) -> FakeConn:
        connects.append(url)
        conn = FakeConn(len(connects))
        conns.append(conn)
        return conn

    monkeypatch.setattr(websocket, "create_connection", fake_create_connection)

    mon = NetworkMonitor("ws://fake/ingest").start()
    mon.on_combine(1)  # 1st connection opens, then send() raises -> the batch is dropped + closed
    assert failed.wait(5)
    mon.on_combine(2)  # buffer not empty -> a 2nd connection opens and this batch goes through
    deadline = time.monotonic() + 5.0
    while len(sent) < 1 and time.monotonic() < deadline:
        time.sleep(0.02)
    mon.close()

    assert connects == ["ws://fake/ingest", "ws://fake/ingest"]  # reconnected once
    assert conns[0].closed is True  # the broken connection was closed before being dropped
    assert sent == [json.dumps(_wire.batch_message([_wire.combine_message(2)]))]


def test_a_batch_is_dropped_when_no_connection_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def refuse(url: str, timeout: float = 5) -> None:
        calls.append(url)
        raise ConnectionRefusedError(url)

    monkeypatch.setattr(websocket, "create_connection", refuse)
    mon = NetworkMonitor("ws://fake/ingest")
    mon.on_combine(1)
    deadline = time.monotonic() + 5.0
    while mon._buf and time.monotonic() < deadline:
        time.sleep(0.01)
    mon.close()
    assert calls and not mon._buf  # the refused connect drops the batch, the buffer never backs up


def test_a_lean_monitor_says_hello_without_control(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, object]] = []

    class FakeConn:
        connected = True

        def send(self, msg: str) -> None:
            sent.append(json.loads(msg))

        def close(self) -> None:
            self.connected = False

    monkeypatch.setattr(websocket, "create_connection", lambda url, timeout=5: FakeConn())
    mon = NetworkMonitor("ws://fake/ingest", lean=True)
    mon.on_combine(3)
    mon.close()
    assert sent[0] == {"type": "hello", "control": False, "lean": True}
    assert sent[1:] == [_wire.batch_message([_wire.combine_message(3)])]
