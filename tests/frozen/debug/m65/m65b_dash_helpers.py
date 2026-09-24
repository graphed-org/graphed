"""Shared helpers for the M65 B dashboard tests: ingest waits, row reads, a spawn-child entry and a
held fake ``websocket.create_connection``."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from typing import Any

import websocket

from graphed.core import TaskEvent, TaskPhase


def until(pred: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.005)
    raise AssertionError(f"condition not met within {timeout}s")


def wait_stat(server: Any, stat: str, n: int, timeout: float = 5.0) -> None:
    try:
        until(lambda: server.snapshot()["stats"][stat] >= n, timeout)
    except AssertionError:
        raise AssertionError(f"stats[{stat!r}] never reached {n}: {server.snapshot()['stats']}") from None


def row(server: Any, key: int) -> tuple[Any, Any, Any]:
    (r,) = [r for r in server._tasks.view().to_records() if r["key"] == key]
    return (r["phase"], r["partition"], r["n_entries"])


def worker(server: Any, name: str) -> dict[str, Any]:
    found: list[dict[str, Any]] = [w for w in server.progress()["workers"] if w["worker"] == name]
    (w,) = found
    return w


def child_emit(factory: Callable[[], Any]) -> None:
    """Spawn-child entry: build the worker monitor, emit one FINISHED, exit without closing it."""
    factory().on_task(TaskEvent(TaskPhase.FINISHED, 0, "child", 1.0, ""))


class _Sock:
    def settimeout(self, *args: Any) -> None:
        return None

    def setsockopt(self, *args: Any) -> None:
        return None


class FakeConn:
    def __init__(self, owner: HeldConnect, n: int) -> None:
        self.owner, self.n, self.connected, self.first = owner, n, True, True
        self.sock = _Sock()
        self._closed = threading.Event()

    def send(self, data: Any, *args: Any, **kwargs: Any) -> int:
        if self.n == 0 and self.first:
            self.first = False
            self.owner.failed.set()
            raise ConnectionError("m65b send failure")
        self.owner.frames.append((self.n, json.loads(data)))
        return len(data)

    def recv(self, *args: Any, **kwargs: Any) -> Any:
        self._closed.wait()
        raise websocket.WebSocketConnectionClosedException("closed")

    def settimeout(self, *args: Any) -> None:
        return None

    def ping(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self, *args: Any, **kwargs: Any) -> None:
        self.connected = False
        self._closed.set()


class HeldConnect:
    """Stands in for ``websocket.create_connection``: every call is recorded and blocks until
    ``release(i)``; the first connection's first ``send`` raises."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.holds: list[threading.Event] = []
        self.frames: list[tuple[int, dict[str, Any]]] = []
        self.failed = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, *args: Any, **kwargs: Any) -> FakeConn:
        hold = threading.Event()
        with self._lock:
            n = len(self.calls)
            self.calls.append((args, kwargs))
            self.holds.append(hold)
        hold.wait(10)
        return FakeConn(self, n)

    def release(self, i: int) -> None:
        self.holds[i].set()
