"""The dashboard **client** (M37): :class:`NetworkMonitor`, a passive ``Monitor`` that forwards a
run's events to a :class:`DashboardServer` over a websocket — loopback for a local dashboard, or
``ws://host:port/ingest`` for a remote one (the network-comms transport).

Passivity: an event is appended to a bounded in-process buffer (drop-oldest when full) and a
background sender thread drains it on a short cadence, building the wire messages and shipping each
drain as one ``batch`` frame; a down connection **drops** that batch and never blocks or raises into
the executor, so the determinism gate is green attached-or-not. Given a ``control=RunControl``, the
monitor also carries commands the other way, and a pause or cancel from the dashboard does change the
run it steers. ``lean=True`` opts into lean events and ``per_worker=True`` offers a factory for one
monitor per worker process (see ``graphed.core.lean_events`` / ``worker_monitor_factory``).
``websocket-client`` is imported lazily (the ``dashboard`` extra).
"""

from __future__ import annotations

import atexit
import base64
import contextlib
import functools
import json
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from graphed.core.execution import Monitor, RunControl, TaskEvent, WorkerProfiler

from .. import _sampler
from . import _wire

_SEND_INTERVAL_S = 0.05  # the sender's drain cadence
_WORKER_EXIT_S = 0.5  # total budget of a worker monitor's exit flush (buffer wait + sender join)
_TASK, _COMBINE, _PROFILE = 0, 1, 2


class NetworkMonitor:
    """A ``graphed.core.execution.Monitor`` that streams events to a dashboard server over a
    websocket. Construct with the server's ingest URL (``DashboardServer.ingest_url``).

    With ``control``, it connects at :meth:`start`, says ``hello`` on every connection, keeps a
    connection open while idle, and applies each command the server relays to ``control``. With
    ``lean``, it says ``hello`` too, so the server derives what lean events leave out."""

    def __init__(
        self,
        ingest_url: str,
        *,
        profile: bool = False,
        queue_size: int = 10000,
        control: RunControl | None = None,
        lean: bool = False,
        per_worker: bool = False,
    ) -> None:
        self._url = ingest_url
        self._control = control
        self.lean_events = bool(lean)
        self._per_worker = per_worker
        self._profile = bool(profile) and _sampler.sampler_available()
        self._buf: deque[tuple[int, Any]] = deque(maxlen=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sender, name="graphed-dash-client", daemon=True)
        self._started = False
        self._lock = threading.Lock()

    def start(self) -> NetworkMonitor:
        with self._lock:
            if not self._started:
                self._thread.start()
                self._started = True
        return self

    def close(self) -> None:
        self._shutdown(5.0, join_s=5.0)

    def _exit_flush(self) -> None:
        self._shutdown(_WORKER_EXIT_S)

    def _shutdown(self, drain_s: float, join_s: float | None = None) -> None:
        # give the sender up to ``drain_s`` to ship what's buffered, then stop it; without ``join_s``
        # the join shares that budget
        deadline = time.monotonic() + drain_s
        while self._buf and time.monotonic() < deadline:
            time.sleep(0.02)
        self._stop.set()
        if self._started:
            self._thread.join(join_s if join_s is not None else max(0.0, deadline - time.monotonic()))

    # ---- Monitor protocol (passive; best-effort) ------------------------

    def on_task(self, event: TaskEvent) -> None:
        self._emit((_TASK, event))

    def on_combine(self, leaves_done: int) -> None:
        self._emit((_COMBINE, leaves_done))

    def on_profile(self, worker: str, payload: bytes) -> None:
        self._emit((_PROFILE, (worker, payload)))

    def worker_profiler_factory(self) -> Callable[[], WorkerProfiler] | None:
        return _sampler.make_worker_profiler if self._profile else None

    def worker_monitor_factory(self) -> Callable[[], Monitor] | None:
        """With ``per_worker``, a picklable factory each worker process calls for its own monitor."""
        return functools.partial(_worker_monitor, self._url, self.lean_events) if self._per_worker else None

    # ---- internals ------------------------------------------------------

    def _emit(self, item: tuple[int, Any]) -> None:
        if not self._started:
            self.start()  # lazy autostart on first event
        self._buf.append(item)  # a full buffer drops its oldest item -> never back-pressure the run

    def _connect(self) -> Any:
        import websocket  # noqa: PLC0415 (optional dep, imported lazily)

        conn = websocket.create_connection(self._url, timeout=5)
        if self._control is not None or self.lean_events:
            try:
                conn.send(
                    json.dumps(_wire.hello_message(control=self._control is not None, lean=self.lean_events))
                )
            except Exception:
                conn.close()
                raise
        if self._control is not None:
            threading.Thread(
                target=self._reader, args=(conn,), name="graphed-dash-control", daemon=True
            ).start()
        return conn

    def _reader(self, conn: Any) -> None:
        import websocket  # noqa: PLC0415 (optional dep, imported lazily)

        assert self._control is not None
        # any recv() failure but a timeout ends this connection
        with contextlib.suppress(Exception):
            while True:
                try:
                    text = conn.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if not conn.connected:  # the server's close frame: recv() returned without raising
                    break
                with contextlib.suppress(ValueError, LookupError, TypeError):
                    self._control.apply(json.loads(text)["cmd"])
        conn.shutdown()  # close() is a no-op once connected is false

    def _drain(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        while True:
            try:
                kind, data = self._buf.popleft()
            except IndexError:
                return messages
            if kind == _TASK:
                messages.append(_wire.task_message(data))
            elif kind == _COMBINE:
                messages.append(_wire.combine_message(data))
            else:
                worker, payload = data
                messages.append(_wire.profile_message(worker, base64.b64encode(payload).decode("ascii")))

    def _sender(self) -> None:
        conn: Any = None
        while True:
            final = self._stop.is_set()  # after a stop, one last drain goes out
            if conn is not None and not conn.connected:
                conn = None
            if conn is None and (self._buf or (self._control is not None and not final)):
                with contextlib.suppress(Exception):  # the server is down: retry on the next tick
                    conn = self._connect()
            messages = self._drain()
            if messages and conn is not None:  # with no connection the batch is dropped
                try:
                    conn.send(json.dumps(_wire.batch_message(messages)))
                except Exception:
                    with contextlib.suppress(Exception):
                        conn.close()
                    conn = None  # the batch is dropped; the next drain reconnects
            if final:
                break
            self._stop.wait(_SEND_INTERVAL_S)
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()


_worker_monitors: dict[tuple[str, bool], NetworkMonitor] = {}
_worker_monitors_lock = threading.Lock()


def _worker_monitor(url: str, lean: bool) -> NetworkMonitor:
    """One monitor per worker process and ``(url, lean)``; it connects on its first event and its
    exit flush is bounded by ``_WORKER_EXIT_S``, so a silent dashboard never holds a worker's exit."""
    with _worker_monitors_lock:
        mon = _worker_monitors.get((url, lean))
        if mon is None:
            mon = _worker_monitors[(url, lean)] = NetworkMonitor(url, lean=lean)
            atexit.register(mon._exit_flush)
    return mon
