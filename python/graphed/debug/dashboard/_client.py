"""The dashboard **client** (M37): :class:`NetworkMonitor`, a passive ``Monitor`` that forwards a
run's events to a :class:`DashboardServer` over a websocket — loopback for a local dashboard, or
``ws://host:port/ingest`` for a remote one (the network-comms transport).

Passivity: events are enqueued and a background sender thread ships them; a full queue or a down
connection **drops** events and never blocks or raises into the executor, so the determinism gate is
green attached-or-not. ``websocket-client`` is imported lazily (the ``dashboard`` extra).
"""

from __future__ import annotations

import base64
import contextlib
import json
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from graphed_core.execution import TaskEvent, WorkerProfiler

from .. import _sampler
from . import _wire


class NetworkMonitor:
    """A ``graphed_core.execution.Monitor`` that streams events to a dashboard server over a
    websocket. Construct with the server's ingest URL (``DashboardServer.ingest_url``)."""

    def __init__(self, ingest_url: str, *, profile: bool = False, queue_size: int = 10000) -> None:
        self._url = ingest_url
        self._profile = bool(profile) and _sampler.pyinstrument_available()
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=queue_size)
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
        # give the sender a moment to flush what's queued, then stop it
        deadline = time.monotonic() + 5.0
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.02)
        self._stop.set()
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)
        if self._started:
            self._thread.join(timeout=5.0)

    # ---- Monitor protocol (passive; best-effort) ------------------------

    def on_task(self, event: TaskEvent) -> None:
        self._emit(_wire.task_message(event))

    def on_combine(self, leaves_done: int) -> None:
        self._emit(_wire.combine_message(leaves_done))

    def on_profile(self, worker: str, payload: bytes) -> None:
        self._emit(_wire.profile_message(worker, base64.b64encode(payload).decode("ascii")))

    def worker_profiler_factory(self) -> Callable[[], WorkerProfiler] | None:
        return _sampler.make_worker_profiler if self._profile else None

    # ---- internals ------------------------------------------------------

    def _emit(self, message: dict[str, Any]) -> None:
        if not self._started:
            self.start()  # lazy autostart on first event
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(message)  # drop on full -> never back-pressure the run

    def _sender(self) -> None:
        import websocket  # noqa: PLC0415 (optional dep, imported lazily)

        conn: Any = None
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            if conn is None:
                try:
                    conn = websocket.create_connection(self._url, timeout=5)
                except Exception:
                    conn = None
                    continue  # drop this item; retry the connection on the next one
            try:
                conn.send(json.dumps(item))
            except Exception:
                with contextlib.suppress(Exception):
                    conn.close()
                conn = None  # reconnect on the next item
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
