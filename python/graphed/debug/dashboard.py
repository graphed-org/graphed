"""The M37 live execution dashboard — an **opt-in**, **passive** observer of a running ``Plan``.

A :class:`Dashboard` is the ``graphed_core.execution.Monitor`` an executor emits through *and* the
owner of a tiny web server (a daemon thread) that streams progress + a sampling flamegraph to a
browser over Server-Sent Events. It works on ``ThreadExecutor``, ``ProcessExecutor``,
``SequentialRunner``, or any future executor honoring the seam — the dashboard only consumes events.

Design (see ``.graphed/M37/decompose.md``):

* **Opt-in** — nothing runs until you construct and ``start()`` (or ``with``) a ``Dashboard``;
  contrast dask's always-on Bokeh server.
* **Passive** — the determinism gate is green attached-or-not. We never call back into the
  executor; emission upstream is best-effort/drop-on-full.
* **Lean rendering** — Python emits JSON; a static SPA (uPlot + d3-flame-graph) renders. No Python
  rendering framework. The statistical sampler is **pyinstrument** behind the ``dashboard`` extra.
"""

from __future__ import annotations

import contextlib
import json
import mimetypes
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

from graphed_core.execution import TaskEvent, TaskPhase, WorkerProfiler

from . import _sampler

if TYPE_CHECKING:
    from graphed_core.execution import Executor

_STATIC = Path(__file__).parent / "static"
_CLIENT_QUEUE_MAX = 512  # per-SSE-client backlog; a slow client drops frames, never blocks updates


class _DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    dashboard: Dashboard


class Dashboard:
    """A passive live-run observer + opt-in web dashboard. Use as a context manager::

        with Dashboard(port=8787, profile=True) as dash:
            ProcessExecutor(monitor=dash.monitor).run(plan)   # or dash.attach(executor)

    or long-lived: ``dash = Dashboard().start(); dash.attach(ex); ...; dash.stop()``.
    """

    def __init__(
        self,
        port: int = 0,
        host: str = "127.0.0.1",
        *,
        profile: bool = False,
        ring: int = 4096,
    ) -> None:
        self._host = host
        self._port = port
        self._profile = bool(profile) and _sampler.pyinstrument_available()
        self._lock = threading.Lock()
        self._t0: float | None = None
        self._counters = {"submitted": 0, "started": 0, "finished": 0, "errored": 0, "combines": 0}
        self._inflight = 0
        self._workers: dict[str, dict[str, int]] = {}
        self._throughput: deque[tuple[float, int]] = deque(maxlen=ring)
        self._recent: deque[dict[str, Any]] = deque(maxlen=256)
        self._sessions: dict[str, Any] = {}
        self._global_session: Any = None
        self._flame: dict[str, Any] = {"name": "root", "value": 0, "children": []}
        self._last_error: dict[str, Any] | None = None
        self._clients: set[queue.Queue[str | None]] = set()
        self._server: _DashboardServer | None = None
        self._thread: threading.Thread | None = None

    # ---- lifecycle ------------------------------------------------------

    def start(self) -> Dashboard:
        if self._server is not None:
            return self  # idempotent
        server = _DashboardServer((self._host, self._port), _Handler)
        server.dashboard = self
        self._server = server
        self._port = server.server_address[1]
        self._thread = threading.Thread(target=server.serve_forever, name="graphed-dashboard", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        server, thread = self._server, self._thread
        self._server, self._thread = None, None
        with self._lock:  # release any streaming clients
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            with contextlib.suppress(queue.Full):
                client.put_nowait(None)
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5.0)

    def attach(self, executor: Executor) -> Executor:
        """Point an executor's passive monitor at this dashboard and return it."""
        executor.monitor = self  # type: ignore[attr-defined]
        return executor

    @property
    def monitor(self) -> Dashboard:
        return self

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}/"

    def __enter__(self) -> Dashboard:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ---- Monitor protocol (called by executors; passive + thread-safe) --

    def on_task(self, event: TaskEvent) -> None:
        with self._lock:
            if self._t0 is None:
                self._t0 = event.t
            self._counters[event.phase.value] = self._counters.get(event.phase.value, 0) + 1
            w = self._workers.setdefault(
                event.worker, {"started": 0, "finished": 0, "errored": 0, "entries": 0}
            )
            if event.phase is TaskPhase.STARTED:
                self._inflight += 1
                w["started"] += 1
            elif event.phase is TaskPhase.FINISHED:
                self._inflight = max(0, self._inflight - 1)
                w["finished"] += 1
                w["entries"] += event.n_entries
                self._throughput.append((event.t - self._t0, self._counters["finished"]))
            elif event.phase is TaskPhase.ERRORED:
                self._inflight = max(0, self._inflight - 1)
                w["errored"] += 1
                self._last_error = {"key": event.key, "worker": event.worker, "message": event.error or ""}
            self._recent.append(
                {
                    "phase": event.phase.value,
                    "key": event.key,
                    "worker": event.worker,
                    "partition": event.partition,
                }
            )
            frame = self._stat_frame()
        self._broadcast(frame)

    def on_profile(self, worker: str, payload: bytes) -> None:
        try:
            session = _sampler.session_from_bytes(payload)
        except Exception:
            return  # a malformed sample must never break the dashboard
        with self._lock:
            prev = self._sessions.get(worker)
            self._sessions[worker] = session if prev is None else _sampler.combine_sessions(prev, session)
            combined: Any = None
            for sess in self._sessions.values():
                combined = sess if combined is None else _sampler.combine_sessions(combined, sess)
            self._global_session = combined
            self._flame = _sampler.flamegraph_tree(combined) if combined is not None else self._flame
            frame = {"type": "flame", "flame": self._flame}
        self._broadcast(frame)

    def on_combine(self, leaves_done: int) -> None:
        with self._lock:
            self._counters["combines"] += 1
            frame = self._stat_frame()
        self._broadcast(frame)

    def worker_profiler_factory(self) -> Callable[[], WorkerProfiler] | None:
        return _sampler.make_worker_profiler if self._profile else None

    # ---- frames + client fan-out ----------------------------------------

    def _elapsed(self) -> float:
        return 0.0 if self._t0 is None else max(0.0, time.perf_counter() - self._t0)

    def _stat_frame(self) -> dict[str, Any]:
        return {
            "type": "stat",
            "elapsed": self._elapsed(),
            "counters": dict(self._counters),
            "inflight": self._inflight,
            "workers": {k: dict(v) for k, v in self._workers.items()},
            "throughput": list(self._throughput)[-1] if self._throughput else None,
            "last_error": self._last_error,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "type": "snapshot",
                "profile": self._profile,
                "elapsed": self._elapsed(),
                "counters": dict(self._counters),
                "inflight": self._inflight,
                "workers": {k: dict(v) for k, v in self._workers.items()},
                "throughput": list(self._throughput),
                "recent": list(self._recent),
                "flame": self._flame,
                "last_error": self._last_error,
            }

    def _broadcast(self, frame: dict[str, Any]) -> None:
        data = json.dumps(frame)
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            with contextlib.suppress(queue.Full):  # a slow client drops frames; never blocks the run
                client.put_nowait(data)

    def _add_client(self, client: queue.Queue[str | None]) -> None:
        with self._lock:
            self._clients.add(client)

    def _remove_client(self, client: queue.Queue[str | None]) -> None:
        with self._lock:
            self._clients.discard(client)


class _Handler(BaseHTTPRequestHandler):
    server: _DashboardServer

    def log_message(self, *args: Any) -> None:  # silence stderr access logs
        pass

    def _dash(self) -> Dashboard:
        return self.server.dashboard

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._serve_file(_STATIC / "index.html", "text/html; charset=utf-8")
        elif path == "/api/state.json":
            self._serve_json(self._dash().snapshot())
        elif path == "/events":
            self._serve_events()
        elif path.startswith("/static/"):
            self._serve_static(path[len("/static/") :])
        else:
            self.send_error(404, "not found")

    def _serve_file(self, file: Path, content_type: str) -> None:
        try:
            body = file.read_bytes()
        except OSError:
            self.send_error(404, "not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, rel: str) -> None:
        target = (_STATIC / rel).resolve()
        if not str(target).startswith(str(_STATIC.resolve())) or not target.is_file():
            self.send_error(404, "not found")  # guard path traversal
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._serve_file(target, ctype)

    def _serve_json(self, obj: dict[str, Any]) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_events(self) -> None:
        dash = self._dash()
        client: queue.Queue[str | None] = queue.Queue(maxsize=_CLIENT_QUEUE_MAX)
        dash._add_client(client)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            self._write_event(json.dumps(dash.snapshot()))
            while True:
                try:
                    data = client.get(timeout=15.0)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")  # comment heartbeat
                    self.wfile.flush()
                    continue
                if data is None:
                    break  # dashboard stopping
                self._write_event(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # client went away
        finally:
            dash._remove_client(client)

    def _write_event(self, data: str) -> None:
        self.wfile.write(b"data: " + data.encode("utf-8") + b"\n\n")
        self.wfile.flush()
