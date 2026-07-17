"""The dashboard **server** (M37): a `perspective` ``Server`` hosting the live ``stats`` table (and a
``tasks`` table fed by the same event stream) over a Tornado app, plus two derived JSON views the
browser polls — a merged profile **flamegraph** at ``/api/flamegraph.json`` and overall + per-worker
**progress** at ``/api/progress.json`` (the dask-style progress bars). Browsers connect a
``<perspective-viewer>`` to the ``/websocket`` endpoint; executors push events to the ``/ingest``
websocket (see :class:`graphed.debug.dashboard.NetworkMonitor`). It runs its own asyncio/Tornado
IOLoop in a daemon thread, so it is decoupled from the executor — the same server serves a local *or*
a remote run.

perspective/tornado are imported lazily (the ``dashboard`` extra), so ``import graphed.debug`` works
without them; :meth:`DashboardServer.start` raises a clear error if they are missing.
"""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from typing import Any

from .. import _sampler
from . import _wire

_STATIC = Path(__file__).parent / "static"


class DashboardServer:
    """Hosts the live Perspective tables and the ingest/viewer websockets. Start it, point one or
    more executors' :class:`NetworkMonitor` at :attr:`ingest_url`, and open :attr:`url` in a browser.
    Thread-safe: all Perspective table writes happen on the IOLoop thread (the ingest handler);
    :meth:`snapshot` reads a plain-Python mirror under a lock."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._host = host
        self._port = port
        self._thread: threading.Thread | None = None
        self._loop: Any = None
        self._http: Any = None
        self._client: Any = None
        self._tasks: Any = None
        self._stats_table: Any = None
        self._lock = threading.Lock()
        self._stats: dict[str, int] = dict.fromkeys(_wire.STATS_KEYS, 0)
        self._workers: dict[str, dict[str, Any]] = {}  # worker -> per-worker progress (for the bars)
        self._last_error: dict[str, Any] | None = None
        self._profile_tree: dict[str, Any] = _sampler._new_node()  # merged sampled stacks -> flamegraph
        self._profile_samples = 0
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._started = False

    # ---- lifecycle ------------------------------------------------------

    def start(self) -> DashboardServer:
        if self._started:
            return self
        self._thread = threading.Thread(target=self._run, name="graphed-dashboard", daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError(f"dashboard server failed to start: {self._error}")
        self._started = True
        return self

    def stop(self) -> None:
        loop = self._loop
        if loop is not None:
            loop.add_callback(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._started = False

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}/"

    @property
    def ingest_url(self) -> str:
        return f"ws://{self._host}:{self._port}/ingest"

    @property
    def port(self) -> int:
        return self._port

    # ---- the IOLoop thread ----------------------------------------------

    def _run(self) -> None:
        try:
            import asyncio  # noqa: PLC0415

            import perspective  # noqa: PLC0415
            from perspective.handlers.tornado import PerspectiveTornadoHandler  # noqa: PLC0415
            from tornado.httpserver import HTTPServer  # noqa: PLC0415
            from tornado.ioloop import IOLoop  # noqa: PLC0415
            from tornado.netutil import bind_sockets  # noqa: PLC0415
            from tornado.web import Application, RequestHandler, StaticFileHandler  # noqa: PLC0415
            from tornado.websocket import WebSocketHandler  # noqa: PLC0415

            asyncio.set_event_loop(asyncio.new_event_loop())
            self._loop = IOLoop.current()
            server = perspective.Server()
            self._client = server.new_local_client()
            self._tasks = self._client.table(_wire.TASKS_SCHEMA, index="key", name="tasks")
            self._stats_table = self._client.table(_wire.STATS_SCHEMA, index="metric", name="stats")
            self._push_stats_table()

            owner = self

            class _Ingest(WebSocketHandler):  # type: ignore[misc]  # tornado base is untyped (Any)
                def check_origin(self, origin: str) -> bool:
                    return True

                def on_message(self, message: str | bytes) -> None:
                    owner._ingest(message if isinstance(message, str) else message.decode("utf-8"))

            class _Index(RequestHandler):  # type: ignore[misc]  # tornado base is untyped (Any)
                def get(self) -> None:
                    self.set_header("Content-Type", "text/html; charset=utf-8")
                    self.finish((_STATIC / "index.html").read_bytes())

            class _Flame(RequestHandler):  # type: ignore[misc]  # the merged profile flamegraph (JSON)
                def get(self) -> None:
                    self.set_header("Content-Type", "application/json")
                    self.set_header("Cache-Control", "no-store")
                    self.finish(owner.flamegraph_json())

            class _Progress(RequestHandler):  # type: ignore[misc]  # overall + per-worker progress (JSON)
                def get(self) -> None:
                    self.set_header("Content-Type", "application/json")
                    self.set_header("Cache-Control", "no-store")
                    self.finish(owner.progress_json())

            app = Application(
                [
                    (r"/websocket", PerspectiveTornadoHandler, {"perspective_server": server}),
                    (r"/ingest", _Ingest),
                    (r"/api/flamegraph.json", _Flame),
                    (r"/api/progress.json", _Progress),
                    (r"/static/(.*)", StaticFileHandler, {"path": str(_STATIC)}),
                    (r"/", _Index),
                ]
            )
            socks = bind_sockets(self._port, address=self._host)
            self._port = socks[0].getsockname()[1]
            self._http = HTTPServer(app)
            self._http.add_sockets(socks)
            self._ready.set()
            self._loop.start()  # blocks until stop()
            self._http.stop()
        except BaseException as exc:  # surface a startup failure to start()
            self._error = exc
            self._ready.set()

    # ---- ingest (runs on the IOLoop thread) -----------------------------

    def _ingest(self, message: str) -> None:
        try:
            msg = json.loads(message)
        except Exception:
            return
        kind = msg.get("type")
        if kind == "task":
            self._ingest_task(msg)
        elif kind == "combine":
            with self._lock:
                self._stats["combines"] += 1
            self._push_stats_table()
        elif kind == "profile":
            self._ingest_profile(msg)

    def _ingest_task(self, msg: dict[str, Any]) -> None:
        self._tasks.update([_wire.task_row(msg)])
        phase = msg.get("phase", "")
        worker = msg.get("worker") or ""
        with self._lock:
            if phase in self._stats:
                self._stats[phase] += 1
            if phase == "started":
                self._stats["inflight"] += 1
            elif phase in ("finished", "errored"):
                self._stats["inflight"] = max(0, self._stats["inflight"] - 1)
            if phase == "errored":
                self._last_error = {
                    "key": msg.get("key"),
                    "worker": worker,
                    "message": msg.get("error", ""),
                }
            # per-worker progress for the bars. SUBMITTED is driver-side (worker=""), so only the
            # worker-side phases (started/finished/errored) populate a worker row. We keep a per-task
            # record (keyed by task key) so the UI can render one hoverable cell PER TASK, not just an
            # aggregate bar — a started task's record is completed in place when it finishes/errors.
            if worker and phase in ("started", "finished", "errored"):
                w = self._workers.setdefault(
                    worker,
                    {"worker": worker, "started": 0, "finished": 0, "errored": 0, "inflight": 0, "tasks": {}},
                )
                w[phase] += 1
                if phase == "started":
                    w["inflight"] += 1
                else:
                    w["inflight"] = max(0, w["inflight"] - 1)
                key = msg.get("key")
                rec = w["tasks"].get(key)
                if rec is None:  # STARTED normally creates it; tolerate an out-of-order finish/error
                    rec = {
                        "key": key,
                        "partition": "",
                        "n_entries": 0,
                        "state": "started",
                        "t_start": 0.0,
                        "t_end": None,
                        "error": "",
                    }
                    w["tasks"][key] = rec
                if phase == "started":
                    rec["partition"] = msg.get("partition", "")
                    rec["n_entries"] = msg.get("n_entries", 0)
                    rec["t_start"] = msg.get("t", 0.0)
                else:
                    rec["state"] = phase
                    rec["t_end"] = msg.get("t", 0.0)
                    if phase == "errored":
                        rec["error"] = msg.get("error", "")
        self._push_stats_table()

    def _ingest_profile(self, msg: dict[str, Any]) -> None:
        try:
            tree = _sampler.tree_from_bytes(base64.b64decode(msg["tree_b64"]))
        except Exception:
            return  # a malformed payload must never break the server
        with self._lock:
            _sampler.merge_into(self._profile_tree, tree)
            self._profile_samples += int(tree.get("count", 0))

    def _push_stats_table(self) -> None:
        with self._lock:
            row = {"metric": "run", **self._stats}
        self._stats_table.update([row])

    # ---- programmatic read (any thread) ---------------------------------

    def flamegraph(self) -> dict[str, Any]:
        """The merged d3-flame-graph tree (``{name, value, children}``) across all flushes/workers."""
        with self._lock:
            return _sampler.flamegraph(self._profile_tree)

    def flamegraph_json(self) -> bytes:
        return json.dumps(self.flamegraph()).encode("utf-8")

    def progress(self) -> dict[str, Any]:
        """Overall + per-worker progress for the bar chart. ``total`` is the submitted count; the
        ``overall`` segments (finished/errored/inflight) plus the implicit pending remainder tile it.
        ``workers`` is one row per worker that has run a task, sorted by id (deterministic order); each
        carries its aggregate counts plus ``tasks`` — one record per task (sorted by start time, then
        key) so the UI can render a hoverable cell per task. Records are deep-copied under the lock so
        JSON encoding (outside the lock) never races a concurrent task update."""
        with self._lock:
            overall = {k: self._stats[k] for k in ("submitted", "started", "finished", "errored", "inflight")}
            workers = []
            for name in sorted(self._workers):
                wd = self._workers[name]
                tasks = sorted(
                    (dict(r) for r in wd["tasks"].values()), key=lambda r: (r["t_start"], r["key"])
                )
                workers.append(
                    {k: wd[k] for k in ("worker", "started", "finished", "errored", "inflight")}
                    | {"tasks": tasks}
                )
        return {"total": overall["submitted"], "overall": overall, "workers": workers}

    def progress_json(self) -> bytes:
        return json.dumps(self.progress()).encode("utf-8")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "stats": dict(self._stats),
                "last_error": self._last_error,
                "profile_samples": self._profile_samples,
                "url": self.url,
            }
