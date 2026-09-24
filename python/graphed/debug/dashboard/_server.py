"""The dashboard **server** (M37): a `perspective` ``Server`` hosting the live ``stats`` table (and a
``tasks`` table fed by the same event stream) over a Tornado app, plus two derived JSON views the
browser polls — a merged profile **flamegraph** at ``/api/flamegraph.json`` and overall + per-worker
**progress** at ``/api/progress.json`` (the dask-style progress bars). Browsers connect a
``<perspective-viewer>`` to the ``/websocket`` endpoint; executors push events to the ``/ingest``
websocket (see :class:`graphed.debug.dashboard.NetworkMonitor`). It runs its own asyncio/Tornado
IOLoop in a daemon thread, so it is decoupled from the executor — the same server serves a local *or*
a remote run.

With ``control=True`` it also relays run control: ``POST /api/control`` with JSON ``{"cmd": ...}``
writes the command down every ``/ingest`` connection whose monitor said ``hello`` with
``control: true`` (a :class:`NetworkMonitor` built with a ``RunControl``), which applies it to the run.

perspective/tornado are imported lazily (the ``dashboard`` extra), so ``import graphed.debug`` works
without them; :meth:`DashboardServer.start` raises a clear error if they are missing.
"""

from __future__ import annotations

import base64
import contextlib
import json
import threading
from pathlib import Path
from typing import Any

from graphed.core.execution import RunState

from .. import _sampler
from . import _wire

_STATIC = Path(__file__).parent / "static"
_SUBMITTED, _STARTED, _TERMINAL = 0, 1, 2  # phase classes, in lifecycle order
_PHASE_CLASS = {"submitted": _SUBMITTED, "started": _STARTED, "finished": _TERMINAL, "errored": _TERMINAL}


class _KeyState:
    """One task key's ingest state: ``life`` is the set of phase classes its current lifecycle has
    seen; ``label`` is the ``(partition, n_entries)`` of its latest SUBMITTED, whatever lifecycle."""

    __slots__ = ("label", "life")

    def __init__(self) -> None:
        self.life: set[int] = set()
        self.label: tuple[str, int] | None = None


class DashboardServer:
    """Hosts the live Perspective tables and the ingest/viewer websockets. Start it, point one or
    more executors' :class:`NetworkMonitor` at :attr:`ingest_url`, and open :attr:`url` in a browser.
    Thread-safe: all Perspective table writes and websocket writes happen on the IOLoop thread;
    :meth:`snapshot` reads a plain-Python mirror under a lock, and :meth:`stop` closes the ingest
    connections through the loop."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, *, control: bool = False) -> None:
        self._host = host
        self._port = port
        self._control = control
        self._control_state: str | None = RunState.RUNNING.value if control else None
        self._live: set[Any] = set()  # open /ingest handlers (IOLoop thread only)
        self._listeners: set[Any] = set()  # those whose monitor sent a control hello; mirrored under _lock
        self._thread: threading.Thread | None = None
        self._loop: Any = None
        self._http: Any = None
        self._client: Any = None
        self._tasks: Any = None
        self._rows: dict[Any, dict[str, Any]] = {}  # this frame's tasks-table rows, merged per key
        self._stats_table: Any = None
        self._lock = threading.RLock()
        self._stats: dict[str, int] = dict.fromkeys(_wire.STATS_KEYS, 0)
        self._workers: dict[str, dict[str, Any]] = {}  # worker -> per-worker progress (for the bars)
        # ingest-only state (IOLoop thread): lean derivation and per-key lifecycles
        self._lean = False  # set once any ingest connection says hello with lean: true
        self._keys: dict[Any, _KeyState] = {}
        self._open: set[Any] = set()  # keys from SUBMITTED to their lifecycle's first terminal
        self._workers_seen: set[str] = set()  # names on STARTED and terminal events
        self._last_end: dict[tuple[int, str], float] = {}  # (connection, worker) -> previous t_end
        self._unlabelled: dict[Any, list[dict[str, Any]]] = {}  # key -> records awaiting a label
        self._n_connections = 0
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
            # the close frames leave inside this callback, so a monitor sees them and reconnects
            loop.add_callback(self._close_ingest)
            loop.add_callback(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._live.clear()
        with self._lock:
            self._listeners.clear()
        self._started = False

    def _close_ingest(self) -> None:
        for conn in list(self._live):
            conn.close()

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

                def open(self, *args: str, **kwargs: str) -> None:
                    owner._live.add(self)
                    with owner._lock:
                        owner._n_connections += 1
                        self.cid = owner._n_connections

                def on_close(self) -> None:
                    owner._live.discard(self)
                    with owner._lock:
                        owner._listeners.discard(self)

                def on_message(self, message: str | bytes) -> None:
                    owner._ingest(message if isinstance(message, str) else message.decode("utf-8"), self)

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

            class _Control(RequestHandler):  # type: ignore[misc]  # POST {"cmd": ...} -> the listening monitors
                def post(self) -> None:
                    # a cross-site form cannot send this type without a preflight nobody answers
                    if (
                        self.request.headers.get("Content-Type", "").partition(";")[0].strip()
                        != "application/json"
                    ):
                        self.send_error(415)
                        return
                    try:
                        cmd = json.loads(self.request.body)["cmd"]
                        state = _wire.CONTROL_STATES[cmd]
                    except (ValueError, LookupError, TypeError):
                        self.send_error(400)
                        return
                    self.finish({"cmd": cmd, "delivered": owner._send_control(cmd, state)})

            routes: list[Any] = [
                (r"/websocket", PerspectiveTornadoHandler, {"perspective_server": server}),
                (r"/ingest", _Ingest),
                (r"/api/flamegraph.json", _Flame),
                (r"/api/progress.json", _Progress),
                (r"/static/(.*)", StaticFileHandler, {"path": str(_STATIC)}),
                (r"/", _Index),
            ]
            if self._control:
                routes.append((r"/api/control", _Control))
            app = Application(routes)
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

    def _ingest(self, message: str, conn: Any = None) -> None:
        try:
            msg = json.loads(message)
        except Exception:
            return
        cid = getattr(conn, "cid", 0)
        # held across the frame so a reader waiting on a count also sees the rows written below
        with self._lock:
            for item in msg.get("items", ()) if msg.get("type") == "batch" else (msg,):
                self._ingest_one(item, conn, cid)
            self._write_rows()
        self._push_stats_table()

    def _write_rows(self) -> None:
        """One Perspective update per column set, never per row: an indexed update keeps the
        columns a row omits, so rows with different sets must not share one."""
        groups: dict[frozenset[str], list[dict[str, Any]]] = {}
        for r in self._rows.values():
            groups.setdefault(frozenset(r), []).append(r)
        self._rows = {}
        for rows in groups.values():
            self._tasks.update(rows)

    def _ingest_one(self, msg: dict[str, Any], conn: Any, cid: int) -> None:
        kind = msg.get("type")
        if kind == "hello":
            if msg.get("control") is True:
                with self._lock:
                    self._listeners.add(conn)
            if msg.get("lean") is True:
                self._lean = True
        elif kind == "task":
            self._ingest_task(msg, cid)
        elif kind == "combine":
            with self._lock:
                self._stats["combines"] += 1
        elif kind == "profile":
            self._ingest_profile(msg)

    def _ingest_task(self, msg: dict[str, Any], cid: int = 0) -> None:
        phase = msg.get("phase", "")
        cls = _PHASE_CLASS.get(phase)
        if cls is None:
            return
        key = msg.get("key")
        worker = msg.get("worker") or ""
        t = msg.get("t", 0.0)
        ks = self._keys.get(key)
        if ks is None:
            ks = self._keys[key] = _KeyState()
        if cls in ks.life:  # a repeated class is the only sign of a new run or a retry
            ks.life = set()
        late = bool(ks.life) and max(ks.life) > cls  # arrived after a later phase of its lifecycle
        derived = self._lean and cls == _TERMINAL and _STARTED not in ks.life
        opens = cls == _SUBMITTED and _TERMINAL not in ks.life
        ks.life.add(cls)
        if cls == _SUBMITTED:
            ks.label = (msg.get("partition", ""), msg.get("n_entries", 0))

        # an indexed update keeps the columns it omits: a late event writes only the label, and an
        # event with an empty label (a lean terminal) writes none
        row = _wire.task_row(msg)
        if late:
            row = {"key": key, "partition": row["partition"], "n_entries": row["n_entries"]}
        if not row["partition"]:
            del row["partition"], row["n_entries"]
        if len(row) > 1:
            self._rows.setdefault(key, {}).update(row)

        with self._lock:
            self._stats[phase] += 1
            if derived:
                self._stats["started"] += 1
            if opens:
                self._open.add(key)
            elif cls == _TERMINAL:
                self._open.discard(key)
            if cls != _SUBMITTED and worker:
                self._workers_seen.add(worker)
            if self._lean:
                self._stats["inflight"] = min(len(self._workers_seen), len(self._open))
            elif cls == _STARTED:
                self._stats["inflight"] += 1
            elif cls == _TERMINAL:
                self._stats["inflight"] = max(0, self._stats["inflight"] - 1)
            if phase == "errored":
                self._last_error = {
                    "key": key,
                    "worker": worker,
                    "message": msg.get("error", ""),
                }
            if cls == _SUBMITTED:
                for rec in self._unlabelled.pop(key, ()):
                    rec["partition"], rec["n_entries"] = msg.get("partition", ""), msg.get("n_entries", 0)
            # per-worker progress for the bars. SUBMITTED is driver-side, so only the worker-side
            # phases populate a worker row. We keep a per-task record (keyed by task key) so the UI
            # can render one hoverable cell PER TASK, not just an aggregate bar — a started task's
            # record is completed in place when it finishes/errors.
            elif worker:
                self._ingest_worker(msg, ks, cls, worker, t, cid, derived)

    def _ingest_worker(
        self, msg: dict[str, Any], ks: _KeyState, cls: int, worker: str, t: float, cid: int, derived: bool
    ) -> None:
        w = self._workers.setdefault(
            worker,
            {"worker": worker, "started": 0, "finished": 0, "errored": 0, "inflight": 0, "tasks": {}},
        )
        phase = msg["phase"]
        w[phase] += 1
        if derived:
            w["started"] += 1
        if cls == _STARTED:
            w["inflight"] += 1
        else:
            w["inflight"] = max(0, w["inflight"] - 1)
        key = msg.get("key")
        rec = w["tasks"].get(key)
        if rec is None:  # STARTED normally creates it; tolerate an out-of-order or lean finish/error
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
        if cls == _STARTED and msg.get("partition"):
            rec["partition"], rec["n_entries"] = msg["partition"], msg.get("n_entries", 0)
        elif not rec["partition"]:
            if ks.label is not None:
                rec["partition"], rec["n_entries"] = ks.label
            else:
                self._unlabelled.setdefault(key, []).append(rec)
        if cls == _STARTED:
            rec["t_start"] = t
        else:
            if derived:  # a lean start: this worker's previous end on this connection, else its own t
                rec["t_start"] = self._last_end.get((cid, worker), t)
            rec["state"] = phase
            rec["t_end"] = t
            self._last_end[(cid, worker)] = t
            if phase == "errored":
                rec["error"] = msg.get("error", "")

    def _ingest_profile(self, msg: dict[str, Any]) -> None:
        try:
            tree = _sampler.tree_from_bytes(base64.b64decode(msg["tree_b64"]))
        except Exception:
            return  # a malformed payload must never break the server
        with self._lock:
            _sampler.merge_into(self._profile_tree, tree)
            self._profile_samples += int(tree.get("count", 0))

    def _send_control(self, cmd: str, state: str) -> int:
        """Write ``cmd`` to every listening monitor (IOLoop thread); return how many took it."""
        from tornado.websocket import WebSocketClosedError  # noqa: PLC0415

        text = json.dumps(_wire.control_message(cmd))
        with self._lock:
            self._control_state = state
            listeners = list(self._listeners)
        delivered = 0
        for conn in listeners:
            with contextlib.suppress(WebSocketClosedError):  # closed between hello and now
                conn.write_message(text)
                delivered += 1
        return delivered

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
            control = self._control_state
        return {"total": overall["submitted"], "overall": overall, "workers": workers, "control": control}

    def progress_json(self) -> bytes:
        return json.dumps(self.progress()).encode("utf-8")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "stats": dict(self._stats),
                "last_error": self._last_error,
                "profile_samples": self._profile_samples,
                "url": self.url,
                "control": self._control_state,
                "control_listeners": len(self._listeners),
                "ingest_connections": self._n_connections,
            }
