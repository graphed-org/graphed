"""The M37 statistical sampler: a thin shim over **pyinstrument** implementing the
``graphed_core.execution.WorkerProfiler`` protocol, plus a helper that flattens a pyinstrument
session into tabular rows for the dashboard's Perspective ``profile`` table.

pyinstrument is an **optional** dependency (the ``dashboard`` extra), imported lazily so
``graphed_debug`` imports cleanly without it. ``graphed-exec-local`` never imports any of this — it
only drives the abstract ``WorkerProfiler``.

Flush semantics: pyinstrument has no mid-run snapshot, so :meth:`PyinstrumentWorkerProfiler.flush`
stops the current profiler, serialises the elapsed session, and starts a fresh one; the dashboard
accumulates successive intervals by appending their rows.
"""

from __future__ import annotations

import importlib.util
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyinstrument.session import Session


def pyinstrument_available() -> bool:
    return importlib.util.find_spec("pyinstrument") is not None


class PyinstrumentWorkerProfiler:
    """A per-worker sampler. Satisfies ``WorkerProfiler`` (``start``/``flush``/``stop`` -> bytes).
    ``flush``/``stop`` return a pyinstrument session serialised as UTF-8 JSON bytes, or ``None``."""

    def __init__(self, interval: float = 0.001) -> None:
        self._interval = interval
        self._profiler: Any = None

    def start(self) -> None:
        from pyinstrument import Profiler  # noqa: PLC0415 (optional dep, imported lazily)

        self._profiler = Profiler(interval=self._interval)
        self._profiler.start()

    def _take(self, *, restart: bool) -> bytes | None:
        if self._profiler is None:
            return None
        session = self._profiler.stop()
        if restart:
            self.start()
        else:
            self._profiler = None
        if session is None or session.sample_count == 0:
            return None
        return json.dumps(session.to_json()).encode("utf-8")

    def flush(self) -> bytes | None:
        return self._take(restart=True)

    def stop(self) -> bytes | None:
        return self._take(restart=False)


def session_from_bytes(payload: bytes) -> Session:
    from pyinstrument.session import Session  # noqa: PLC0415 (optional dep, imported lazily)

    return Session.from_json(json.loads(payload.decode("utf-8")))


def _location(frame: Any) -> str:
    pos = getattr(frame, "code_position_short", None)
    if callable(pos):
        try:
            pos = pos()
        except Exception:
            pos = None
    return str(pos) if pos else ""


def profile_rows(session: Session, worker: str) -> list[dict[str, Any]]:
    """Flatten a pyinstrument session into one row per frame: function, location, worker, and
    self/total microseconds. The Perspective ``profile`` table aggregates these (group by function,
    sum self_us) in the browser — the tabular analogue of a flamegraph."""
    root = session.root_frame()
    if root is None:
        return []
    rows: list[dict[str, Any]] = []

    def walk(frame: Any) -> None:
        child_total = sum(c.time for c in frame.children)
        rows.append(
            {
                "function": frame.function or "?",
                "location": _location(frame),
                "worker": worker,
                "self_us": max(0, round((frame.time - child_total) * 1_000_000)),
                "total_us": round(frame.time * 1_000_000),
            }
        )
        for child in frame.children:
            walk(child)

    walk(root)
    return rows


def make_worker_profiler() -> PyinstrumentWorkerProfiler:
    """A module-level, **picklable** factory shipped to ``ProcessExecutor`` workers (M37)."""
    return PyinstrumentWorkerProfiler()
