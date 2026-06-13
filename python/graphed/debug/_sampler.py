"""The M37 statistical sampler: a thin shim over **pyinstrument** (an existing, lean sampling
profiler) implementing the ``graphed_core.execution.WorkerProfiler`` protocol, plus helpers to
combine per-worker sessions and adapt pyinstrument's frame tree to the d3-flame-graph JSON shape.

pyinstrument is an **optional** dependency (the ``dashboard`` extra). This module imports it lazily
so ``graphed_debug`` imports cleanly without it; :func:`pyinstrument_available` reports presence.
``graphed-exec-local`` never imports any of this — it only drives the abstract ``WorkerProfiler``.

Flush semantics: pyinstrument has no mid-run snapshot, so :meth:`PyinstrumentWorkerProfiler.flush`
stops the current profiler, renders the elapsed session, and starts a fresh one; the driver
accumulates successive sessions with :func:`combine_sessions`. Each ``flush`` therefore ships an
*interval* of samples, and the driver's running combine is the cumulative profile.
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
    ``flush``/``stop`` return a pyinstrument session serialized as UTF-8 JSON bytes, or ``None`` when
    nothing was sampled."""

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


def combine_sessions(a: Session, b: Session) -> Session:
    from pyinstrument.session import Session  # noqa: PLC0415 (optional dep, imported lazily)

    return Session.combine(a, b)


def _frame_label(frame: Any) -> str:
    pos = getattr(frame, "code_position_short", None)
    if callable(pos):  # a method in some pyinstrument versions, a string-ish property in others
        try:
            pos = pos()
        except Exception:
            pos = None
    name = frame.function or "?"
    return f"{name}  ({pos})" if pos else name


def _frame_to_tree(frame: Any) -> dict[str, Any]:
    # pyinstrument's ``frame.time`` is inclusive, so a parent's value always covers its children's
    # (the invariant d3-flame-graph needs). Microseconds keep integer widths readable.
    return {
        "name": _frame_label(frame),
        "value": max(1, round(frame.time * 1_000_000)),
        "children": [_frame_to_tree(child) for child in frame.children],
    }


def flamegraph_tree(session: Session) -> dict[str, Any]:
    """Adapt a pyinstrument session to a d3-flame-graph ``{name, value, children}`` tree."""
    root = session.root_frame()
    if root is None:
        return {"name": "(no samples)", "value": 0, "children": []}
    return _frame_to_tree(root)


def make_worker_profiler() -> PyinstrumentWorkerProfiler:
    """A module-level, **picklable** factory shipped to ``ProcessExecutor`` workers (M37). Returns a
    fresh per-worker sampler. Picklable-by-reference because it is a plain module function."""
    return PyinstrumentWorkerProfiler()
