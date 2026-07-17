"""The M37 statistical sampler: an **off-thread** stack sampler implementing the
``graphed_core.execution.WorkerProfiler`` protocol.

Unlike a per-call profiler (pyinstrument hooks every Python call → ~3x overhead on call-heavy HEP
code, regardless of sample rate), this samples the worker's *task thread* from a **separate daemon
thread** via ``sys._current_frames()`` on a timer. The task thread is never hooked, so the
data-processing path pays ~nothing; the only cost is the sampler thread briefly taking the GIL to
read + fold a stack every ``interval`` (and array kernels release the GIL anyway). It is pure stdlib
(no third-party dependency) and needs no privileges — the technique is dask's ``distributed.profile``.

A sample folds the stack into a nested **count tree** (``{name, count, children}``) keyed by
``func;file;line``; counts are *inclusive* (every frame on the stack is incremented), so a parent's
count always covers its children's — exactly the d3-flame-graph invariant. ``flush`` serialises the
tree accumulated since the last flush (then resets); the dashboard server merges trees across flushes
and workers into one live flamegraph.
"""

from __future__ import annotations

import json
import sys
import threading
from typing import Any

_MAX_DEPTH = 200  # cap the folded stack depth (real stacks are well under this; bounds the work)


def _identifier(frame: Any) -> str:
    co = frame.f_code
    return f"{co.co_name};{co.co_filename};{co.co_firstlineno}"


def _display(frame: Any) -> str:
    co = frame.f_code
    short = co.co_filename.rsplit("/", 1)[-1]
    return f"{co.co_name}  ({short}:{frame.f_lineno})"


def _new_node(name: str = "all") -> dict[str, Any]:
    return {"name": name, "count": 0, "children": {}}


def _fold(leaf: Any, root: dict[str, Any]) -> None:
    """Fold one sampled stack (root→leaf) into the count tree, incrementing every frame's inclusive
    count. Iterative (no recursion on deep stacks)."""
    stack = []
    frame: Any = leaf
    depth = 0
    while frame is not None and depth < _MAX_DEPTH:
        stack.append(frame)
        frame = frame.f_back
        depth += 1
    stack.reverse()  # outermost frame first
    node = root
    node["count"] += 1
    for fr in stack:
        ident = _identifier(fr)
        child = node["children"].get(ident)
        if child is None:
            child = {"name": _display(fr), "count": 0, "children": {}}
            node["children"][ident] = child
        child["count"] += 1
        node = child


class StackSampler:
    """A ``WorkerProfiler`` that samples its starting thread's stack from a background daemon thread.

    ``start`` is called on the worker's task thread (the thread it then samples); ``flush``/``stop``
    return the accumulated count tree as JSON bytes, or ``None`` if nothing was sampled."""

    def __init__(self, interval: float = 0.01) -> None:
        self._interval = interval
        self._tid: int = -1  # the task thread's id; set on start(). -1 is no thread -> .get() misses.
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._tree = _new_node()

    def start(self) -> None:
        self._tid = threading.get_ident()  # the worker's task thread (the caller)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="graphed-profile-sampler", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        tid = self._tid
        while not self._stop.is_set():
            frame = sys._current_frames().get(tid)  # the task thread's CURRENT stack; we don't own it
            if frame is not None:
                with self._lock:
                    _fold(frame, self._tree)
                del frame
            self._stop.wait(self._interval)

    def flush(self) -> bytes | None:
        with self._lock:
            tree = self._tree
            self._tree = _new_node()
        if tree["count"] == 0:
            return None
        return json.dumps(tree).encode("utf-8")

    def stop(self) -> bytes | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return self.flush()


def sampler_available() -> bool:
    """The sampler is pure stdlib — always available (kept for symmetry with the old optional dep)."""
    return True


def tree_from_bytes(payload: bytes) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(payload.decode("utf-8"))
    return result


def merge_into(acc: dict[str, Any], tree: dict[str, Any]) -> None:
    """Merge ``tree`` into ``acc`` (summing inclusive counts per identifier). Iterative."""
    work = [(acc, tree)]
    while work:
        na, nb = work.pop()
        na["count"] += nb["count"]
        a_children = na["children"]
        for ident, b_child in nb["children"].items():
            a_child = a_children.get(ident)
            if a_child is None:
                a_children[ident] = b_child
            else:
                work.append((a_child, b_child))


def flamegraph(tree: dict[str, Any]) -> dict[str, Any]:
    """Convert the count tree to a d3-flame-graph ``{name, value, children}`` tree (value = inclusive
    sample count, so a parent always covers its children)."""

    def conv(node: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": node["name"],
            "value": node["count"],
            "children": [conv(c) for c in node["children"].values()],
        }

    return {
        "name": "all",
        "value": tree.get("count", 0),
        "children": [conv(c) for c in tree.get("children", {}).values()],
    }


def make_worker_profiler() -> StackSampler:
    """A module-level, **picklable** factory shipped to ``ProcessExecutor`` workers (M37)."""
    return StackSampler()
