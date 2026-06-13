"""The execution-layer contract (plan M7) — PROVISIONAL until exercised by a real adapter.

graphed-core owns the *contract*; the reference executors (a thread pool AND a process pool) live in
``graphed-exec-local``. This module is data-only — no awkward/numpy/backend imports — so the contract
stays a stable, minimal seam. A `Plan` is reduced to a single result by an `Executor`:

- work is a stream of `Task`s (fixed `tasks`, or pulled adaptively from `next_tasks`);
- each task runs `process(partition, resources)` on a worker and returns a **partial**;
- partials are combined by an **associative** `combine` via tree reduction;
- `open_once` gives file-locality (a uri opened once per worker);
- `StopCondition` ends the run early (target events / wall-clock / error budget / …);
- **error-propagation obligation**: a worker failure (process OR thread) must reach the driver
  *intact and picklable* — in particular a `graphed_debug.StageError` must NOT degrade to an opaque
  string, so the driver can render the user-source traceback (plan A.3 #8).
"""

from __future__ import annotations

import contextlib
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, Protocol, TypeVar, runtime_checkable

R = TypeVar("R")  # a partial result (e.g. a histogram array)


@dataclass(frozen=True)
class Partition:
    """A unit of input work (plan glossary): a uri + tree + half-open entry range.

    A **blind** partition (M10) defers entry-range resolution to read time: it records
    ``(step, n_steps)`` instead of an entry range, and the reader calls :meth:`resolve` against the
    file's actual entry count when the partition is read. Construct with :meth:`Partition.blind` —
    this replaces the old host-reader convention of smuggling the step through a negative
    ``entry_stop``, which any unaware consumer silently misread."""

    uri: str
    tree: str = ""
    entry_start: int = 0
    entry_stop: int = 0
    blind_step: int | None = None
    blind_n_steps: int | None = None

    @classmethod
    def blind(cls, uri: str, tree: str, step: int, n_steps: int) -> Partition:
        """A partition describing step ``step`` of ``n_steps`` over a file NOT opened yet."""
        if n_steps < 1 or not 0 <= step < n_steps:
            raise ValueError(f"blind partition needs 0 <= step < n_steps, got {step}/{n_steps}")
        return cls(uri, tree, 0, 0, blind_step=step, blind_n_steps=n_steps)

    @property
    def is_blind(self) -> bool:
        return self.blind_step is not None

    def resolve(self, num_entries: int) -> Partition:
        """Resolve a blind partition against the file's actual entry count (every entry is read
        exactly once across the file's n_steps partitions). A non-blind partition returns itself."""
        if self.blind_step is None or self.blind_n_steps is None:
            return self
        start = (self.blind_step * num_entries) // self.blind_n_steps
        stop = ((self.blind_step + 1) * num_entries) // self.blind_n_steps
        return Partition(self.uri, self.tree, start, stop)

    @property
    def n_entries(self) -> int:
        return max(0, self.entry_stop - self.entry_start)


@dataclass(frozen=True)
class Task:
    """One schedulable unit: a partition plus a deterministic ordering ``key`` (fixes the reduction
    tree shape so a fixed partition set reduces bit-for-bit regardless of completion order)."""

    key: int
    partition: Partition


@runtime_checkable
class WorkerResources(Protocol):
    """Per-worker resources. ``open_once`` returns a cached handle so a uri is opened exactly once
    per worker across that worker's chunks (file-locality directive)."""

    def open_once(self, uri: str, opener: Callable[[str], object]) -> object: ...


class StopReason(StrEnum):
    EXHAUSTED = "exhausted"  # all data processed (the normal end)
    TARGET_EVENTS = "target_events"
    PRECISION = "precision"
    WALL_CLOCK = "wall_clock"
    ERROR_BUDGET = "error_budget"


@dataclass
class ExecContext:
    """Mutable run state handed to ``next_tasks`` and stopping conditions (adaptive reshaping)."""

    n_done: int = 0
    events_done: int = 0
    errors: int = 0
    elapsed_s: float = 0.0
    last_durations: dict[int, float] = field(default_factory=dict)  # task key -> seconds observed


@dataclass(frozen=True)
class StopCondition:
    """Declarative stopping conditions; the first satisfied one ends submission."""

    target_events: int | None = None
    max_wall_s: float | None = None
    max_errors: int | None = None

    def reason(self, ctx: ExecContext) -> StopReason | None:
        if self.target_events is not None and ctx.events_done >= self.target_events:
            return StopReason.TARGET_EVENTS
        if self.max_wall_s is not None and ctx.elapsed_s >= self.max_wall_s:
            return StopReason.WALL_CLOCK
        if self.max_errors is not None and ctx.errors > self.max_errors:
            return StopReason.ERROR_BUDGET
        return None


@dataclass
class Plan(Generic[R]):
    """A minimal, PROVISIONAL execution plan (the real serializable Plan is M8)."""

    process: Callable[[Partition, WorkerResources], R]  # picklable; runs analysis on a chunk
    combine: Callable[[R, R], R]  # associative (and commutative) reducer
    empty: Callable[[], R]  # identity, for zero partitions / an empty fold
    tasks: Sequence[Task] = ()  # a fixed partition set -> a deterministic reduction tree
    next_tasks: Callable[[ExecContext], Iterable[Task] | None] | None = None  # adaptive hook (DONE=None)
    stop: StopCondition | None = None
    open_once: bool = False


@dataclass(frozen=True)
class ExecResult(Generic[R]):
    value: R
    n_partitions: int
    n_combines: int
    stopped: StopReason | None = None


@runtime_checkable
class Executor(Protocol):
    """Run a `Plan` to a single reduced result. Reference impls live in graphed-exec-local
    (thread-pool and process-pool). Implementations MUST surface a worker failure to the driver
    intact (picklable) — never as an opaque string."""

    def run(self, plan: Plan[R]) -> ExecResult[R]: ...


# ---- M37: the passive live-dashboard seam (data-only; no web, no profiler dep) --------------


class TaskPhase(StrEnum):
    SUBMITTED = "submitted"
    STARTED = "started"
    FINISHED = "finished"
    ERRORED = "errored"


@dataclass(frozen=True)
class TaskEvent:
    """A passive, display-only record of one task's lifecycle transition (M37 dashboard seam).

    Carries no un-picklable objects — it crosses a process boundary from a ``ProcessExecutor`` worker
    back to the driver. ``error`` is a pre-rendered summary string, never an exception object;
    ``partition`` is a human label. Per task the contract is exactly one ``SUBMITTED``, then one
    ``STARTED``, then exactly one of ``FINISHED`` | ``ERRORED``."""

    phase: TaskPhase
    key: int
    worker: str
    t: float
    partition: str = ""
    n_entries: int = 0
    bytes_read: int | None = None
    error: str | None = None


@runtime_checkable
class WorkerProfiler(Protocol):
    """A per-worker statistical sampler (M37). graphed-debug supplies the implementation
    (pyinstrument-backed); graphed-exec-local *drives* it without importing any profiler.
    ``flush``/``stop`` return a serialized sample tree (bytes) the driver merges, or ``None``."""

    def start(self) -> None: ...
    def flush(self) -> bytes | None: ...
    def stop(self) -> bytes | None: ...


@runtime_checkable
class Monitor(Protocol):
    """A passive observer of a run (M37). An executor *emits* through it; it MUST NOT influence task
    order, the reduction tree, or results — the determinism gate is green attached-or-not. A monitor
    that raises is swallowed by the emitting executor (see :func:`emit_task`). ``worker_profiler_factory``
    returns a *picklable* zero-arg factory shipped to workers (``None`` ⇒ no sampling)."""

    def on_task(self, event: TaskEvent) -> None: ...
    def on_profile(self, worker: str, payload: bytes) -> None: ...
    def on_combine(self, leaves_done: int) -> None: ...
    def worker_profiler_factory(self) -> Callable[[], WorkerProfiler] | None: ...


def emit_task(monitor: Monitor | None, event: TaskEvent) -> None:
    """Best-effort task emission: a ``None`` monitor is a no-op, and a monitor that raises must never
    break a run (M37 passivity)."""
    if monitor is None:
        return
    with contextlib.suppress(Exception):
        monitor.on_task(event)


def partition_label(partition: Partition) -> str:
    """A short human label for a partition (dashboard display only)."""
    return f"{partition.uri}:{partition.tree}:{partition.entry_start}-{partition.entry_stop}"


def _close_handle(handle: object) -> None:
    close = getattr(handle, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):  # a handle's own close must not break the run
            close()


class LocalResources:
    """Reference :class:`WorkerResources`: ``open_once(uri, opener)`` opens a uri at most once and
    reuses the handle for that runner/worker's later chunks. The set of simultaneously-open
    handles is **bounded** to ``max_open`` — when a new open would exceed it, the
    least-recently-used handle is closed and dropped (a uri reopened after eviction is opened
    again). ``close()`` releases every handle. The bound matters for a long-lived worker (a
    persistent process pool over many files): without it, ``open_once`` would accumulate every
    file ever opened for the worker's whole lifetime."""

    def __init__(self, max_open: int = 128) -> None:
        self._handles: OrderedDict[str, object] = OrderedDict()
        self._max_open = max_open
        self.open_count = 0  # real opens performed (test/diagnostic introspection)

    def open_once(self, uri: str, opener: Callable[[str], object]) -> object:
        if uri in self._handles:
            self._handles.move_to_end(uri)  # mark most-recently-used
            return self._handles[uri]
        handle = opener(uri)
        self.open_count += 1
        self._handles[uri] = handle
        while len(self._handles) > self._max_open:
            _, evicted = self._handles.popitem(last=False)  # close the least-recently-used
            _close_handle(evicted)
        return handle

    def close(self) -> None:
        for handle in self._handles.values():
            _close_handle(handle)
        self._handles.clear()


class SequentialRunner:
    """The dependency-free reference :class:`Executor` of the ``Plan`` contract: runs a plan's
    tasks **in key order**, in-process, with no worker pool. It lives beside the contract it
    executes so every layer — the frontend's deferred writers, histogram aggregation,
    preservation, the benchmarks — can run a ``Plan`` without depending on the executor package
    (which the frontend may not import). It is the canonical baseline any real executor
    (graphed-exec-local's thread/process pools) must match bit-for-bit.

    An optional :class:`Monitor` observes the run (M37). It is purely passive: emission is
    best-effort and a misbehaving monitor cannot change the result."""

    def __init__(self, monitor: Monitor | None = None) -> None:
        self._monitor = monitor

    def run(self, plan: Plan[R]) -> ExecResult[R]:
        resources = LocalResources()
        monitor = self._monitor
        try:
            value = plan.empty()
            n = 0
            ordered = sorted(plan.tasks, key=lambda t: t.key)
            for task in ordered:
                emit_task(monitor, self._event(TaskPhase.SUBMITTED, task))
            for task in ordered:
                emit_task(monitor, self._event(TaskPhase.STARTED, task))
                try:
                    partial = plan.process(task.partition, resources)
                except Exception as exc:
                    emit_task(
                        monitor, self._event(TaskPhase.ERRORED, task, error=f"{type(exc).__name__}: {exc}")
                    )
                    raise
                value = plan.combine(value, partial)
                emit_task(monitor, self._event(TaskPhase.FINISHED, task))
                n += 1
            return ExecResult(value=value, n_partitions=n, n_combines=max(0, n - 1))
        finally:
            resources.close()  # release file handles deterministically at end of run

    @staticmethod
    def _event(phase: TaskPhase, task: Task, *, error: str | None = None) -> TaskEvent:
        return TaskEvent(
            phase=phase,
            key=task.key,
            worker="seq",
            t=time.perf_counter(),
            partition=partition_label(task.partition),
            n_entries=task.partition.n_entries,
            error=error,
        )
