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


class LocalResources:
    """Reference :class:`WorkerResources`: opens each uri at most once per runner (``open_once``)."""

    def __init__(self) -> None:
        self._handles: dict[str, object] = {}

    def open_once(self, uri: str, opener: Callable[[str], object]) -> object:
        if uri not in self._handles:
            self._handles[uri] = opener(uri)
        return self._handles[uri]


class SequentialRunner:
    """The dependency-free reference :class:`Executor` of the ``Plan`` contract: runs a plan's
    tasks **in key order**, in-process, with no worker pool. It lives beside the contract it
    executes so every layer — the frontend's deferred writers, histogram aggregation,
    preservation, the benchmarks — can run a ``Plan`` without depending on the executor package
    (which the frontend may not import). It is the canonical baseline any real executor
    (graphed-exec-local's thread/process pools) must match bit-for-bit."""

    def run(self, plan: Plan[R]) -> ExecResult[R]:
        resources = LocalResources()
        value = plan.empty()
        n = 0
        for task in sorted(plan.tasks, key=lambda t: t.key):
            value = plan.combine(value, plan.process(task.partition, resources))
            n += 1
        return ExecResult(value=value, n_partitions=n, n_combines=max(0, n - 1))
