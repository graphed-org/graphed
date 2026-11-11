"""The execution-layer contract (plan M7) — PROVISIONAL until exercised by a real adapter.

graphed-core owns the *contract*; the reference executors (a thread pool AND a process pool) live
in ``graphed-exec-local``. This module is data-only — no awkward/numpy/backend imports — so the
contract stays a stable, minimal seam. A `Plan` is reduced to a single result by an `Executor`:

- work is a stream of `Task`s (fixed `tasks`, or pulled adaptively from `next_tasks`);
- each task runs `process(partition, resources)` on a worker and returns a **partial**;
- partials are combined by an **associative** `combine` via tree reduction;
- `open_once` gives file-locality (a uri opened once per worker);
- `StopCondition` ends the run early (target events / wall-clock / error budget / …);
- **error-propagation obligation**: a worker failure (process OR thread) must reach the driver
  *intact and picklable* — in particular a `graphed.debug.StageError` must NOT degrade to an opaque
  string, so the driver can render the user-source traceback (plan A.3 #8).
"""

from __future__ import annotations

import contextlib
import functools
import os
import pickle
import threading
import time
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import PureWindowsPath
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

from .graphed_core import version
from .services import Bindable, Resolvable, UnboundServiceError
from .worker_warnings import on_value

if TYPE_CHECKING:
    from .plan import DurablePlanV2
    from .services import ServiceSpec

R = TypeVar("R")  # a partial result (e.g. a histogram array)
Block = TypeVar("Block")  # a backend-native partition of rows (opaque to the engine)
# M39: a phantom second parameter of ShuffleBackend (used by no exchange method). Covariant, as an
# unused Protocol type parameter must be.
Index_co = TypeVar("Index_co", covariant=True)
# M40 (E3): the row-index handle the join half actually uses. INVARIANT — ``match_indices``
# produces it (return position) and ``take`` consumes it (parameter position), so it is neither
# purely covariant nor contravariant. Lives on :class:`JoinBackend`, not on ``ShuffleBackend``.
Index = TypeVar("Index")


@runtime_checkable
class ShuffleBackend(Protocol[Block, Index_co]):
    """The vectorized data-movement primitives the generic exchange/repartition engine calls.

    Like every contract in this module it is **pure** — graphed-core must never depend on a backend
    array library, so the engine deals only in opaque ``Block``/``Index`` handles; the real
    primitives live in the array-backend packages.

    ``identity`` is the backend's versioned shuffle-format token (folded into the V2 task ids so
    two conforming backends never journal the same id for different content). The
    ``@runtime_checkable`` check is **import hygiene only** — it witnesses method NAMES, never the
    routing rule; the pinned SHA-256 route ``partition`` implements is witnessed by the backends'
    golden-vector suites. The join half is ``match_indices``/``take``/``merge_records``.
    """

    identity: str

    def partition(
        self, block: Block, key_field: str, parts: int, *, salt: int = 0, boundaries: object = None
    ) -> tuple[Block, ...]:
        """Route each row of ``block`` to one of ``parts`` sub-blocks.

        Routing is by the pinned §4 hash of its ``key_field`` (``range`` uses ``boundaries``).
        Row-conserving; deterministic; §4/B2 contract.
        """
        ...

    def concat(self, blocks: Sequence[Block]) -> Block:
        """Vertically concatenate blocks in order (the deterministic ascending-src_pid merge)."""
        ...

    def slice_rows(self, block: Block, start: int, stop: int) -> Block:
        """Return a half-open row slice at event boundaries (keeps each row's structure intact)."""
        ...

    def estimated_bytes(self, block_or_form: object) -> int:
        """Measured payload bytes (NOT entry count — jagged bytes are not a function of #rows)."""
        ...

    def to_wire(self, block: Block) -> bytes:
        """Deterministic serialization to the wire (content-addressing hashes these bytes)."""
        ...

    def from_wire(self, data: bytes) -> Block:
        """Inverse of :meth:`to_wire`."""
        ...


@runtime_checkable
class JoinBackend(ShuffleBackend[Block, Index], Protocol[Block, Index]):
    """The M39 exchange half (inherited) **plus** the M40 relational join half (plan §3.0/§3.3).

    The generic radix-hash join engine needs a backend that is BOTH a :class:`ShuffleBackend`
    (move/route rows) AND supplies these relational primitives; the real array backends satisfy it,
    while an exchange-only M39 backend stays a plain :class:`ShuffleBackend` (its frozen
    ``isinstance`` contract is unchanged — the join half is additive, not a widening of
    ``ShuffleBackend``).
    """

    def match_indices(
        self, build: Block, probe: Block, *, on: Sequence[str], how: str = "inner"
    ) -> tuple[Index, Index]:
        """Relationally match two co-partitioned blocks on ``on``.

        Return aligned ``(build_idx, probe_idx)`` **with duplication** — a probe row with k build
        matches yields k aligned pairs (SQL/pandas semantics, NOT list-of-matches).
        ``how=left``/``outer`` emit a ``-1`` sentinel where a side is missing (→ :meth:`take`
        yields a null/option row).
        """
        ...

    def take(self, block: Block, index: Index) -> Block:
        """Gather rows of ``block`` by ``index``.

        A ``-1`` entry is a **miss** → a null/option row, **never** the last row (an
        ``np.take``-style gather is a bug — plan M40 trap #1).
        """
        ...

    def merge_records(self, left: Block, right: Block, *, on: Sequence[str]) -> Block:
        """Flat relational record-merge of two aligned taken blocks.

        The union of both sides' fields minus the duplicated ``on`` key, one flat record per
        aligned row (plan M40 §3.3).
        """
        ...


@dataclass(frozen=True)
class Partition:
    """A unit of input work (plan glossary): a uri + tree + half-open entry range.

    A **blind** partition defers entry-range resolution to read time: it records
    ``(step, n_steps)`` instead of an entry range, and the reader calls :meth:`resolve` against the
    file's actual entry count when the partition is read. Construct with :meth:`Partition.blind` —
    this replaces the old host-reader convention of smuggling the step through a negative
    ``entry_stop``, which any unaware consumer silently misread.
    """

    uri: str
    tree: str = ""
    entry_start: int = 0
    entry_stop: int = 0
    blind_step: int | None = None
    blind_n_steps: int | None = None
    #: a blind partition's cap on its entries; only its reader knows how many the file has
    blind_limit: int | None = None

    @classmethod
    def blind(cls, uri: str, tree: str, step: int, n_steps: int) -> Partition:
        """Return a partition for step ``step`` of ``n_steps`` over a file NOT opened yet."""
        if n_steps < 1 or not 0 <= step < n_steps:
            raise ValueError(f"blind partition needs 0 <= step < n_steps, got {step}/{n_steps}")
        return cls(uri, tree, 0, 0, blind_step=step, blind_n_steps=n_steps)

    @property
    def is_blind(self) -> bool:
        """Whether its reader may move its range; a capped one is exact, but unresolved too."""
        return self.blind_step is not None and self.blind_limit is None

    def resolve(self, num_entries: int) -> Partition:
        """Resolve a blind partition against the file's actual entry count.

        Every entry is read exactly once across the file's n_steps partitions. A partition with no
        blind step returns itself.
        """
        if self.blind_step is None or self.blind_n_steps is None:
            return self
        start = (self.blind_step * num_entries) // self.blind_n_steps
        stop = ((self.blind_step + 1) * num_entries) // self.blind_n_steps
        if self.blind_limit is not None:
            stop = min(stop, start + self.blind_limit)
        return Partition(self.uri, self.tree, start, stop)

    def first(self, n: int) -> Partition:
        """Return the first ``n`` entries of its unmoved range, or all of them if fewer."""
        if self.blind_step is not None:
            limit = n if self.blind_limit is None else min(n, self.blind_limit)
            return replace(self, blind_limit=limit)
        return replace(self, entry_stop=min(self.entry_stop, self.entry_start + n))

    @property
    def n_entries(self) -> int:
        """The number of entries this partition covers."""
        return max(0, self.entry_stop - self.entry_start)


@dataclass(frozen=True)
class Task:
    """One schedulable unit.

    A partition plus a deterministic ordering ``key`` (fixes the reduction tree shape so a fixed
    partition set reduces bit-for-bit regardless of completion order).
    """

    key: int
    partition: Partition


@runtime_checkable
class WorkerResources(Protocol):
    """Per-worker resources.

    ``open_once`` returns a cached handle so a uri is opened exactly once per worker across that
    worker's chunks (file-locality directive).
    """

    def open_once(self, uri: str, opener: Callable[[str], object]) -> object:
        """Open ``uri`` with ``opener`` once, and reuse the handle afterwards."""
        ...


class StopReason(StrEnum):
    """Why a run stopped."""

    EXHAUSTED = "exhausted"  # all data processed (the normal end)
    TARGET_EVENTS = "target_events"
    PRECISION = "precision"
    WALL_CLOCK = "wall_clock"
    ERROR_BUDGET = "error_budget"
    CANCELLED = "cancelled"  # a RunControl cancel stopped the run with work left to start


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
        """Return why ``ctx`` should stop the run, or ``None`` to go on."""
        if self.target_events is not None and ctx.events_done >= self.target_events:
            return StopReason.TARGET_EVENTS
        if self.max_wall_s is not None and ctx.elapsed_s >= self.max_wall_s:
            return StopReason.WALL_CLOCK
        if self.max_errors is not None and ctx.errors > self.max_errors:
            return StopReason.ERROR_BUDGET
        return None


@dataclass(frozen=True)
class FailedChunk:
    """A chunk a run went on without; ``record`` is a checkpoint dead-letter descriptor."""

    partition: str
    error: str
    record: dict[str, Any]


class TooManyFailedChunksError(RuntimeError):
    """The run went past its failure budget, lost-worker reruns included; ``failed`` so far."""

    def __init__(self, failed: tuple[FailedChunk, ...]) -> None:
        """Hold ``failed``; ``args`` is what unpickling rebuilds from."""
        super().__init__(failed)
        self.failed = failed

    def __str__(self) -> str:
        """List the failures."""
        lines = "".join(f"\n  {f.partition}: {f.error}" for f in self.failed)
        return f"{counted(len(self.failed), 'chunk')} failed:{lines}"


def dead_letter_descriptor(
    task_id: str, partition: Partition, exc: BaseException
) -> dict[str, Any]:
    """Build a reproducible, JSON-serializable dead-letter record from a failure."""
    desc: dict[str, Any] = {
        "task_id": task_id,
        "uri": partition.uri,
        "tree": partition.tree,
        "entry_start": partition.entry_start,
        "entry_stop": partition.entry_stop,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    # StageError carries structured user-source provenance (duck-typed so graphed-debug stays a
    # soft dependency: any error exposing these fields is harvested with full provenance).
    frames = getattr(exc, "frames", None)
    op = getattr(exc, "op", None)
    if frames is not None and op is not None:
        top = frames[0] if frames else None
        desc["stage_error"] = {
            "op": op,
            "cause_type": getattr(exc, "cause_type", ""),
            "cause_message": getattr(exc, "cause_message", ""),
            "user_file": getattr(top, "filename", "") if top else "",
            "user_line": getattr(top, "lineno", 0) if top else 0,
            "user_source": getattr(top, "source", "") if top else "",
        }
    return desc


@dataclass(frozen=True)
class _Partial(Generic[R]):
    """A partial under ``skip_failed``: reductions carry the failures alongside the value."""

    value: R
    failed: tuple[FailedChunk, ...] = ()


def _failed_chunk(label: str, partition: Partition, error: BaseException) -> FailedChunk:
    return FailedChunk(
        label, f"{type(error).__name__}: {error}", dead_letter_descriptor(label, partition, error)
    )


def _by_label(failed: Iterable[FailedChunk]) -> tuple[FailedChunk, ...]:
    return tuple(sorted(failed, key=lambda f: f.partition))


@dataclass(frozen=True)
class _SkipProcess(Generic[R]):
    """``process`` turning a failed chunk into an empty partial, so no reduction tree changes."""

    process: Callable[[Partition, WorkerResources], R]
    empty: Callable[[], R]
    # records, not errors: a lost worker's error holds its run's frames, and with them its futures
    lost: Mapping[str, FailedChunk] = field(default_factory=dict)

    def __call__(self, partition: Partition, resources: WorkerResources) -> _Partial[R]:
        """Return the chunk's partial, or an empty one naming why there is none."""
        failed = self.lost.get(chunk_label(partition)) if self.lost else None
        if failed is None:
            try:
                return _Partial(self.process(partition, resources))
            except Exception as exc:  # an interrupt or exit stops the run; it is not a dead letter
                failed = _failed_chunk(chunk_label(partition), partition, exc)
        return _Partial(self.empty(), (failed,))

    def bind_services(self, endpoints: Mapping[str, str]) -> _SkipProcess[R]:
        """Bind ``endpoints`` into the wrapped process."""
        if not isinstance(self.process, Bindable):
            return self
        return replace(self, process=self.process.bind_services(endpoints))

    def resolve_services(self, value: _Partial[R]) -> _Partial[R]:
        """Resolve the merged value through the wrapped process; the failures stay."""
        if not isinstance(self.process, Resolvable):
            return value
        return replace(value, value=on_value(self.process.resolve_services, value.value))


@dataclass(frozen=True)
class _SkipCombine(Generic[R]):
    """``combine`` on partials, which ends the run once a merge holds too many failures."""

    combine: Callable[[R, R], R]
    budget: int

    def __call__(self, a: _Partial[R], b: _Partial[R]) -> _Partial[R]:
        """Merge values and failures."""
        failed = a.failed + b.failed
        if len(failed) > self.budget:
            raise TooManyFailedChunksError(_by_label(failed))
        return _Partial(self.combine(a.value, b.value), failed)


def _empty_partial(empty: Callable[[], R]) -> _Partial[R]:
    return _Partial(empty())


def n_failed(partial: object) -> int:
    """Return how many failed chunks ``partial`` holds, for ``ExecContext.errors``."""
    return len(partial.failed) if isinstance(partial, _Partial) else 0


def skipped_error(partial: object) -> str | None:
    """Return why ``partial`` holds no data, for a monitor's ERRORED event; ``None`` if it does."""
    return partial.failed[0].error if isinstance(partial, _Partial) and partial.failed else None


#: first bytes of a file :meth:`Plan.save` wrote; the graphed version and a newline follow
_SAVED_PLAN = b"graphed-plan "


def counted(n: int, noun: str) -> str:
    """Return ``n`` and ``noun``, plural unless ``n`` is 1."""
    return f"{n} {noun}{'' if n == 1 else 's'}"


def capped(items: Sequence[str]) -> str:
    """Return ``items`` joined for a summary line, those past the sixth counted."""
    rest = f", +{len(items) - 6} more" if len(items) > 6 else ""
    return ", ".join(items[:6]) + rest


@dataclass(repr=False)
class Plan(Generic[R]):
    """A partition-wise execution plan: what a runner runs.

    Its repr is a one-line summary that reads no data. :meth:`save` and :meth:`load` carry a plan
    from the session that built it to a later one, on the same install. That file is a transport,
    a pickle: the durable, preservable form of an analysis is its IR
    (:class:`~graphed.core.plan.DurablePlan`, :mod:`graphed.preserve`).
    """

    process: Callable[[Partition, WorkerResources], R]  # picklable; runs analysis on a chunk
    combine: Callable[[R, R], R]  # associative (and commutative) reducer
    empty: Callable[[], R]  # identity, for zero partitions / an empty fold
    tasks: Sequence[Task] = ()  # a fixed partition set -> a deterministic reduction tree
    next_tasks: Callable[[ExecContext], Iterable[Task] | None] | None = (
        None  # adaptive hook (DONE=None)
    )
    stop: StopCondition | None = None
    open_once: bool = False
    # a plain default, not a factory: a Plan pickled before this field existed still loads
    services: tuple[ServiceSpec, ...] = ()  # the services the plan's nodes name (graphed.services)
    # the run's value in its final form, called once on the head after the last combine
    finalize: Callable[[R], R] | None = None

    def __repr__(self) -> str:
        """Summarize the tasks, files, and whatever ``process.summary()`` adds."""
        head = f"Plan: {counted(len(self.tasks), 'task')}"
        sources = {(t.partition.uri, t.partition.tree) for t in self.tasks}
        if len(sources) == 1:
            uri, tree = next(iter(sources))
            # PureWindowsPath splits on both separators
            head += f" over 1 file ({PureWindowsPath(uri).name}{f':{tree}' if tree else ''})"
        elif sources:
            head += f" over {counted(len({uri for uri, _ in sources}), 'file')}"
        parts = [head, "adaptive"] if self.next_tasks is not None else [head]
        hook = getattr(self.process, "summary", None)
        name = getattr(self.process, "__qualname__", type(self.process).__name__)
        parts.append(f"runs {name}" if hook is None else hook())
        return ", ".join(parts)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write this plan to ``path``, for :meth:`load` on this install.

        It is a pickle, not the preservation format: see the class docstring. What a script
        defines goes by value, so it loads in a later session.
        """
        import cloudpickle

        # pickled first: one that cannot pickle leaves a plan already at ``path`` alone
        data = cloudpickle.dumps(self)
        with open(path, "wb") as file:
            file.write(_SAVED_PLAN + version().encode() + b"\n" + data)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Plan[Any]:
        """Return the plan :meth:`save` wrote to ``path``.

        Loading runs pickle, which can run any code, so load only plans you trust. A plan saved by
        another graphed version is refused, as what it pickles is private to its version. The
        versions of the other packages a plan pickles are not recorded, so they are not checked.
        """
        with open(path, "rb") as file:
            header = file.readline()
            if not header.startswith(_SAVED_PLAN):
                raise ValueError(f"{os.fspath(path)!r} is not a file Plan.save wrote")
            saved = header.removeprefix(_SAVED_PLAN).strip().decode(errors="replace")
            if saved != version():
                raise ValueError(
                    f"{os.fspath(path)!r} was saved by graphed {saved} and this is graphed "
                    f"{version()}: a saved plan loads only on the version that wrote it, so "
                    "rebuild it from the analysis"
                )
            plan: Plan[Any] = pickle.load(file)
        return plan


@dataclass(frozen=True)
class ExecResult(Generic[R]):
    """A run's value, with how many partitions and combines produced it."""

    value: R
    n_partitions: int
    n_combines: int
    stopped: StopReason | None = None
    failed: tuple[FailedChunk, ...] = ()  # the chunks ``skip_failed`` left out of ``value``


def finalize_result(plan: Plan[R] | DurablePlanV2, result: ExecResult[R]) -> ExecResult[R]:
    """Return ``result`` with its value passed through ``plan.finalize``.

    Every runner calls it once, on the head, on each result it returns — an empty, stopped or
    resumed run's too. A plan without ``finalize`` and a cancelled ``DurablePlanV2`` run (which has
    no value) come back as they are. Under ``skip_failed`` it also moves the failed chunks from the
    value to ``failed``.
    """
    if isinstance(result.value, _Partial):
        failed = _by_label(result.value.failed)
        if failed:
            warnings.warn(
                f"{len(failed)} failed chunk(s) are missing from this value; see .failed",
                stacklevel=3,
            )
        result = replace(result, value=result.value.value, failed=failed)
    if isinstance(plan, Plan):
        fn = plan.finalize
    elif plan.finalize is None or result.stopped is StopReason.CANCELLED:
        return result
    else:
        fn = plan.finalize.resolve()
    return result if fn is None else replace(result, value=fn(result.value))


def skipping(plan: Plan[Any] | DurablePlanV2, skip_failed: int = 0) -> Plan[Any] | DurablePlanV2:
    """Return ``plan`` going on past ``skip_failed`` failed chunks, or ``plan.stop.max_errors``.

    Every runner calls it on entry. With no budget it returns ``plan``.
    """
    if not isinstance(plan, Plan):
        if skip_failed:
            raise TypeError("skip_failed needs a Plan; a DurablePlanV2 has no chunk to leave out")
        return plan
    budget = skip_failed or (plan.stop.max_errors if plan.stop else None)
    if budget is not None and budget < 0:
        raise ValueError(f"a failed-chunk budget must be >= 0, got {budget}")
    if not budget or isinstance(plan.process, _SkipProcess):
        return plan
    return replace(
        plan,
        process=_SkipProcess(plan.process, plan.empty),
        combine=_SkipCombine(plan.combine, budget),
        empty=functools.partial(_empty_partial, plan.empty),
        stop=replace(plan.stop or StopCondition(), max_errors=budget),
    )


def without_lost(plan: Plan[Any] | DurablePlanV2, err: BaseException) -> Plan[Any]:
    """Return a skipping ``plan`` for a rerun with lost-worker ``err``'s chunk failed.

    Each rerun costs one from the budget: a new blame leaves its chunk out, a repeat blame (dask's
    can be unfair) lowers the budget. Past it this raises :class:`TooManyFailedChunksError` instead
    of running again. Raises ``err`` when there is no such rerun: the plan is not skipping, the
    worker held no chunk of the plan, or a pull stream cannot be replayed.
    """
    if (
        not isinstance(plan, Plan)
        or not isinstance(plan.process, _SkipProcess)
        or plan.next_tasks is not None
    ):
        raise err
    label = getattr(err, "partition", "")
    parts = {chunk_label(t.partition): t.partition for t in plan.tasks}
    if label not in parts:
        raise err
    process = plan.process
    combine = cast("_SkipCombine[Any]", plan.combine)  # skipping wraps both
    budget = combine.budget - (label in process.lost)
    lost = {**process.lost, label: _failed_chunk(label, parts[label], err)}
    if len(lost) > budget:
        raise TooManyFailedChunksError(_by_label(lost.values())) from err
    return replace(
        plan,
        process=replace(process, lost=lost),
        combine=replace(combine, budget=budget),
        stop=replace(cast("StopCondition", plan.stop), max_errors=budget),
    )


def _stage_processes(plan: DurablePlanV2) -> tuple[Any, ...]:
    return tuple(stage.process.resolve() for stage in plan.stages)


def require_bound(plan: Plan[Any] | DurablePlanV2) -> None:
    """Raise :class:`UnboundServiceError` naming every service ``plan`` has no endpoint for.

    The plan's process (a ``DurablePlanV2``'s stage processes) names them; a runner calls it before
    its first task. A plan without ``services`` returns at once.
    """
    if not plan.services:
        return
    parts = (plan.process,) if isinstance(plan, Plan) else _stage_processes(plan)
    _require_parts([p for p in parts if isinstance(p, Bindable)])


def _require_parts(parts: Sequence[Bindable]) -> None:
    missing: dict[str, str] = {}
    while True:  # each pass through bind_services' own traversal names one more unbound service
        try:
            for part in parts:
                part.bind_services(missing)
        except UnboundServiceError as err:
            # a part refusing a name it was handed breaks the Bindable contract
            if err.name in missing:
                raise
            missing[err.name] = "tcp://unbound:0"
            continue
        if missing:
            raise UnboundServiceError(*sorted(missing))
        return


@runtime_checkable
class Executor(Protocol):
    """Run a `Plan` to a single reduced result.

    Reference impls live in graphed-exec-local (thread-pool and process-pool). Implementations MUST
    surface a worker failure to the driver intact (picklable) — never as an opaque string.
    """

    def run(self, plan: Plan[R]) -> ExecResult[R]:
        """Run ``plan`` to its result."""
        ...


# ---- M38: the inter-worker comms seam (peer reduction + work-stealing) ----------------------


@runtime_checkable
class WorkerTransport(Protocol):
    """An addressable, **best-effort, non-blocking** message channel.

    It runs between the driver and workers (and worker↔worker). It is the seam under peer reduction
    and work-stealing; a single-machine executor backs it with IPC (queues), and a future
    distributed executor backs the *same* interface with sockets/HTTP — so neither the reduction
    protocol nor the executor changes when the topology does.

    Contract:

    - every endpoint has a string ``address`` (``"driver"`` is reserved); :meth:`peers` lists the
      addresses this endpoint may send to;
    - :meth:`send` is **non-blocking** and **best-effort** — it enqueues and returns ``True``, or
      drops and returns ``False`` if the destination's inbox is full (back-pressure must never
      reach the data path, exactly as M37 telemetry is off-path);
    - messages are arbitrary **picklable** objects; their meaning (a steal request, a partial
      handoff) is defined by the protocol layer above, not here;
    - :meth:`poll` drains the local inbox without blocking; :meth:`recv` blocks up to ``timeout``
      and is for a dedicated coordination thread, never the task thread;
    - the transport imposes **no ordering or delivery guarantees** — determinism of a reduction is
      the protocol layer's job (it keys every combine by leaf index, never by arrival time or
      worker).
    """

    address: str

    def peers(self) -> tuple[str, ...]:
        """Return the addresses of the other endpoints."""
        ...

    def send(self, dest: str, message: object) -> bool:
        """Send ``message`` to ``dest``; return whether it was accepted."""
        ...

    def broadcast(self, message: object) -> None:
        """Send ``message`` to every peer."""
        ...

    def poll(self) -> list[tuple[str, object]]:
        """Return every message waiting, without blocking."""
        ...

    def recv(self, timeout: float | None = None) -> tuple[str, object] | None:
        """Return the next message, waiting up to ``timeout``, or ``None``."""
        ...

    def close(self) -> None:
        """Release the channel."""
        ...


# ---- M41: the Phase-2 ClusterExecutor launch seam (plan §6.1.3) ---------------------------------
# Three data-only siblings of Executor/WorkerTransport. The concrete cross-process Store + launcher
# live in graphed-exec-local BEHIND these types; this module stays pure (§A.4 — no backend/exec
# import).
@dataclass
class AddressTable:
    """A data-only ``node_id -> routable (host, port)`` table a *launcher* populates.

    Each entry is tagged with ``registered_by`` provenance — the CHILD that announced it, not the
    driver.

    This is **ephemeral runtime state**: announced host/port/pid live here and MUST NOT be folded
    into the durable plan (:class:`~graphed.core.plan.DurablePlanV2`) or the content-addressed task
    ids — addresses vary run-to-run, so baking them into the canonical IR would make task ids
    non-deterministic. Multi-host launch stays deferred (Phase-2); only the seam ships
    now.
    """

    addresses: dict[object, tuple[str, int]] = field(default_factory=dict)
    provenance: dict[object, object] = field(default_factory=dict)

    def register(self, node_id: object, host: str, port: int, *, registered_by: object) -> None:
        """Record ``node_id -> (host, port)`` and who announced it.

        The announcer is the registering child, not the driver.
        """
        self.addresses[node_id] = (host, int(port))
        self.provenance[node_id] = registered_by

    def lookup(self, node_id: object) -> tuple[str, int]:
        """Return the routable ``(host, port)`` a launcher registered for ``node_id``."""
        return self.addresses[node_id]

    def registered_by(self, node_id: object) -> object:
        """Return the registration provenance for ``node_id``.

        That is the child endpoint that announced its address.
        """
        return self.provenance[node_id]


@runtime_checkable
class NodeStore(Protocol):
    """The per-node Store handle the cluster runtime pulls blobs through.

    The concrete node-local Store in graphed-exec-local satisfies this *behind* the core type.

    ``fetch(digest)`` is the bulk data-plane endpoint: **idempotent** (re-fetching a digest already
    present returns the same bytes and moves none) and **byte-backpressured / spillable** (a large
    blob streams to the node-local Store instead of sitting whole in RAM — the *body* is the
    executor's; the seam pins only the *signature*, left open for an Arrow-Flight/RDMA transport of
    the same shape).
    """

    def fetch(self, digest: str) -> object:
        """Return the object stored under ``digest``."""
        ...


@runtime_checkable
class ClusterExecutor(Protocol):
    """An :class:`Executor`-shaped launch seam over a cluster.

    A launcher populates an :class:`AddressTable`, each node exposes a :class:`NodeStore` ``fetch``
    endpoint, and ``run`` drives a ``Plan`` to a reduced result across them. Multi-host launch is
    Phase-2 — defining the seam here keeps the reduction protocol and the single-machine executors
    unchanged when the topology grows.
    """

    def run(self, plan: Plan[R]) -> ExecResult[R]:
        """Run ``plan`` across the cluster to its result."""
        ...


# ---- M37: the passive live-dashboard seam (data-only; no web, no profiler dep) --------------


class TaskPhase(StrEnum):
    """A task's lifecycle phase, as a monitor sees it."""

    SUBMITTED = "submitted"
    STARTED = "started"
    FINISHED = "finished"
    ERRORED = "errored"


@dataclass(frozen=True)
class TaskEvent:
    """A passive, display-only record of one task's lifecycle transition.

    Carries no un-picklable objects — it crosses a process boundary from a ``ProcessExecutor``
    worker back to the driver. ``error`` is a pre-rendered summary string, never an exception
    object; ``partition`` is a human label. Per task the contract is exactly one ``SUBMITTED``,
    then one ``STARTED``, then exactly one of ``FINISHED`` | ``ERRORED`` — except that a task a
    :class:`RunControl` cancel kept from starting ends at its ``SUBMITTED``, and that a backend
    retrying a task repeats ``STARTED`` and the terminal event for each attempt. For a monitor that
    opts into lean events (:func:`lean_events`) the sequence is ``SUBMITTED``, then ``FINISHED`` |
    ``ERRORED`` with an empty ``partition``: only the driver's ``SUBMITTED`` carries the label.
    """

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
    """A per-worker statistical sampler.

    graphed-debug supplies the implementation (pyinstrument-backed); graphed-exec-local *drives* it
    without importing any profiler. ``flush``/``stop`` return a serialized sample tree (bytes) the
    driver merges, or ``None``.
    """

    def start(self) -> None:
        """Start sampling."""
        ...

    def flush(self) -> bytes | None:
        """Return the samples since the last flush, or ``None``."""
        ...

    def stop(self) -> bytes | None:
        """Stop sampling and return the remaining samples, or ``None``."""
        ...


@runtime_checkable
class Monitor(Protocol):
    """A passive observer of a run.

    An executor *emits* through it; it MUST NOT influence task order, the reduction tree, or
    results — the determinism gate is green attached-or-not. A monitor that raises is swallowed by
    the emitting executor (see :func:`emit_task`). ``worker_profiler_factory`` returns
    a *picklable* zero-arg factory shipped to workers (``None`` ⇒ no sampling). :class:`RunControl`
    is its non-passive counterpart: the object a runner reads to pause or cancel.
    """

    def on_task(self, event: TaskEvent) -> None:
        """Observe a task event."""
        ...

    def on_profile(self, worker: str, payload: bytes) -> None:
        """Observe a worker's profile payload."""
        ...

    def on_combine(self, leaves_done: int) -> None:
        """Observe the reduction's progress."""
        ...

    def worker_profiler_factory(self) -> Callable[[], WorkerProfiler] | None:
        """Return the factory workers build their profiler with, or ``None``."""
        ...


class MonitorBase:
    """A :class:`Monitor` that ignores everything: subclass it and override what you need.

    It adds the two optional run hooks, ``on_run_start`` and ``on_run_end``, which a runner calls
    on any monitor that has them (:func:`observed_run`).
    """

    def on_task(self, event: TaskEvent) -> None:
        """Observe a task event."""

    def on_profile(self, worker: str, payload: bytes) -> None:
        """Observe a worker's profile payload."""

    def on_combine(self, leaves_done: int) -> None:
        """Observe the reduction's progress."""

    def worker_profiler_factory(self) -> Callable[[], WorkerProfiler] | None:
        """Return no factory: workers do not sample."""
        return None

    def on_run_start(self, n_tasks: int) -> None:
        """Observe a run starting with ``n_tasks`` known tasks."""

    def on_run_end(self) -> None:
        """Observe the run ending, however it ended."""


class RunState(StrEnum):
    """A run's control state."""

    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"


_COMMANDS = ("pause", "resume", "cancel")


class RunControl:
    """A thread-safe pause/resume/cancel switch a runner reads before starting each task.

    ``pause`` holds a RUNNING run, ``resume`` releases a PAUSED one, and ``cancel`` is sticky: once
    CANCELLED only ``reset`` (which a runner calls when a run ends) returns it to RUNNING.
    """

    def __init__(self) -> None:
        """Start in the running state."""
        self._cond = threading.Condition()
        self._state = RunState.RUNNING

    @property
    def state(self) -> RunState:
        """The current control state."""
        return self._state

    def _move(self, to: RunState, *, only_from: RunState | None = None) -> None:
        with self._cond:
            if only_from is None or self._state is only_from:
                self._state = to
            self._cond.notify_all()

    def pause(self) -> None:
        """Pause a running run."""
        self._move(RunState.PAUSED, only_from=RunState.RUNNING)

    def resume(self) -> None:
        """Resume a paused run."""
        self._move(RunState.RUNNING, only_from=RunState.PAUSED)

    def cancel(self) -> None:
        """Cancel the run."""
        self._move(RunState.CANCELLED)

    def reset(self) -> None:
        """Return to the running state."""
        self._move(RunState.RUNNING)

    def apply(self, cmd: str) -> None:
        """Run the named command (the dashboard's wire form)."""
        if cmd not in _COMMANDS:
            raise ValueError(
                f"unknown run-control command {cmd!r} (expected pause, resume or cancel)"
            )
        getattr(self, cmd)()

    def wait(self, timeout: float | None = None) -> RunState:
        """Block while PAUSED; return the state woken in (PAUSED only when ``timeout`` elapsed)."""
        with self._cond:
            self._cond.wait_for(lambda: self._state is not RunState.PAUSED, timeout)
            return self._state


def emit_task(monitor: Monitor | None, event: TaskEvent) -> None:
    """Best-effort task emission.

    A ``None`` monitor is a no-op, and a monitor that raises must never break a run.
    """
    if monitor is None:
        return
    with contextlib.suppress(Exception):
        monitor.on_task(event)


def lean_events(monitor: Monitor | None) -> bool:
    """Return True when ``monitor`` opts into lean events.

    It opts in by carrying ``lean_events = True`` (exactly).
    """
    return getattr(monitor, "lean_events", False) is True


def complete_events(monitor: Monitor | None) -> bool:
    """Return True when ``monitor`` asks for every event of a run.

    It asks for them by the time ``run()`` returns or raises, by carrying ``complete_events =
    True`` (exactly).
    """
    return getattr(monitor, "complete_events", False) is True


def call_monitor(monitor: object, hook: str, *args: object) -> Any:
    """Return ``monitor.hook(*args)``, or ``None`` when it has no such hook or it raises.

    A monitor is passive: neither a missing hook nor a raising one may end a run.
    """
    with contextlib.suppress(Exception):
        return getattr(monitor, hook)(*args)
    return None


def worker_profiler_factory(monitor: Monitor | None) -> Callable[[], WorkerProfiler] | None:
    """Return the factory ``monitor`` gives its workers, or ``None``."""
    return cast(
        "Callable[[], WorkerProfiler] | None", call_monitor(monitor, "worker_profiler_factory")
    )


@contextlib.contextmanager
def observed_run(monitor: Monitor | None, plan: Plan[Any] | DurablePlanV2) -> Iterator[None]:
    """Bracket a run in ``monitor``'s optional ``on_run_start`` and ``on_run_end``.

    A runner enters this once per run; the end is reported when the run raises or is cancelled too.
    A staged plan reports 0 tasks: it emits no task events.
    """
    call_monitor(monitor, "on_run_start", len(plan.tasks) if isinstance(plan, Plan) else 0)
    try:
        yield
    finally:
        call_monitor(monitor, "on_run_end")


def worker_monitor_factory(monitor: Monitor | None) -> Callable[[], Monitor] | None:
    """Return the picklable zero-arg factory a monitor offers for per-worker push.

    ``None`` when the monitor defines no ``worker_monitor_factory`` (the ``Monitor`` protocol stays
    four methods) or it raises.
    """
    return cast("Callable[[], Monitor] | None", call_monitor(monitor, "worker_monitor_factory"))


def partition_label(partition: Partition) -> str:
    """Return a short human label for a partition (display only).

    A blind one names its step.
    """
    if partition.blind_step is not None:
        return f"{partition.uri}:{partition.tree}:{partition.blind_step}/{partition.blind_n_steps}"
    return f"{partition.uri}:{partition.tree}:{partition.entry_start}-{partition.entry_stop}"


def chunk_label(
    partition: Partition, *, file_entries: int | None = None, rows: int | None = None
) -> str:
    """Return the label an error shows for ``partition``.

    ``chunk 7/20 of data.root:Events (entries 15000-17500)``. A blind partition's range comes from
    ``file_entries``, its file's entry count; without one, ``rows`` gives its size. The driver of a
    lost worker has neither, so the label stops at the step.
    """
    tree = partition.tree.strip("/").partition(";")[0]
    name = f"{partition.uri}:{tree}" if tree else partition.uri
    if not partition.is_blind:
        return f"{name} (entries {partition.entry_start}-{partition.entry_stop})"
    chunk = f"chunk {cast(int, partition.blind_step) + 1}/{partition.blind_n_steps} of {name}"
    if file_entries is not None:
        known = partition.resolve(file_entries)
        return f"{chunk} (entries {known.entry_start}-{known.entry_stop})"
    return chunk if rows is None else f"{chunk} ({rows} entries)"


def _close_handle(handle: object) -> None:
    close = getattr(handle, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):  # a handle's own close must not break the run
            close()


class LocalResources:
    """Reference :class:`WorkerResources`.

    ``open_once(uri, opener)`` opens a uri at most once and reuses the handle for that
    runner/worker's later chunks. The set of simultaneously-open handles is **bounded** to
    ``max_open`` — when a new open would exceed it, the least-recently-used handle is closed and
    dropped (a uri reopened after eviction is opened again). ``close()`` releases every handle. The
    bound matters for a long-lived worker (a persistent process pool over many files): without it,
    ``open_once`` would accumulate every file ever opened for the worker's whole lifetime.
    """

    def __init__(self, max_open: int = 128) -> None:
        """Keep at most ``max_open`` handles open at once."""
        self._handles: OrderedDict[str, object] = OrderedDict()
        self._max_open = max_open
        self.open_count = 0  # real opens performed (test/diagnostic introspection)

    def open_once(self, uri: str, opener: Callable[[str], object]) -> object:
        """Open ``uri`` once and reuse the handle, evicting the least recently used."""
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
        """Close every open handle."""
        for handle in self._handles.values():
            _close_handle(handle)
        self._handles.clear()


class SequentialRunner:
    """The dependency-free reference :class:`Executor` of the ``Plan`` contract.

    Runs a plan's tasks **in key order**, in-process, with no worker pool. It lives beside the
    contract it executes so every layer — the frontend's deferred writers, histogram aggregation,
    preservation, the benchmarks — can run a ``Plan`` without depending on the executor package
    (which the frontend may not import). It is the canonical baseline any real executor
    (graphed-exec-local's thread/process pools) must match bit-for-bit.

    An optional :class:`Monitor` observes the run. It is purely passive: emission is
    best-effort and a misbehaving monitor cannot change the result. An optional :class:`RunControl`
    steers it: checked on entry and before each task, a pause holds the next task and a cancel
    returns the fold of the tasks that completed with ``stopped=StopReason.CANCELLED``.

    A :class:`~graphed.core.plan.DurablePlanV2` runs stage by stage, each task handed every
    upstream payload, to ``plan.value`` of the last stage's payloads; ``n_partitions`` counts the
    tasks of stages with no inputs and ``n_combines`` the rest. The control is checked on entry and
    before each stage, and a cancel returns ``value=None``: a staged run cut short has no value. It
    emits no task events.

    The plan's ``finalize`` runs on the value it returns (:func:`finalize_result`).

    It has the shape of a pooled runner, so one ``with`` block serves both: ``submit`` runs the
    plan before returning a finished future, and ``close`` has nothing to release.

    >>> import tempfile, awkward as ak
    >>> import graphed
    >>> from graphed.awkward import from_parquet, gak
    >>> from graphed.core import SequentialRunner
    >>> path = tempfile.mkdtemp() + "/events.parquet"
    >>> _ = ak.to_parquet(ak.Array({"MET_pt": [10.0, 55.0, 30.0]}), path)
    >>> plan = graphed.aggregate_plan(gak.sum(from_parquet("events", path).MET_pt))
    >>> float(SequentialRunner().run(plan).value)
    95.0
    """

    #: ``submit`` returns once its plan has finished, so none is ever waiting on another.
    max_in_flight = 1

    def __init__(self, monitor: Monitor | None = None, control: RunControl | None = None) -> None:
        """Run in-process, reporting to ``monitor`` and obeying ``control``."""
        self.monitor = monitor
        self.control = control

    def __enter__(self) -> SequentialRunner:
        """Return the runner."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the runner."""
        self.close()

    def close(self) -> None:
        """Do nothing: every plan has finished, and the runner holds no pool."""

    @overload
    def submit(self, plan: Plan[R]) -> Future[ExecResult[R]]: ...
    @overload
    def submit(self, plan: DurablePlanV2) -> Future[ExecResult[Any]]: ...
    def submit(self, plan: Plan[R] | DurablePlanV2) -> Future[Any]:
        """Run ``plan`` and return a finished future of its result, or of the failure.

        The run holds the caller: a paused ``control`` blocks this call, and a cancel sent after it
        returns lands on the next run.
        """
        fut: Future[Any] = Future()
        try:
            fut.set_result(self.run(plan))
        except Exception as exc:
            fut.set_exception(exc)
        try:
            return fut
        finally:
            del fut  # a failure's traceback holds this frame, so ``fut`` here would be a cycle

    @overload
    def run(self, plan: Plan[R], *, skip_failed: int = 0) -> ExecResult[R]: ...
    @overload
    def run(self, plan: DurablePlanV2, *, skip_failed: int = 0) -> ExecResult[Any]: ...
    def run(
        self, plan: Plan[R] | DurablePlanV2, *, skip_failed: int = 0
    ) -> ExecResult[R] | ExecResult[Any]:
        """Run ``plan`` one task at a time in this process, past ``skip_failed`` failed chunks."""
        require_bound(plan)
        plan = skipping(plan, skip_failed)
        with observed_run(self.monitor, plan):
            result = self._run_tasks(plan) if isinstance(plan, Plan) else self._run_stages(plan)
        return finalize_result(plan, result)

    def _run_tasks(self, plan: Plan[R]) -> ExecResult[R]:
        resources = LocalResources()
        monitor = self.monitor
        control = self.control
        lean = lean_events(monitor)
        try:
            value = plan.empty()
            n = 0
            stopped: StopReason | None = None
            if control is not None and control.state is RunState.CANCELLED:
                return ExecResult(
                    value=value, n_partitions=0, n_combines=0, stopped=StopReason.CANCELLED
                )
            ordered = sorted(plan.tasks, key=lambda t: t.key)
            if monitor is not None:
                for task in ordered:
                    emit_task(monitor, self._event(TaskPhase.SUBMITTED, task))
            for task in ordered:
                if control is not None and control.wait() is RunState.CANCELLED:
                    stopped = StopReason.CANCELLED
                    break
                if monitor is not None and not lean:
                    emit_task(monitor, self._event(TaskPhase.STARTED, task))
                try:
                    partial = plan.process(task.partition, resources)
                except Exception as exc:
                    if monitor is not None:
                        error = f"{type(exc).__name__}: {exc}"
                        emit_task(
                            monitor, self._event(TaskPhase.ERRORED, task, error=error, lean=lean)
                        )
                    raise
                value = plan.combine(value, partial)
                if monitor is not None:
                    why = skipped_error(partial)
                    phase = TaskPhase.ERRORED if why else TaskPhase.FINISHED
                    emit_task(monitor, self._event(phase, task, error=why, lean=lean))
                n += 1
            return ExecResult(
                value=value, n_partitions=n, n_combines=max(0, n - 1), stopped=stopped
            )
        finally:
            resources.close()  # release file handles deterministically at end of run
            if control is not None and control.state is RunState.CANCELLED:
                control.reset()  # a cancel ends this run only, whether or not a check saw it

    def _run_stages(self, plan: DurablePlanV2) -> ExecResult[Any]:
        resources = LocalResources()
        control = self.control
        ran = [0, 0]  # tasks of stages with no inputs, tasks of the others
        try:
            payloads: list[list[bytes]] = []
            for stage in plan.stages:
                if control is not None and control.wait() is RunState.CANCELLED:
                    return ExecResult(
                        value=None,
                        n_partitions=ran[0],
                        n_combines=ran[1],
                        stopped=StopReason.CANCELLED,
                    )
                process = stage.process.resolve()
                upstream = tuple(p for dep in stage.inputs for p in payloads[dep])
                out = []
                for task in stage.tasks:
                    out.append(process(task, upstream, resources))
                    ran[bool(stage.inputs)] += 1
                payloads.append(out)
            return ExecResult(
                value=plan.value(payloads[-1]), n_partitions=ran[0], n_combines=ran[1]
            )
        finally:
            resources.close()
            if control is not None and control.state is RunState.CANCELLED:
                control.reset()  # a cancel ends this run only, whether or not a check saw it

    @staticmethod
    def _event(
        phase: TaskPhase, task: Task, *, error: str | None = None, lean: bool = False
    ) -> TaskEvent:
        return TaskEvent(
            phase=phase,
            key=task.key,
            worker="seq",
            t=time.perf_counter(),
            partition="" if lean else partition_label(task.partition),
            n_entries=task.partition.n_entries,
            error=error,
        )
