"""Partition-wise aggregation plans: the multi-output, one-pass-over-a-shared-sub-graph engine.

A query producing several outputs that share a sub-graph — one selection feeding two histograms, a
sum and a count over the same cut, ... — must evaluate the shared sub-graph ONCE, not once per
output. :func:`aggregate_plan` compiles all outputs into ONE IR (so a shared sub-expression interns
to a single node), reads each partition once (projected to the UNION of the outputs' columns),
evaluates the IR once, and reduces the result. It is the dask multi-output ``compute`` analogue at
graphed's plan layer; the per-output REDUCTION is the caller's (``reduce`` folds one partition's
output-node values into a partition result; ``combine``/``empty`` reduce across partitions — each
output is whatever monoid the caller supplies: histograms add, counts sum, ...). graphed-histogram
specializes this for boost histograms; any other partition-wise reduction reuses it directly.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Generic, TypeVar

from graphed.core import GraphStore, Partition
from graphed.core.execution import Plan, Task, WorkerResources
from graphed.core.plan import _partition_bytes, _sha256_hex

from .array import Array
from .errors import GraphedError
from .execute import (
    CompiledGraph,
    Frame,
    Key,
    OnFailure,
    compile_ir,
    evaluate_ir,
    external_key,
    refuse_chunk_partials,
)
from .projection import read_columns
from .services import Bindable, ServiceSpec, bind_externals, referenced_services
from .session import Session
from .varied import refuse_container
from .write import PartitionedSource, PartWrite, declared_columns

V = TypeVar("V")

#: A write as shipped: ``(codec, destination, name, value slot, kv)``, where ``kv`` is ``None`` or
#: ``((key, slot | str), ...)`` — an int slot for an Array value, the build-time ``str()`` otherwise.
#: Never a `PartWrite`: its Array holds the session's store, which does not pickle.
_ShippedWrite = tuple[
    Callable[[object, str, Mapping[str, str] | None], None],
    str,
    Callable[[Partition], str],
    int,
    tuple[tuple[str, int | str], ...] | None,
]


def resolve_backend(ref: Callable[[], Any] | str) -> Any:
    """A worker's evaluation backend: a zero-arg factory/class, or an importable ``"module:attr"``
    reference resolved HERE in the worker — behavior-carrying backends (whose behavior dicts hold
    lambdas) travel by import ref, never by pickling, so losing them is loud, not silent."""
    if isinstance(ref, str):
        import importlib  # noqa: PLC0415

        mod_name, _, attr = ref.partition(":")
        target = getattr(importlib.import_module(mod_name), attr)
        return target() if callable(target) else target
    return ref()


@dataclass(frozen=True)
class _PartitionReduce(Generic[V]):
    """One partition's work for a multi-output graph: read once, evaluate the shared IR once into the
    output-node values, then ``reduce`` them to this partition's result. Picklable for process pools."""

    ir: bytes
    source_name: str
    backend_factory: Callable[[], Any] | str
    reader: PartitionedSource
    columns: tuple[str, ...] | None
    externals: tuple[tuple[str, Callable[..., object]], ...]
    reduce: Callable[[list[object]], V]
    #: §8.2(i): the shipped closure's variation-label channel — declared here at m48 and fed by
    #: `aggregate_plan(on_compiled=...)`'s return value; m49's lowering populates it.
    variation_labels: tuple[Any, ...] | None = None
    #: §8.2(i)'s frames re-keyed onto the shipped IR, one per key: what lets EVERY raw worker
    #: failure point at the user's line, labelled or not.
    frames: tuple[tuple[Key, Frame], ...] = ()
    #: `aggregate_plan(store=)`: the checkpoint root each task captures its input and partial into,
    #: for `graphed.debug.replay`. The input is the chunk as read: buffers a projected read skipped
    #: stay unread placeholders.
    store: str | None = None
    #: `aggregate_plan(writes=)`, lowered; each writes one part from this partition's values.
    writes: tuple[_ShippedWrite, ...] = ()
    #: with writes, how many leading values are the outputs' (`reduce` gets those, then the paths)
    n_values: int | None = None

    def __call__(self, partition: Partition, resources: WorkerResources) -> V:
        chunk = self.reader.read_partition(partition, self.columns, resources)
        if self.store is None:
            # _evaluate inlined: the default path keeps its pre-capture frame count
            values = evaluate_ir(
                self.ir,
                resolve_backend(self.backend_factory),
                {self.source_name: chunk},
                externals=dict(self.externals),
                on_failure=self._attribute(str(partition)),
            )
            return self.reduce(self._write(values, partition) if self.writes else values)
        import cloudpickle  # noqa: PLC0415  (only a capturing plan needs it)

        from graphed.checkpoint import PickleCodec  # noqa: PLC0415

        store = self._open_store(f"{os.getpid()}-{threading.get_ident()}")
        cid, label, codec = self._capture_id(partition), str(partition), PickleCodec()
        capturable = getattr(resolve_backend(self.backend_factory), "capturable", lambda chunk: chunk)
        # the input is kept before evaluating, so a failing task's input survives it
        blob = store.put(cloudpickle.dumps(capturable(chunk), protocol=codec.PROTOCOL))
        store.record_done(f"{cid}:input", label, blob, stage="replay-input")
        result = self.reduce(self._evaluate(chunk, partition))
        store.record_done(f"{cid}:output", label, store.put(codec.encode(result)), stage="replay-output")
        return result

    def bind_services(self, endpoints: Mapping[str, str]) -> _PartitionReduce[V]:
        """A copy whose External evaluators and ``reduce`` carry ``endpoints`` where they take them."""
        reduce = self.reduce.bind_services(endpoints) if isinstance(self.reduce, Bindable) else self.reduce
        return replace(self, externals=bind_externals(self.externals, endpoints), reduce=reduce)

    def part_paths(self, partition: Partition) -> list[str]:
        """Where this partition's parts land, without reading."""
        return [os.path.join(destination, name(partition)) for _, destination, name, _, _ in self.writes]

    def _write(self, values: list[object], partition: Partition) -> list[object]:
        paths = self.part_paths(partition)
        for (codec, _, _, slot, kv), path in zip(self.writes, paths, strict=True):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            meta = None if kv is None else {k: v if isinstance(v, str) else str(values[v]) for k, v in kv}
            codec(values[slot], path, meta)
        return [*values[: self.n_values], *paths]

    def _evaluate(self, chunk: object, partition: Partition) -> list[object]:
        return evaluate_ir(
            self.ir,
            resolve_backend(self.backend_factory),
            {self.source_name: chunk},
            externals=dict(self.externals),
            on_failure=self._attribute(str(partition)),
        )

    def _capture_id(self, partition: Partition) -> str:
        """The id a task's captures are journaled under: the IR and the partition, nothing about the
        run, so one capture root holds one run."""
        return _sha256_hex(b"graphed-replay-capture-v1", self.ir, _partition_bytes(partition))

    def _open_store(self, node: str | None = None) -> Any:
        """The capture root as a checkpoint store: an fsspec URL when it contains ``://``, else a
        directory. Task and replay both open it here, so one root string always means one layout."""
        from graphed.checkpoint import FsspecStore, Store  # noqa: PLC0415  (only a capturing plan needs it)

        assert self.store is not None
        return FsspecStore(self.store, node) if "://" in self.store else Store(self.store, node)

    def _attribute(self, partition: str) -> OnFailure | None:
        """§8.2(ii): the worker-side wrap. A RAW failure at any key with a frame becomes a
        `StageError` pointing at the user's line, carrying the key's variation label when the label
        channel has an entry; a key with no frame re-raises the original untouched, since
        `StageError` needs frames at construction."""
        entries = dict(self.variation_labels or ())
        frames = dict(self.frames)
        if not entries and not frames:
            return None
        from .debug.errors import SourceFrame, StageError  # noqa: PLC0415  (import cycle)

        def attribute(key: Key, op: str, ins: list[object], exc: BaseException) -> BaseException | None:
            # §8.2(ii): "a `GraphedError` re-raises untouched on EVERY arm regardless of entry — it
            # is already an attributed error, and §6.1d's blame parity (the plan path re-raises the
            # guard's message verbatim) binds it".
            if isinstance(exc, GraphedError):
                return None
            entry = entries.get(key)
            if entry is not None:
                labels, frame = entry
            else:
                frame = frames.get(key)
                if frame is None:
                    return None
                labels = ()
            return StageError(
                op=op,
                frames=(SourceFrame(*frame),),
                # a worker holds values, not forms: the runtime types are what it can honestly report
                input_forms=tuple(type(value).__name__ for value in ins),
                partition=partition,
                cause_type=type(exc).__name__,
                cause_message=str(exc),
                opt_level=1,  # `aggregate_plan` always compiles optimized
                variation=",".join(sorted(labels)),
            )

        return attribute


def external_evaluators(session: Session, compiled: CompiledGraph) -> dict[str, Callable[..., object]]:
    """Every External surviving in the compiled IR, keyed by :func:`external_key`, resolved to the
    evaluator the recording session holds for it.

    This is the SINGLE wiring point for a plan's External evaluators. Every External — a
    ``hist.graphed`` fill FillEvaluator AND an upstream correctionlib/ONNX scale factor alike — is
    registered on ``session._externals`` at record time, so one pass over the compiled External nodes
    wires them all: a fill whose input cone reads a correctionlib SF now carries that SF's evaluator,
    which is what makes "hundreds of histograms with systematic variations" run through a plan. The
    key includes the node's params, so N systematic universes off one CorrectionSet each resolve to
    their OWN evaluator (they share a payload ``content_hash`` but not a params digest)."""
    by_key: dict[str, Callable[..., object]] = {}
    recorded = session._store.nodes()
    for node_id, (fn, _inputs) in session._externals.items():
        by_key[external_key(recorded[node_id])] = fn
    wired: dict[str, Callable[..., object]] = {}
    for node in GraphStore.deserialize(bytes(compiled.ir)).nodes():
        if node["kind"] == "external":
            key = external_key(node)
            evaluator = by_key.get(key)
            if evaluator is not None:
                wired[key] = evaluator
    return wired


def plan_services(
    session: Session, compiled: CompiledGraph, names: Sequence[str] | None = None
) -> tuple[ServiceSpec, ...]:
    """The ``Plan.services`` of a plan over ``compiled``: the session's specs its External nodes name,
    plus ``names``."""
    return referenced_services(session, GraphStore.deserialize(bytes(compiled.ir)).nodes(), names)


def aggregate_plan(
    *outputs: Array,
    reduce: Callable[[list[Any]], V],
    combine: Callable[[V, V], V],
    empty: Callable[[], V],
    externals: Mapping[str, Callable[..., object]] | None = None,
    backend: Callable[[], Any] | str | None = None,
    steps_per_file: int = 1,
    partitions: Sequence[Partition] | None = None,
    on_compiled: Callable[[CompiledGraph], Any] | None = None,
    store: str | os.PathLike[str] | None = None,
    services: Sequence[str] | None = None,
    writes: Sequence[PartWrite] = (),
) -> Plan[V]:
    """Build a one-pass partition-wise reduction :class:`~graphed.core.execution.Plan` over the
    session's single partitioned source (see module docstring). ``outputs`` are the output Arrays
    (their shared sub-graph is compiled to one IR and evaluated once per partition); ``externals``
    binds any External payload evaluator; ``backend`` is the workers' evaluation backend (factory,
    class, or ``"module:attr"`` ref; defaults to the session backend's type). ``run(plan).value`` is
    the ``reduce``+``combine`` aggregate over all partitions.

    ``on_compiled`` is §7.2's seam onto the internally compiled :class:`CompiledGraph` — the
    artifact is otherwise unreachable from the caller. It fires ONCE, and whatever it returns is
    carried onto the shipped closure's ``variation_labels``.

    ``store`` (a directory, or an fsspec URL) makes each task capture its input chunk and its
    ``reduce`` partial into that checkpoint root, so :func:`graphed.debug.replay` can re-run a task
    of this plan later from exactly what it read. A directory must be one every worker shares.

    ``Plan.services`` holds the session's specs named by the compiled External nodes'
    ``params["service"]`` and by ``services`` (names a node does not carry, e.g. a service the
    ``reduce`` calls), in name order.

    ``writes`` (:class:`~graphed.write.PartWrite`) write one part per task from the same read and
    evaluation: ``reduce`` then receives the outputs' values as without writes, followed by one
    part path per write. ``outputs`` may be empty when ``writes`` is not."""
    writes = tuple(writes)
    metadata = [value for w in writes for value in (w.metadata or {}).values()]
    refuse_container("graphed.aggregate_plan", *outputs, *(w.array for w in writes), *metadata)
    if not outputs and not writes:
        raise ValueError("aggregate_plan needs at least one output Array or write")
    if writes and store is not None:
        # the capturing path evaluates and reduces only; it would skip every write
        raise TypeError("aggregate_plan(store=) does not capture writes")
    arrays = [*outputs, *(w.array for w in writes), *(v for v in metadata if isinstance(v, Array))]
    session = arrays[0].session
    if any(a.session is not session for a in arrays):
        raise TypeError("all outputs of one plan must record into one session")
    partitioned = {nid: d for nid, d in session.sources().items() if isinstance(d, PartitionedSource)}
    if len(partitioned) != 1:
        raise TypeError(
            f"aggregate_plan needs exactly one partitioned source; this session has {len(partitioned)}"
        )
    ((nid, data),) = partitioned.items()
    compiled = compile_ir(session, *arrays)
    order = {cid: i for i, cid in enumerate(GraphStore.deserialize(bytes(compiled.ir)).outputs())}

    def slot(array: Array) -> int:
        # the optimizer may merge outputs (`w * 1.0` into `w`); the correspondence knows where each landed
        return order[compiled.correspondence.node_map[array.node_id][0]]

    written = {compiled.correspondence.node_map[w.array.node_id][0] for w in writes}
    refuse_chunk_partials(compiled, as_outputs=written)
    # Wire EVERY External surviving in the compiled IR from the session (fills + upstream corrections
    # alike); an explicit `externals=` (keyed by `external_key`) overrides the auto-wired evaluator.
    wired = external_evaluators(session, compiled)
    if externals:
        wired.update(externals)
    declared = declared_columns(data, arrays)
    process = _PartitionReduce(
        ir=bytes(compiled.ir),
        source_name=session.source_name(nid),
        backend_factory=backend if backend is not None else type(session.backend),
        reader=data,
        columns=read_columns(arrays, nid) if declared is None else declared,
        externals=tuple(wired.items()),
        reduce=reduce,
        variation_labels=None if on_compiled is None else on_compiled(compiled),
        frames=compiled.correspondence.frames,
        store=None if store is None else os.fspath(store),
        writes=tuple(
            (
                w.codec,
                w.destination,
                w.name,
                slot(w.array),
                None
                if w.metadata is None
                else tuple((k, slot(v) if isinstance(v, Array) else str(v)) for k, v in w.metadata.items()),
            )
            for w in writes
        ),
        # outputs are marked first, so their distinct values lead the evaluated list
        n_values=len({slot(o) for o in outputs}) if writes else None,
    )
    if partitions is None:
        partitions = data.partitions(steps_per_file)
    tasks = tuple(Task(i, p) for i, p in enumerate(partitions))
    _refuse_shared_parts((process, t.partition) for t in tasks)
    return Plan(
        process=process,
        combine=combine,
        empty=empty,
        tasks=tasks,
        services=plan_services(session, compiled, services),
    )


def _written_parts(process: object, partition: Partition) -> Sequence[str]:
    """The parts a plan's ``process`` writes for ``partition``, from its optional
    ``part_paths(partition)`` hook (no I/O); a process without the hook is not checked."""
    hook = getattr(process, "part_paths", None)
    return () if hook is None else hook(partition)


def _refuse_shared_parts(runs: Iterable[tuple[object, Partition]]) -> None:
    """Driver-side, one hook call per task: a part two writes share would be silently overwritten."""
    seen: set[str] = set()
    for process, partition in runs:
        for path in map(os.path.normpath, _written_parts(process, partition)):
            if path in seen:
                raise ValueError(
                    f"two writes of this plan write the same part {path!r}: `name` must tell every"
                    " partition apart (a blind partition's entry range is 0-0 until it is read), and"
                    " plans collated together need distinct parts"
                )
            seen.add(path)


@dataclass(frozen=True)
class _Collated:
    """A collated plan's process: the task's ``(uri, tree)`` picks the sub-plan whose graph reads it."""

    processes: Mapping[str, Callable[[Partition, WorkerResources], Any]]
    #: O(files), never O(tasks): it ships once per worker, not in any task
    route: Mapping[tuple[str, str], str]

    def __call__(self, partition: Partition, resources: WorkerResources) -> dict[str, Any]:
        name = self.route[(partition.uri, partition.tree)]
        return {name: self.processes[name](partition, resources)}

    def part_paths(self, partition: Partition) -> Sequence[str]:
        return _written_parts(self.processes[self.route[(partition.uri, partition.tree)]], partition)


@dataclass(frozen=True)
class _CollatedCombine:
    """Per name with that sub-plan's combine; a name on one side only passes through."""

    combines: Mapping[str, Callable[[Any, Any], Any]]

    def __call__(self, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        merged = {**a, **{n: self.combines[n](a[n], v) if n in a else v for n, v in b.items()}}
        return {n: merged[n] for n in self.combines if n in merged}  # mapping order, whatever the tree


def collate(plans: Mapping[str, Plan[Any]]) -> Plan[dict[str, Any]]:
    """ONE executable plan over several plans' tasks — plans over different sources or graphs (data
    and MC, say) — whose value is ``{name: that plan's value}``.

    Tasks are each plan's in key order, concatenated in mapping order and re-keyed ``0..N-1``, so
    the reduction tree stays the runner's. A task runs the process of the plan that holds its
    ``(uri, tree)``; a ``(uri, tree)`` held by two plans is refused (record both over one source so
    they share the read), and so is a part two tasks would both write, across plans too (named by
    each process's optional ``part_paths(partition)`` hook; a process without it is not checked).
    A name is in the value exactly when its plan has at least one task.
    Running each plan on its own and collecting ``{name: value}`` gives the same product when each
    ``combine`` is exact."""
    if not plans:
        raise ValueError("collate needs at least one plan")
    route: dict[tuple[str, str], str] = {}
    tasks: list[Task] = []
    for name, plan in plans.items():
        if plan.next_tasks is not None or plan.stop is not None:
            raise TypeError(f"collate needs plans with fixed tasks; {name!r} is adaptive or stoppable")
        for task in sorted(plan.tasks, key=lambda t: t.key):
            key = (task.partition.uri, task.partition.tree)
            owner = route.setdefault(key, name)
            if owner != name:
                raise ValueError(
                    f"{key!r} appears in plans {owner!r} and {name!r}; record both over one source so"
                    " one read serves both graphs"
                )
            tasks.append(Task(len(tasks), task.partition))
    process = _Collated({n: p.process for n, p in plans.items()}, route)
    _refuse_shared_parts((process, t.partition) for t in tasks)
    return Plan(
        process=process,
        combine=_CollatedCombine({n: p.combine for n, p in plans.items()}),
        empty=dict,
        tasks=tuple(tasks),
        open_once=any(p.open_once for p in plans.values()),
    )
