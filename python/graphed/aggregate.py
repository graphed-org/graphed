"""Partition-wise aggregation plans: the multi-output, one-pass-over-a-shared-sub-graph engine.

A query producing several outputs that share a sub-graph — one selection feeding two histograms, a
sum and a count over the same cut, ... — must evaluate the shared sub-graph ONCE, not once per
output. :func:`aggregate_plan` compiles all outputs into ONE IR (so a shared sub-expression interns
to a single node), reads each partition once (projected to the UNION of the outputs' columns),
evaluates the IR once, and reduces the result. It is the dask multi-output ``compute`` analogue at
graphed's plan layer; the per-output REDUCTION is the caller's (``reduce`` folds one partition's
output-node values into a partition result; ``combine``/``empty`` reduce across partitions — each
output is whatever monoid the caller supplies: histograms add, counts sum, ...), or the output's
own when it is a whole-dataset reduction with a known combine (:func:`fold_of`). graphed-histogram
specializes this for boost histograms; any other partition-wise reduction reuses it directly.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import operator
import os
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Generic, TypeVar, cast

from graphed.core import GraphStore, Partition
from graphed.core.execution import (
    Plan,
    Task,
    WorkerResources,
    capped,
    chunk_label,
    counted,
)
from graphed.core.plan import DurablePlanV2, _partition_bytes, _sha256_hex
from graphed.core.worker_warnings import (
    WarnedCombine,
    WarnedFinalize,
    carrying,
    keeping_warnings,
    on_value,
)

from .array import Array
from .errors import (
    GraphedError,
    OutputTypeError,
    PackKeyError,
    PackKeyOverflowError,
    SourceFrame,
    StageError,
    raising_frame,
)
from .execute import (
    CompiledGraph,
    Frame,
    Key,
    OnFailure,
    _compile_cone,
    compile_ir,
    evaluate_ir,
    external_key,
    refuse_chunk_partials,
)
from .projection import read_columns
from .services import Bindable, Resolvable, ServiceSpec, bind_externals, referenced_services
from .session import Session
from .varied import refuse_container
from .write import (
    PartitionedSource,
    PartWrite,
    declared_columns,
    join_part,
    prepare_part,
    sources_read,
    steps_for,
)

T = TypeVar("T")
V = TypeVar("V")

#: A write as shipped: ``(codec, destination, name, value slot, kv[, storage_options])``, where
#: ``kv`` is ``None`` or ``((key, slot | str), ...)`` — an int slot for an Array value, the
#: build-time ``str()`` otherwise — and ``storage_options`` items ship only when given, so a plan
#: without them pickles as it did before they existed.
#: Never a `PartWrite`: its Array holds the session's store, which does not pickle.
_Codec = Callable[[object, str, Mapping[str, str] | None], None]
_Kv = tuple[tuple[str, int | str], ...] | None
_ShippedWrite = (
    tuple[_Codec, str, Callable[[Partition], str], int, _Kv]
    | tuple[_Codec, str, Callable[[Partition], str], int, _Kv, tuple[tuple[str, Any], ...]]
)


@dataclass(frozen=True)
class _ByValue:
    """A user callable stdlib pickle cannot name (a lambda, closure or local def).

    It ships by value, the one use of cloudpickle §A.3.1 allows, so every runner's own pickling
    carries it.
    """

    fn: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def _keyed_as(self) -> Callable[..., Any]:
        return self.fn

    def __reduce__(self) -> tuple[Any, ...]:
        import cloudpickle

        return _load_by_value, (cloudpickle.dumps(self.fn),)


def _load_by_value(blob: bytes) -> _ByValue:
    import cloudpickle

    return _ByValue(cloudpickle.loads(blob))


def by_value(fn: T) -> T:
    """``fn``, a function or partial, wrapped to pickle through cloudpickle.

    By name when importable, by value otherwise (a lambda, a closure, a ``__main__`` def a worker
    cannot import). A callable object stays itself, so its ``bind_services``/``resolve_services``
    hooks are still seen.
    """
    if isinstance(fn, types.FunctionType | functools.partial):
        return _ByValue(fn)  # type: ignore[return-value]  # calls as fn does
    return fn


def worker_backend(session: Session, backend: Callable[[], Any] | str | None) -> Any:
    """Return the backend factory or import ref a plan's workers build their backend from.

    ``backend`` when given, else the session backend's ``worker_factory()`` when it has one (it
    rebuilds what the class alone does not, e.g. an awkward behavior), else its class.
    """
    if backend is not None:
        return backend
    hook = getattr(session.backend, "worker_factory", None)
    return type(session.backend) if hook is None else hook()


@functools.cache
def resolved_backend(ref: str) -> Any:
    """Return the backend the import ``ref`` names, resolved once per process."""
    return resolve_backend(ref)


def _adopted(backend: Any, value: T) -> T:
    """``value`` with its arrays given back the behavior the built ``backend`` carries.

    An op over several of them leaves an equal copy, which does not pickle as the shipped one.
    """
    adopt = getattr(backend, "adopt", None)
    return value if adopt is None else cast("T", adopt(value))


def _adopting(fn: Callable[..., Any], backend: Callable[[], Any] | str) -> Callable[..., Any]:
    """``fn``, its results' arrays given back the behavior of the backend ``backend`` builds.

    Only an import ref or a ``worker_factory()`` partial builds a behavior that pickles as shipped,
    so any other backend keeps ``fn`` itself.
    """
    return _Adopting(fn, backend) if isinstance(backend, str | functools.partial) else fn


@dataclass(frozen=True)
class _Adopting:
    """A plan's ``combine``/``empty``, its result's arrays given back a behavior.

    The behavior is the one the backend ``ref`` carries.
    """

    fn: Callable[..., Any]
    ref: Callable[[], Any] | str

    def __call__(self, *args: Any) -> Any:
        ref = self.ref
        built = resolved_backend(ref) if isinstance(ref, str) else resolve_backend(ref)
        return _adopted(built, self.fn(*args))


def resolve_backend(ref: Callable[[], Any] | str) -> Any:
    """Return a worker's evaluation backend.

    A zero-arg factory/class, or an importable ``"module:attr"`` reference resolved HERE in the
    worker — behavior-carrying backends (whose behavior dicts hold lambdas) travel by import ref,
    never by pickling, so losing them is loud, not silent.
    """
    if isinstance(ref, str):
        mod_name, _, attr = ref.partition(":")
        target = getattr(importlib.import_module(mod_name), attr)
        backend = target() if callable(target) else target
        hook = getattr(backend, "resolved_from", None)
        return backend if hook is None else hook(ref)
    return ref()


@dataclass(frozen=True)
class _PartitionReduce(Generic[V]):
    """One partition's work for a multi-output graph.

    Read once, evaluate the shared IR once into the output-node values, then ``reduce`` them to
    this partition's result. Picklable for process pools.
    """

    checkpointable: ClassVar[bool] = True
    checkpoint_ignore: ClassVar[tuple[str, ...]] = ("frames", "variation_labels")
    checkpoint_resolve: ClassVar[Mapping[str, Callable[[Any], Any]]] = {
        "backend_factory": resolve_backend
    }

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
    #: `aggregate_plan(storage_options=)`: passed to the capture root's fsspec filesystem.
    storage_options: tuple[tuple[str, Any], ...] = ()
    #: `aggregate_plan(writes=)`, lowered; each writes one part from this partition's values.
    writes: tuple[_ShippedWrite, ...] = ()
    #: with writes, how many leading values are the outputs' (`reduce` gets those, then the paths)
    n_values: int | None = None
    #: `aggregate_plan(opt_level=)`: the level `ir` was compiled at.
    opt_level: int = 1

    def __call__(self, partition: Partition, resources: WorkerResources) -> V:
        with keeping_warnings() as caught:
            chunk = self.reader.read_partition(partition, self.columns, resources)
            backend = resolve_backend(self.backend_factory)
            if self.store is None:
                # evaluate inlined: the default path keeps its pre-capture frame count
                with self.floating_point():
                    values = evaluate_ir(
                        self.ir,
                        backend,
                        {self.source_name: chunk},
                        externals=dict(self.externals),
                        on_failure=self._attribute(self._label(partition, chunk)),
                        chunked=True,
                    )
                result = _adopted(
                    backend,
                    self.reduce(self._write(values, partition) if self.writes else values),
                )
            else:
                result = self._captured(chunk, partition, backend)
        return cast("V", carrying(result, caught, self._label(partition, chunk)))

    def floating_point(self) -> contextlib.AbstractContextManager[object]:
        """Return what a task evaluates under: numpy's own state, unless it raises."""
        return contextlib.nullcontext()

    def _captured(self, chunk: object, partition: Partition, backend: Any) -> V:
        import cloudpickle

        from graphed.checkpoint import PickleCodec

        store = self._task_store()
        cid, label, codec = self.capture_id(partition), str(partition), PickleCodec()
        capturable = getattr(backend, "capturable", lambda chunk: chunk)
        # the input is kept before evaluating, so a failing task's input survives it
        blob = store.put(cloudpickle.dumps(capturable(chunk), protocol=codec.PROTOCOL))
        store.record_done(f"{cid}:input", label, blob, stage="replay-input")
        result = _adopted(backend, self.reduce(self.evaluate(chunk, partition)))
        store.record_done(
            f"{cid}:output", label, store.put(codec.encode(result)), stage="replay-output"
        )
        return result

    def bind_services(self, endpoints: Mapping[str, str]) -> _PartitionReduce[V]:
        """Return a copy carrying ``endpoints`` where it takes them.

        The copy's External evaluators and ``reduce`` carry them.
        """
        reduce = (
            self.reduce.bind_services(endpoints)
            if isinstance(self.reduce, Bindable)
            else self.reduce
        )
        return replace(self, externals=bind_externals(self.externals, endpoints), reduce=reduce)

    def resolve_services(self, value: V) -> V:
        """``value`` resolved by ``reduce`` when it takes it; ``value`` itself otherwise."""
        if not isinstance(self.reduce, Resolvable):
            return value
        resolved: V = self.reduce.resolve_services(value)
        return resolved

    def output_summaries(self) -> list[str]:
        """Describe each output, for a ``reduce`` that says what they are."""
        hook = getattr(self.reduce, "output_summaries", None)
        return [] if hook is None else list(hook())

    def summary(self) -> str:
        """Say what the plan's repr shows of this process; it reads no data and no IR."""
        named = self.output_summaries()
        outputs = f"outputs: {capped(named)}, " if named else ""
        columns = "every column" if self.columns is None else counted(len(self.columns), "column")
        writes = f", writes {counted(len(self.writes), 'dataset')}" if self.writes else ""
        return f"{outputs}reads {columns}{writes}"

    def part_paths(self, partition: Partition) -> list[str]:
        """Where this partition's parts land, without reading."""
        return [
            join_part(destination, name(partition)) for _, destination, name, *_ in self.writes
        ]

    def _write(self, values: list[object], partition: Partition) -> list[object]:
        paths = self.part_paths(partition)
        for (codec, _, _, slot, kv, *storage_options), path in zip(
            self.writes, paths, strict=True
        ):
            prepare_part(path, **dict(*storage_options))
            meta = (
                None
                if kv is None
                else {k: v if isinstance(v, str) else str(values[v]) for k, v in kv}
            )
            codec(values[slot], path, meta)
        return [*values[: self.n_values], *paths]

    def evaluate(self, chunk: object, partition: Partition) -> list[object]:
        """Evaluate the IR on ``chunk`` as a task does, its failures attributed alike."""
        with self.floating_point():
            return evaluate_ir(
                self.ir,
                resolve_backend(self.backend_factory),
                {self.source_name: chunk},
                externals=dict(self.externals),
                on_failure=self._attribute(self._label(partition, chunk)),
                chunked=True,
            )

    def capture_id(self, partition: Partition) -> str:
        """Return the id a task's captures are journaled under.

        The IR and the partition, nothing about the run, so one capture root holds one run.
        """
        return _sha256_hex(b"graphed-replay-capture-v1", self.ir, _partition_bytes(partition))

    def _task_store(self) -> Any:
        # only a capturing plan needs it
        from graphed.checkpoint.resume import task_store

        assert self.store is not None
        return task_store(self.store, self.storage_options)

    def open_store(self, node: str | None = None) -> Any:
        """Return the capture root as a checkpoint store, as every task and replay opens it."""
        # only a capturing plan needs it
        from graphed.checkpoint.resume import open_store

        assert self.store is not None
        return open_store(self.store, node, dict(self.storage_options))

    def _attribute(self, partition: str | Callable[[], str]) -> OnFailure | None:
        return attribute_failures(self.frames, partition, self.variation_labels, self.opt_level)

    def _label(self, partition: Partition, chunk: object) -> Callable[[], str]:
        return functools.partial(read_label, self.reader, partition, chunk)


def read_label(reader: Any, partition: Partition, chunk: Any) -> str:
    """Return ``partition``'s :func:`chunk_label`, resolved as far as this worker can."""
    entries: int | None = None
    rows: int | None = None
    if partition.is_blind:
        # a label must not mask the failure it reports
        with contextlib.suppress(Exception):
            entries = reader.num_entries(partition)
        with contextlib.suppress(Exception):
            rows = len(chunk)
    return chunk_label(partition, file_entries=entries, rows=rows)


def _type_string(value: object, limit: int = 200) -> str:
    """Return ``value``'s type as an array library prints it, its class name otherwise."""
    text = getattr(value, "typestr", None)  # awkward
    if not isinstance(text, str):
        dtype = getattr(value, "dtype", None)
        shape = getattr(value, "shape", ())
        text = (
            type(value).__name__ if dtype is None else " * ".join((*map(str, shape), str(dtype)))
        )
    return text if len(text) <= limit else f"{text[:limit]}..."  # a NanoEvents record runs long


def attribute_failures(
    frames: Sequence[tuple[Key, Frame]],
    partition: str | Callable[[], str],
    variation_labels: Sequence[Any] | None = None,
    opt_level: int = 1,
) -> OnFailure | None:
    """§8.2(ii): the worker-side wrap.

    A RAW failure at any key with a frame becomes a `StageError` pointing at the user's line,
    carrying the key's variation label when the label channel has an entry; a key with no frame
    re-raises the original untouched, since `StageError` needs frames at construction.
    ``partition`` may be a callable, so a task pays for its label only when it fails.
    """
    entries = dict(variation_labels or ())
    by_key = dict(frames)
    if not entries and not by_key:
        return None

    def attribute(
        key: Key, op: str, ins: list[object], exc: BaseException
    ) -> BaseException | None:
        # §8.2(ii): "a `GraphedError` re-raises untouched on EVERY arm regardless of entry — it
        # is already an attributed error, and §6.1d's blame parity (the plan path re-raises the
        # guard's message verbatim) binds it". A declared-type check fails only here, in the
        # worker, so it is attributed: the External's key carries its declaring line.
        if isinstance(exc, GraphedError) and not isinstance(exc, OutputTypeError):
            return None
        entry = entries.get(key)
        if entry is not None:
            labels, frame = entry
        else:
            frame = by_key.get(key)
            if frame is None:
                return None
            labels = ()
        kind = PackKeyError if isinstance(exc, PackKeyOverflowError) else StageError
        return kind(
            op=op,
            frames=(SourceFrame(*frame),),
            input_forms=tuple(_type_string(value) for value in ins),
            partition=partition if isinstance(partition, str) else partition(),
            cause_type=type(exc).__name__,
            cause_message=str(exc),
            opt_level=opt_level,
            variation=",".join(sorted(labels)),
            raised_at=raising_frame(exc),
        )

    return attribute


def _compile_at(opt_level: int, session: Session, arrays: Sequence[Array]) -> CompiledGraph:
    return (_compile_cone if opt_level == 0 else compile_ir)(session, *arrays)


def _slot_of(compiled: CompiledGraph) -> Callable[[Array], int]:
    """Where an Array's value sits among the values ``compiled`` evaluates to.

    The optimizer may merge outputs (``w * 1.0`` into ``w``), and the correspondence knows where
    each landed.
    """
    order = {cid: i for i, cid in enumerate(GraphStore.deserialize(bytes(compiled.ir)).outputs())}
    return lambda array: order[compiled.correspondence.node_map[array.node_id][0]]


def external_evaluators(
    session: Session, compiled: CompiledGraph
) -> dict[str, Callable[..., object]]:
    """Return every External surviving in the compiled IR, keyed by :func:`external_key`.

    Each is resolved to the evaluator the recording session holds for it.

    This is the SINGLE wiring point for a plan's External evaluators. Every External — a
    ``hist.graphed`` fill FillEvaluator AND an upstream correctionlib/ONNX scale factor alike — is
    registered on its Session at record time, so one pass over the compiled External
    nodes wires them all: a fill whose input cone reads a correctionlib SF now carries that SF's
    evaluator, which is what makes "hundreds of histograms with systematic variations" run through
    a plan. The key includes the node's params, so N systematic universes off one CorrectionSet
    each resolve to their OWN evaluator (they share a payload ``content_hash`` but not a params
    digest).
    """
    by_key: dict[str, Callable[..., object]] = {}
    recorded, externals = session._external_view()
    for node_id, (fn, _inputs) in externals.items():
        by_key[external_key(recorded[node_id])] = fn
    wired: dict[str, Callable[..., object]] = {}
    for node in GraphStore.deserialize(bytes(compiled.ir)).nodes():
        if node["kind"] == "external":
            key = external_key(node)
            evaluator = by_key.get(key)
            if evaluator is not None:
                wired[key] = evaluator
    return wired


def shipped_evaluators(
    session: Session, compiled: CompiledGraph
) -> tuple[tuple[str, Callable[..., object]], ...]:
    """Return :func:`external_evaluators` for a plan runners pickle with stdlib pickle.

    Each evaluator is wrapped by :func:`by_value`.
    """
    return tuple((key, by_value(fn)) for key, fn in external_evaluators(session, compiled).items())


def plan_services(
    session: Session, compiled: CompiledGraph, names: Sequence[str] | None = None
) -> tuple[ServiceSpec, ...]:
    """Return the ``Plan.services`` of a plan over ``compiled``.

    The session's specs its External nodes name, plus ``names``.
    """
    return referenced_services(session, GraphStore.deserialize(bytes(compiled.ir)).nodes(), names)


def _max(a: Any, b: Any) -> Any:
    """``max``, where NaN wins as it does in ``ak.max`` and ``np.max``."""
    return b if b != b or a < b else a


def _min(a: Any, b: Any) -> Any:
    """``min``, where NaN wins as it does in ``ak.min`` and ``np.min``."""
    return b if b != b or b < a else a


@dataclass(frozen=True)
class Fold:
    """How a whole-dataset reduction's chunk values fold into its value over the dataset.

    ``empty`` is the value of no chunks. ``None`` is a chunk with nothing to reduce (``ak.max`` of
    no rows), so it leaves the other side as it is.
    """

    combine: Callable[[Any, Any], Any]
    empty: Any = None

    def __call__(self, a: Any, b: Any) -> Any:
        """Fold two chunk values."""
        if a is None or b is None:
            return b if a is None else a
        return self.combine(a, b)


_FOLDS = {
    "sum": Fold(operator.add, 0),
    "count": Fold(operator.add, 0),
    "count_nonzero": Fold(operator.add, 0),
    "any": Fold(operator.or_, False),
    "all": Fold(operator.and_, True),
    "max": Fold(_max),
    "min": Fold(_min),
}

#: why these reductions' chunk values do not fold
_NOT_FOLDABLE = {
    "mean": "a mean of chunk means is not the dataset's mean",
    "var": "a variance of chunk variances is not the dataset's variance",
    "std": "a std of chunk stds is not the dataset's std",
}


def _fold_of(node: Mapping[str, Any]) -> Fold:
    kind, op, params = node["kind"], node["name"], node["params"]
    if kind != "reduction":
        raise TypeError(f"is {'per-event' if kind == 'op' else kind}, not a reduction")
    name = str(op).rsplit(".", 1)[-1]
    if name in _NOT_FOLDABLE:
        raise TypeError(f"is {op}: {_NOT_FOLDABLE[name]}")
    if name not in _FOLDS:
        raise TypeError(f"is {op}, which has no known combine")
    if params.get("axis", "none") != "none" or params.get("keepdims"):
        raise TypeError(f"is {op} with axis= or keepdims=, not one value for the dataset")
    return replace(_FOLDS[name], empty=None) if params.get("mask_identity") else _FOLDS[name]


def fold_of(array: Array) -> Fold:
    """Return how ``array``, a whole-dataset reduction, folds across partitions.

    Those are ``sum``, ``count``, ``count_nonzero``, ``any``, ``all``, ``min`` and ``max`` with no
    axis. Anything else raises a ``TypeError`` completing the sentence ``output 'x' ...``.
    """
    return _fold_of(array.session._nodes()[array.node_id])


@dataclass(frozen=True)
class _Folded:
    """``aggregate_plan``'s default reduction: one fold per output, read at its compiled slot."""

    slots: tuple[int, ...]
    folds: tuple[Fold, ...]
    then: Callable[[Any], Any] | None = None

    def reduce(self, values: list[Any]) -> tuple[Any, ...]:
        return tuple(values[slot] for slot in self.slots)

    def combine(self, a: tuple[Any, ...], b: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(fold(x, y) for fold, x, y in zip(self.folds, a, b, strict=True))

    def empty(self) -> tuple[Any, ...]:
        return tuple(fold.empty for fold in self.folds)

    def finalize(self, value: tuple[Any, ...]) -> Any:
        out = value[0] if len(value) == 1 else value
        return out if self.then is None else self.then(out)


def _folded(outputs: Sequence[Array], slot: Callable[[Array], int]) -> _Folded:
    """Return the default reduction of ``outputs``; refuse an output no fold covers."""
    nodes = outputs[0].session._nodes()
    folds = []
    for i, output in enumerate(outputs):
        try:
            folds.append(_fold_of(nodes[output.node_id]))
        except TypeError as err:
            raise TypeError(
                f"aggregate_plan: output {i} {err}; pass reduce=, combine=, empty="
            ) from err
    return _Folded(tuple(slot(o) for o in outputs), tuple(folds))


def aggregate_plan(
    *outputs: Array,
    reduce: Callable[[list[Any]], Any] | None = None,
    combine: Callable[[Any, Any], Any] | None = None,
    empty: Callable[[], Any] | None = None,
    externals: Mapping[str, Callable[..., object]] | None = None,
    backend: Callable[[], Any] | str | None = None,
    steps_per_file: int | None = None,
    events_per_task: int | None = None,
    partitions: Sequence[Partition] | None = None,
    on_compiled: Callable[[CompiledGraph], Any] | None = None,
    store: str | os.PathLike[str] | None = None,
    storage_options: Mapping[str, Any] | None = None,
    services: Sequence[str] | None = None,
    writes: Sequence[PartWrite] = (),
    opt_level: int = 1,
    finalize: Callable[[Any], Any] | None = None,
    read_by: Sequence[Array] | None = None,
) -> Plan[Any] | DurablePlanV2:
    """Build a one-pass partition-wise reduction :class:`~graphed.core.execution.Plan`.

    It runs over the session's single partitioned source (see module docstring). ``outputs`` are
    the output Arrays (their shared sub-graph is compiled to one IR and evaluated once per
    partition); ``externals`` binds any External payload evaluator; ``backend`` is the workers'
    evaluation backend (factory, class, or ``"module:attr"`` ref; defaults to the session backend's
    class, with its behavior). ``run(plan).value`` is the ``reduce``+``combine`` aggregate over
    all partitions.

    With none of ``reduce``, ``combine`` and ``empty``, every output must be a whole-dataset
    reduction with a known combine (:func:`fold_of`), and ``run(plan).value`` is its value, or a
    tuple of them for several outputs; ``finalize`` then takes that value. A reduction with no
    value for no rows (``ak.min``, ``ak.max``, ``mask_identity=True``) skips such a chunk, and is
    ``None`` with no rows at all. A float sum's bits are fixed by the runner family and the
    partition count, as any float reduction's are, and can differ from
    ``graphed.materialize``'s in the last places. Anything else is refused, naming the output, as
    a mean of chunk means is not the dataset's mean, and so is ``writes``.

    The tasks are ``partitions`` when given, else the source's partitions at ``steps_per_file``
    steps per file; ``None`` takes the source's own default (``from_parquet(steps_per_file=)``).
    ``events_per_task`` instead cuts each file into tasks of that many entries, which needs a
    source that can count them (``num_entries(partition)``); a blind source then opens each file on
    the driver, one after another, to count it.

    ``on_compiled`` is the seam onto the internally compiled :class:`CompiledGraph` — the
    artifact is otherwise unreachable from the caller. It fires ONCE, and whatever it returns is
    carried onto the shipped closure's ``variation_labels``.

    ``store`` (a directory, or an fsspec URL) makes each task capture its input chunk and its
    ``reduce`` partial into that checkpoint root, so :func:`graphed.debug.replay` can re-run a task
    of this plan later from exactly what it read. A directory must be one every worker shares.
    ``storage_options`` pass through to the URL's filesystem (credentials, endpoint, ...).

    ``Plan.services`` holds the session's specs named by the compiled External nodes'
    ``params["service"]`` and by ``services`` (names a node does not carry, e.g. a service the
    ``reduce`` calls), in name order.

    ``writes`` (:class:`~graphed.write.PartWrite`) write one part per task from the same read and
    evaluation: ``reduce`` then receives the outputs' values as without writes, followed by one
    part path per write. ``outputs`` may be empty when ``writes`` is not.

    ``opt_level`` picks the compile. ``1``, the default, optimizes: equal outputs may merge, and
    ``reduce`` then receives one value for them. ``0`` ships the 1:1 cone of the outputs, the
    written arrays and the metadata arrays, every recorded operation its own node (the graph
    :func:`graphed.debug.lower` calls ``opt_level=0``), so ``reduce`` receives one value per
    distinct output. A ``StageError`` reports the level, and :func:`graphed.debug.replay`
    recompiles at it. Any other value is refused.

    ``finalize`` becomes ``Plan.finalize``: the runner calls it once, on the head, on the combined
    value (``empty()``'s too), and ``run(plan).value`` is what it returns.

    ``read_by`` are the Arrays whose columns the read covers, in place of those the plan evaluates,
    for a caller using the chunks otherwise.

    Outputs past a :func:`graphed.join` read two sources, so the plan is the join's
    :class:`~graphed.core.DurablePlanV2`: a stage per side and one per destination, with ``reduce``
    per destination. It runs on ``SequentialRunner`` and ``SubmitRunner``, and has no ``tasks`` or
    ``process``; ``isinstance(plan, graphed.core.Plan)`` tells the two apart.

    >>> import tempfile, awkward as ak
    >>> import graphed
    >>> from graphed.awkward import from_parquet, gak
    >>> from graphed.core import SequentialRunner
    >>> root = tempfile.mkdtemp()
    >>> for i, met in enumerate([[10.0, 55.0], [30.0]]):
    ...     _ = ak.to_parquet(ak.Array({"MET_pt": met}), f"{root}/part{i}.parquet")
    >>> events = from_parquet("events", root)
    >>> plan = graphed.aggregate_plan(gak.sum(events.MET_pt))
    >>> len(plan.tasks), float(SequentialRunner().run(plan).value)
    (2, 95.0)
    """
    writes = tuple(writes)
    if reduce is None and combine is None and empty is None:
        if writes:
            raise TypeError("aggregate_plan(writes=) needs reduce=, combine= and empty=")
    elif reduce is None or combine is None or empty is None:
        raise TypeError("aggregate_plan takes reduce=, combine= and empty= together, or none")
    arrays = _planned_arrays(outputs, writes, store=store, opt_level=opt_level)
    session = arrays[0].session
    compiled = _compile_at(opt_level, session, arrays)
    partitioned = _partitioned(arrays)
    over_join = len(partitioned) == 2 and _has_join(compiled)
    if over_join:
        for name, given in (
            ("partitions", partitions),
            ("events_per_task", events_per_task),
            ("store", store),
            ("writes", writes or None),
            ("read_by", read_by),
        ):
            if given is not None:
                raise TypeError(f"aggregate_plan over a join takes no {name}=")
        if opt_level != 1:
            raise TypeError("aggregate_plan over a join takes no opt_level=0")
    else:
        nid, data = _one_source(session, outputs, writes, partitioned)
    slot = _slot_of(compiled)
    written = {compiled.correspondence.node_map[w.array.node_id][0] for w in writes}
    refuse_chunk_partials(compiled, as_outputs=written)
    if reduce is None or combine is None or empty is None:
        folded = _folded(outputs, slot)
        reduce, combine, empty = folded.reduce, folded.combine, folded.empty
        # the workers' copy carries no finalize: only the head calls it
        finalize = replace(folded, then=finalize and by_value(finalize)).finalize
    if over_join:
        from .shuffle import _build  # shuffle imports this module

        return _build(
            "join_plan",
            outputs,
            (reduce, combine, empty),
            backend=backend,
            steps_per_file=steps_per_file,
            names=services,
            compiled=compiled,
            externals=externals,
            on_compiled=on_compiled,
            finalize=finalize,
        )
    if events_per_task is not None:
        if steps_per_file is not None or partitions is not None:
            raise TypeError(
                "events_per_task= chunks the plan; omit steps_per_file= and partitions="
            )
        partitions = _event_partitions(data, events_per_task)
    elif partitions is None:
        partitions = data.partitions(steps_for(data, steps_per_file))
    # Wire EVERY External surviving in the compiled IR from the session (fills + upstream
    # corrections alike); an explicit `externals=` (keyed by `external_key`) overrides the
    # auto-wired evaluator.
    wired = dict(shipped_evaluators(session, compiled))
    if externals:
        wired.update((key, by_value(fn)) for key, fn in externals.items())
    read = arrays if read_by is None else read_by
    declared = declared_columns(data, read)
    factory = worker_backend(session, backend)
    process = _PartitionReduce(
        ir=bytes(compiled.ir),
        source_name=session.source_name(nid),
        backend_factory=factory,
        reader=data,
        columns=read_columns(read, nid) if declared is None else declared,
        externals=tuple(wired.items()),
        reduce=by_value(reduce),
        variation_labels=None if on_compiled is None else on_compiled(compiled),
        frames=compiled.correspondence.frames,
        store=None if store is None else os.fspath(store),
        storage_options=tuple((storage_options or {}).items()),
        writes=tuple(_shipped(w, slot) for w in writes),
        # outputs are marked first, so their distinct values lead the evaluated list
        n_values=len({slot(o) for o in outputs}) if writes else None,
        opt_level=opt_level,
    )
    tasks = tuple(Task(i, p) for i, p in enumerate(partitions))
    _refuse_shared_parts((process, t.partition) for t in tasks)
    return Plan(
        process=process,
        combine=WarnedCombine(_adopting(by_value(combine), factory)),
        empty=_adopting(by_value(empty), factory),
        tasks=tasks,
        services=plan_services(session, compiled, services),
        finalize=WarnedFinalize(None if finalize is None else by_value(finalize)),
    )


def _event_partitions(source: PartitionedSource, events_per_task: int) -> list[Partition]:
    """Cut each of ``source``'s files into ranges of ``events_per_task`` entries."""
    if events_per_task < 1:
        raise ValueError(f"events_per_task must be >= 1, got {events_per_task}")
    count = getattr(source, "num_entries", None)
    out: list[Partition] = []
    for file in source.partitions(1):
        if file.blind_step is not None:
            if count is None:
                raise TypeError(
                    f"{type(source).__name__} cannot count its entries, so events_per_task= "
                    "cannot chunk it; pass steps_per_file= instead"
                )
            file = file.resolve(count(file))
        start, stop = file.entry_start, file.entry_stop
        # an empty file keeps one task, as eager_ranges keeps one step for it
        out.extend(
            Partition(file.uri, file.tree, a, min(a + events_per_task, stop))
            for a in range(start, max(stop, start + 1), events_per_task)
        )
    return out


def _planned_arrays(
    outputs: Sequence[Array], writes: tuple[PartWrite, ...], *, store: object, opt_level: int
) -> list[Array]:
    """Refuse what :func:`aggregate_plan` cannot plan; return every Array its tasks evaluate."""
    if type(opt_level) is not int or opt_level not in (0, 1):
        raise ValueError(
            "aggregate_plan(opt_level=) is 0 (the outputs' 1:1 cone) or 1 (optimized), not "
            f"{opt_level!r}"
        )
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
        raise TypeError(
            "all outputs of one plan must record into one session; collate one plan per Session"
            " (graphed.collate), or record them in one (`with output.session:` around the later"
            " reads)"
        )
    return arrays


def _partitioned(arrays: Sequence[Array]) -> dict[int, PartitionedSource]:
    return {nid: d for nid, d in sources_read(arrays).items() if isinstance(d, PartitionedSource)}


def _has_join(compiled: CompiledGraph) -> bool:
    return any(n["kind"] == "join" for n in GraphStore.deserialize(bytes(compiled.ir)).nodes())


def _one_source(
    session: Session,
    outputs: Sequence[Array],
    writes: Sequence[PartWrite],
    partitioned: Mapping[int, PartitionedSource],
) -> tuple[int, PartitionedSource]:
    """Return the one partitioned source in ``partitioned``, with its node id."""
    if len(partitioned) != 1:
        names = sorted(session.source_name(nid) for nid in partitioned)
        listed = f" ({', '.join(names)})" if names else ""
        # collate splits whole outputs and writes; one name is one dataset, read twice it refuses.
        units = [[o] for o in outputs] + [
            _planned_arrays((), (w,), store=None, opt_level=1) for w in writes
        ]
        if (
            names
            and len(set(names)) == len(names)
            and all(len(_partitioned(unit)) < 2 for unit in units)
        ):
            listed += "; collate one plan per source (graphed.collate)"
        raise TypeError(
            "aggregate_plan needs exactly one partitioned source; the outputs read "
            f"{len(partitioned)}{listed}"
        )
    ((nid, data),) = partitioned.items()
    return nid, data


def _shipped(w: PartWrite, slot: Callable[[Array], int]) -> _ShippedWrite:
    """Return ``w`` as a task ships it, its arrays replaced by their output slots."""
    kv = (
        None
        if w.metadata is None
        else tuple((k, slot(v) if isinstance(v, Array) else str(v)) for k, v in w.metadata.items())
    )
    if w.storage_options:
        return (
            w.codec,
            w.destination,
            w.name,
            slot(w.array),
            kv,
            tuple(w.storage_options.items()),
        )
    return (w.codec, w.destination, w.name, slot(w.array), kv)


def _written_parts(process: object, partition: Partition) -> Sequence[str]:
    """Return the parts a plan's ``process`` writes for ``partition``.

    They come from its optional ``part_paths(partition)`` hook (no I/O); a process without the hook
    is not checked.
    """
    hook = getattr(process, "part_paths", None)
    return () if hook is None else hook(partition)


def _refuse_shared_parts(runs: Iterable[tuple[object, Partition]]) -> None:
    """Refuse a part two tasks share, which would be silently overwritten.

    Driver-side, one hook call per task.
    """
    seen: set[str] = set()
    for process, partition in runs:
        for path in map(os.path.normpath, _written_parts(process, partition)):
            if path in seen:
                raise ValueError(
                    f"two writes of this plan write the same part {path!r}: `name` must tell "
                    "every partition apart (a blind partition's entry range is 0-0 until it is "
                    "read), and plans collated together need distinct parts"
                )
            seen.add(path)


@dataclass(frozen=True)
class _Collated:
    """A collated plan's process.

    The task's ``(uri, tree)`` picks the sub-plan whose graph reads it.
    """

    @property
    def checkpointable(self) -> bool:
        return all(getattr(fn, "checkpointable", False) for fn in self.processes.values())

    processes: Mapping[str, Callable[[Partition, WorkerResources], Any]]
    #: O(files), never O(tasks): it ships once per worker, not in any task
    route: Mapping[tuple[str, str], str]

    def __call__(self, partition: Partition, resources: WorkerResources) -> Any:
        name = self.route[(partition.uri, partition.tree)]
        # what a partial carries stays on the outside, where a store and a finalize look for it
        return on_value(lambda value: {name: value}, self.processes[name](partition, resources))

    def summary(self) -> str:
        """Say what the plan's repr shows of this process: each output, else each plan's name."""
        named = [
            output
            for name, fn in self.processes.items()
            for output in getattr(fn, "output_summaries", list)() or [name]
        ]
        return f"outputs: {capped(named)}"

    def part_paths(self, partition: Partition) -> Sequence[str]:
        return _written_parts(
            self.processes[self.route[(partition.uri, partition.tree)]], partition
        )

    def bind_services(self, endpoints: Mapping[str, str]) -> _Collated:
        return replace(self, processes=dict(bind_externals(self.processes.items(), endpoints)))

    def resolve_services(self, value: Mapping[str, Any]) -> dict[str, Any]:
        """Each name's value resolved by its process when it takes it.

        A name absent from ``value`` is not called.
        """
        return {
            name: fn.resolve_services(value[name])
            if isinstance(fn := self.processes.get(name), Resolvable)
            else value[name]
            for name in value
        }


@dataclass(frozen=True)
class _CollatedFinalize:
    """Each name's value through its sub-plan's finalize; a name without one passes through."""

    finalizers: Mapping[str, Callable[[Any], Any]]

    def __call__(self, value: dict[str, Any]) -> dict[str, Any]:
        return {n: f(v) if (f := self.finalizers.get(n)) else v for n, v in value.items()}


@dataclass(frozen=True)
class _CollatedCombine:
    """Per name with that sub-plan's combine; a name on one side only passes through."""

    combines: Mapping[str, Callable[[Any, Any], Any]]

    def __call__(self, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        merged = {**a, **{n: self.combines[n](a[n], v) if n in a else v for n, v in b.items()}}
        # mapping order, whatever the tree
        return {n: merged[n] for n in self.combines if n in merged}


class _RaisingReduce(_PartitionReduce[V]):
    """A task that raises where numpy would warn and leave an inf or a NaN.

    It keys as the plain task: one that fired left no result, so a kept result is the same.
    """

    def floating_point(self) -> contextlib.AbstractContextManager[object]:
        """Trap what numpy would warn about and leave as an inf or a NaN."""
        import numpy as np

        # not underflow: rounding to zero is ordinary, and gives no inf or NaN
        return np.errstate(divide="raise", over="raise", invalid="raise")

    def _keyed_as(self) -> _PartitionReduce[V]:
        return _PartitionReduce(**vars(self))


def _raising(process: Any) -> Any:
    if isinstance(process, _PartitionReduce):
        return _RaisingReduce(**vars(process))
    if isinstance(process, _Collated):
        return replace(process, processes={n: _raising(p) for n, p in process.processes.items()})
    raise TypeError(
        "raise_on_floating_point takes a plan from aggregate_plan, collate or a histogram's "
        f"to_plan, before resumable or histserv wraps it; got a {type(process).__name__} process"
    )


def raise_on_floating_point(plan: Plan[V] | DurablePlanV2) -> Plan[V]:
    """Return ``plan``, its tasks raising on a division by zero, an overflow or an invalid value.

    What numpy would warn about and turn into an inf or a NaN becomes a ``StageError`` at the line
    that recorded the operation, naming the chunk, on any runner. Underflow still rounds to zero.
    A task's key does not change: results a store holds for ``plan`` are served, unchecked. A
    plan over a join runs in stages and is refused.
    """
    if isinstance(plan, DurablePlanV2):
        raise TypeError(
            "raise_on_floating_point needs a plan with one task kind; this runs in stages"
        )
    return replace(plan, process=_raising(plan.process))


def collate(plans: Mapping[str, Plan[Any] | DurablePlanV2]) -> Plan[dict[str, Any]]:
    """ONE executable plan over several plans' tasks.

    Plans over different sources or graphs (data and MC, say) — whose value is ``{name: that plan's
    value}``.

    Tasks are each plan's in key order, concatenated in mapping order and re-keyed ``0..N-1``, so
    the reduction tree stays the runner's. A task runs the process of the plan that holds its
    ``(uri, tree)``; a ``(uri, tree)`` held by two plans is refused (record both over one source so
    they share the read), and so is a part two tasks would both write, across plans too (named by
    each process's optional ``part_paths(partition)`` hook; a process without it is not checked).
    A name is in the value exactly when its plan has at least one task. ``Plan.services`` is the
    union of the plans' services; one name declared as two different specs is refused. Each name's
    value passes through its plan's ``finalize``, if it has one. Running each plan on its own and
    collecting ``{name: value}`` gives the same product when each ``combine`` is exact. A plan that
    runs in stages, is adaptive or can stop early has no fixed tasks and is refused.

    >>> import tempfile, awkward as ak
    >>> import graphed
    >>> from graphed.awkward import from_parquet, gak
    >>> from graphed.core import SequentialRunner
    >>> root = tempfile.mkdtemp()
    >>> _ = ak.to_parquet(ak.Array({"MET_pt": [10.0, 55.0]}), f"{root}/mc.parquet")
    >>> _ = ak.to_parquet(ak.Array({"MET_pt": [30.0]}), f"{root}/data.parquet")
    >>> def met_sum(name):
    ...     events = from_parquet(name, f"{root}/{name}.parquet")
    ...     return graphed.aggregate_plan(gak.sum(events.MET_pt))
    >>> plans = {name: met_sum(name) for name in ("mc", "data")}
    >>> {k: float(v) for k, v in SequentialRunner().run(graphed.collate(plans)).value.items()}
    {'mc': 65.0, 'data': 30.0}
    """
    if not plans:
        raise ValueError("collate needs at least one plan")
    services: dict[str, ServiceSpec] = {}
    for name, plan in plans.items():
        for spec in plan.services:
            if services.setdefault(spec.name, spec) != spec:
                raise ValueError(
                    f"plan {name!r} declares service {spec.name!r} unlike another plan does"
                )
    route: dict[tuple[str, str], str] = {}
    tasks: list[Task] = []
    fixed: dict[str, Plan[Any]] = {}
    for name, plan in plans.items():
        if not isinstance(plan, Plan):
            raise TypeError(f"collate needs plans with fixed tasks; {name!r} runs in stages")
        if plan.next_tasks is not None or plan.stop is not None:
            raise TypeError(
                f"collate needs plans with fixed tasks; {name!r} is adaptive or stoppable"
            )
        fixed[name] = plan
        for task in sorted(plan.tasks, key=lambda t: t.key):
            key = (task.partition.uri, task.partition.tree)
            owner = route.setdefault(key, name)
            if owner != name:
                raise ValueError(
                    f"{key!r} appears in plans {owner!r} and {name!r}; record both over one "
                    "source so one read serves both graphs"
                )
            tasks.append(Task(len(tasks), task.partition))
    process = _Collated({n: p.process for n, p in fixed.items()}, route)
    _refuse_shared_parts((process, t.partition) for t in tasks)
    finalizers = {n: p.finalize for n, p in fixed.items() if p.finalize is not None}
    return Plan(
        process=process,
        combine=WarnedCombine(_CollatedCombine({n: p.combine for n, p in fixed.items()})),
        empty=dict,
        tasks=tuple(tasks),
        open_once=any(p.open_once for p in fixed.values()),
        services=tuple(services[n] for n in sorted(services)),
        finalize=WarnedFinalize(_CollatedFinalize(finalizers) if finalizers else None),
    )
