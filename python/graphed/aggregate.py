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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
from .session import Session
from .varied import refuse_container
from .write import PartitionedSource, declared_columns

V = TypeVar("V")


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
    #: for `graphed.debug.replay`.
    store: str | None = None

    def __call__(self, partition: Partition, resources: WorkerResources) -> V:
        chunk = self.reader.read_partition(partition, self.columns, resources)
        if self.store is None:
            # _evaluate inlined: the default path keeps its pre-capture frame count
            return self.reduce(
                evaluate_ir(
                    self.ir,
                    resolve_backend(self.backend_factory),
                    {self.source_name: chunk},
                    externals=dict(self.externals),
                    on_failure=self._attribute(str(partition)),
                )
            )
        from graphed.checkpoint import PickleCodec  # noqa: PLC0415  (only a capturing plan needs it)

        store = self._open_store(f"{os.getpid()}-{threading.get_ident()}")
        cid, label, codec = self._capture_id(partition), str(partition), PickleCodec()
        # the input is kept before evaluating, so a failing task's input survives it
        store.record_done(f"{cid}:input", label, store.put(codec.encode(chunk)), stage="replay-input")
        result = self.reduce(self._evaluate(chunk, partition))
        store.record_done(f"{cid}:output", label, store.put(codec.encode(result)), stage="replay-output")
        return result

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
    of this plan later from exactly what it read. A directory must be one every worker shares."""
    refuse_container("graphed.aggregate_plan", *outputs)
    if not outputs:
        raise ValueError("aggregate_plan needs at least one output Array")
    session = outputs[0].session
    if any(o.session is not session for o in outputs):
        raise TypeError("all outputs of one plan must record into one session")
    partitioned = {nid: d for nid, d in session.sources().items() if isinstance(d, PartitionedSource)}
    if len(partitioned) != 1:
        raise TypeError(
            f"aggregate_plan needs exactly one partitioned source; this session has {len(partitioned)}"
        )
    ((nid, data),) = partitioned.items()
    compiled = compile_ir(session, *outputs)
    refuse_chunk_partials(compiled, as_outputs=False)
    # Wire EVERY External surviving in the compiled IR from the session (fills + upstream corrections
    # alike); an explicit `externals=` (keyed by `external_key`) overrides the auto-wired evaluator.
    wired = external_evaluators(session, compiled)
    if externals:
        wired.update(externals)
    declared = declared_columns(data, outputs)
    process = _PartitionReduce(
        ir=bytes(compiled.ir),
        source_name=session.source_name(nid),
        backend_factory=backend if backend is not None else type(session.backend),
        reader=data,
        columns=read_columns(list(outputs), nid) if declared is None else declared,
        externals=tuple(wired.items()),
        reduce=reduce,
        variation_labels=None if on_compiled is None else on_compiled(compiled),
        frames=compiled.correspondence.frames,
        store=None if store is None else os.fspath(store),
    )
    if partitions is None:
        partitions = data.partitions(steps_per_file)
    tasks = tuple(Task(i, p) for i, p in enumerate(partitions))
    return Plan(process=process, combine=combine, empty=empty, tasks=tasks)
