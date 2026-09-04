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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from graphed.core import GraphStore, Partition
from graphed.core.execution import Plan, Task, WorkerResources

from .array import Array
from .errors import GraphedError
from .execute import CompiledGraph, Key, OnFailure, compile_ir, evaluate_ir, external_key
from .projection import read_columns
from .session import Session
from .varied import refuse_container
from .write import PartitionedSource

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

    def __call__(self, partition: Partition, resources: WorkerResources) -> V:
        chunk = self.reader.read_partition(partition, self.columns, resources)
        values = evaluate_ir(
            self.ir,
            resolve_backend(self.backend_factory),
            {self.source_name: chunk},
            externals=dict(self.externals),
            on_failure=self._attribute(str(partition)),
        )
        return self.reduce(values)

    def _attribute(self, partition: str) -> OnFailure | None:
        """§8.2(ii): the worker-side wrap. A RAW failure at a key the label channel has an ENTRY for
        becomes a `StageError` carrying that key's label and the user's line; a key with no entry —
        and a closure with no channel at all — re-raises the original untouched, since `StageError`
        needs frames at construction and there are none to build one from."""
        entries = dict(self.variation_labels or ())
        if not entries:
            return None
        from .debug.errors import SourceFrame, StageError  # noqa: PLC0415  (import cycle)

        def attribute(key: Key, op: str, ins: list[object], exc: BaseException) -> BaseException | None:
            # §8.2(ii): "a `GraphedError` re-raises untouched on EVERY arm regardless of entry — it
            # is already an attributed error, and §6.1d's blame parity (the plan path re-raises the
            # guard's message verbatim) binds it".
            if isinstance(exc, GraphedError):
                return None
            entry = entries.get(key)
            if entry is None:
                return None
            labels, frame = entry
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


def _external_evaluators(session: Session, compiled: CompiledGraph) -> dict[str, Callable[..., object]]:
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
) -> Plan[V]:
    """Build a one-pass partition-wise reduction :class:`~graphed.core.execution.Plan` over the
    session's single partitioned source (see module docstring). ``outputs`` are the output Arrays
    (their shared sub-graph is compiled to one IR and evaluated once per partition); ``externals``
    binds any External payload evaluator; ``backend`` is the workers' evaluation backend (factory,
    class, or ``"module:attr"`` ref; defaults to the session backend's type). ``run(plan).value`` is
    the ``reduce``+``combine`` aggregate over all partitions.

    ``on_compiled`` is §7.2's seam onto the internally compiled :class:`CompiledGraph` — the
    artifact is otherwise unreachable from the caller. It fires ONCE, and whatever it returns is
    carried onto the shipped closure's ``variation_labels``."""
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
    # Wire EVERY External surviving in the compiled IR from the session (fills + upstream corrections
    # alike); an explicit `externals=` (keyed by `external_key`) overrides the auto-wired evaluator.
    wired = _external_evaluators(session, compiled)
    if externals:
        wired.update(externals)
    process = _PartitionReduce(
        ir=bytes(compiled.ir),
        source_name=session.source_name(nid),
        backend_factory=backend if backend is not None else type(session.backend),
        reader=data,
        columns=read_columns(list(outputs), nid),
        externals=tuple(wired.items()),
        reduce=reduce,
        variation_labels=None if on_compiled is None else on_compiled(compiled),
    )
    if partitions is None:
        partitions = data.partitions(steps_per_file)
    tasks = tuple(Task(i, p) for i, p in enumerate(partitions))
    return Plan(process=process, combine=combine, empty=empty, tasks=tasks)
