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

from graphed.core import Partition
from graphed.core.execution import Plan, Task, WorkerResources

from .array import Array
from .execute import compile_ir, evaluate_ir
from .projection import read_columns
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

    def __call__(self, partition: Partition, resources: WorkerResources) -> V:
        chunk = self.reader.read_partition(partition, self.columns, resources)
        values = evaluate_ir(
            self.ir,
            resolve_backend(self.backend_factory),
            {self.source_name: chunk},
            externals=dict(self.externals),
        )
        return self.reduce(values)


def aggregate_plan(
    *outputs: Array,
    reduce: Callable[[list[Any]], V],
    combine: Callable[[V, V], V],
    empty: Callable[[], V],
    externals: Mapping[str, Callable[..., object]] | None = None,
    backend: Callable[[], Any] | str | None = None,
    steps_per_file: int = 1,
    partitions: Sequence[Partition] | None = None,
) -> Plan[V]:
    """Build a one-pass partition-wise reduction :class:`~graphed.core.execution.Plan` over the
    session's single partitioned source (see module docstring). ``outputs`` are the output Arrays
    (their shared sub-graph is compiled to one IR and evaluated once per partition); ``externals``
    binds any External payload evaluator; ``backend`` is the workers' evaluation backend (factory,
    class, or ``"module:attr"`` ref; defaults to the session backend's type). ``run(plan).value`` is
    the ``reduce``+``combine`` aggregate over all partitions."""
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
    process = _PartitionReduce(
        ir=bytes(compiled.ir),
        source_name=session.source_name(nid),
        backend_factory=backend if backend is not None else type(session.backend),
        reader=data,
        columns=read_columns(list(outputs), nid),
        externals=tuple((externals or {}).items()),
        reduce=reduce,
    )
    if partitions is None:
        partitions = data.partitions(steps_per_file)
    tasks = tuple(Task(i, p) for i, p in enumerate(partitions))
    return Plan(process=process, combine=combine, empty=empty, tasks=tasks)
