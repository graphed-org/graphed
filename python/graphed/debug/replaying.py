"""Replay one task of an ``aggregate_plan`` run on the driver, one operation at a time (opt_level=0).

``replay(plan, key, *outputs)`` re-executes the task with the input it read — from the plan's
capture root (``aggregate_plan(store=)``) when the run kept it, else by re-reading its partition —
through the unfused graph, so every step carries the user's line. A failing step raises the
``StageError`` M6's ``run`` would; ``diff()`` compares the replayed partial with the recorded one.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from graphed import Array
from graphed.aggregate import _PartitionReduce, resolve_backend
from graphed.core import GraphStore, LocalResources
from graphed.core.execution import Plan, Task
from graphed.execute import _unbound_external, _unbound_source, compile_ir, external_key

from .lowering import LoweredOp, lower
from .runner import _stage_error


@dataclass(frozen=True)
class Step:
    """One replayed operation: its lowered op (with the user's frame), its value, and how long it took."""

    node: LoweredOp
    value: Any
    seconds: float


@dataclass(frozen=True)
class ReplayDiff:
    """The replayed partial against the reference: ``"recorded"`` (the run's captured output) or
    ``"re-evaluated"`` (the plan's optimized graph run again on the same input)."""

    reference: str
    equal: bool
    recorded: Any
    replayed: Any


def _decode(store: Any, blob: str) -> Any:
    from graphed.checkpoint import PickleCodec  # noqa: PLC0415  (workers import graphed.debug)

    data = store.get(blob)
    if data is None:
        raise FileNotFoundError(f"capture blob {blob} is missing or corrupt")
    return PickleCodec().decode(data)


def _equal(a: Any, b: Any) -> bool:
    """By value: dicts by key, sequences by position, leaves by ``numpy.array_equal`` (which
    dispatches to awkward and answers False on a length or shape mismatch). NaN is unequal."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(map(_equal, a, b))
    import numpy  # noqa: PLC0415  (graphed.debug loads no numpy)

    return bool(numpy.array_equal(a, b))


class Replay:
    """One task of a run, ready to replay. ``steps()`` is lazy; ``value`` runs it once and caches."""

    def __init__(self, process: _PartitionReduce[Any], task: Task, outputs: tuple[Array, ...]) -> None:
        self.task = task
        self._process = process
        self._session = outputs[0].session
        self._output_ids = [o.node_id for o in outputs]
        cone = {op.node_id: op for o in outputs for op in lower(self._session, o, opt_level=0).ops}
        self._cone = [cone[i] for i in sorted(cone)]
        # the unfused IR keeps record ids; it is the whole arena, so only the cone is evaluated
        arena = GraphStore.deserialize(bytes(compile_ir(self._session, *outputs, optimize=False).ir))
        self._nodes = [n for n in arena.nodes() if n["id"] in cone]
        self._store: Any = None
        self._captured: dict[str, str] = {}
        if process.store is not None:
            self._store = process._open_store()
            cid = process._capture_id(task.partition)
            done = self._store.completed()
            self._captured = {k: done[f"{cid}:{k}"].blob for k in ("input", "output") if f"{cid}:{k}" in done}
        self.input_source = "store" if "input" in self._captured else "re-read"

    @cached_property
    def _chunk(self) -> Any:
        if "input" in self._captured:
            return _decode(self._store, self._captured["input"])
        return self._process.reader.read_partition(
            self.task.partition, self._process.columns, LocalResources()
        )

    def steps(self) -> Iterator[Step]:
        """Evaluate the task's opt_level=0 cone node by node, yielding each :class:`Step`. The input is
        read before the first step, and a failed read raises as the run's own read would."""
        from graphed.preserve.interpreter import iter_ir  # noqa: PLC0415  (workers import graphed.debug)

        chunk = self._chunk
        source_name = self._process.source_name
        externals = dict(self._process.externals)

        # bound as evaluate_ir binds them for the run, so an unbound name fails replay as it failed the task
        def source(node: dict[str, Any]) -> Any:
            if node["name"] != source_name:
                raise _unbound_source(node["name"])
            return chunk

        def external(node: dict[str, Any], ins: list[Any]) -> Any:
            chash = node["descriptor"]["content_hash"]
            fn = externals.get(external_key(node)) or externals.get(chash)
            if fn is None:
                raise _unbound_external(chash)
            return fn(*ins)

        values = iter_ir(
            self._nodes,
            source=source,
            external=external,
            eval_op=resolve_backend(self._process.backend_factory).eval_stage,
        )
        for op in self._cone:
            start = time.perf_counter()
            try:
                _nid, value = next(values)
            except Exception as exc:
                failing = lower(self._session, Array(self._session, op.node_id), opt_level=0)
                raise _stage_error(failing, op.node_id, str(self.task.partition), exc) from exc
            yield Step(op, value, time.perf_counter() - start)

    @cached_property
    def value(self) -> Any:
        """The replayed partial: the plan's ``reduce`` over the replayed output values."""
        out = {s.node.node_id: s.value for s in self.steps() if s.node.node_id in self._output_ids}
        return self._process.reduce([out[i] for i in self._output_ids])

    def diff(self) -> ReplayDiff:
        """Compare the replayed partial with the run's recorded one, or, when the run kept no output
        for this task, with the plan's optimized graph evaluated on the same input."""
        replayed = self.value
        if "output" in self._captured:
            reference, recorded = "recorded", _decode(self._store, self._captured["output"])
        else:
            evaluated = self._process._evaluate(self._chunk, self.task.partition)
            reference, recorded = "re-evaluated", self._process.reduce(evaluated)
        return ReplayDiff(reference, _equal(recorded, replayed), recorded, replayed)


def replay(plan: Plan[Any], key: int, *outputs: Array) -> Replay:
    """Replay task ``key`` of an ``aggregate_plan`` plan. ``outputs`` are the Arrays the plan was
    built from, in order: they supply the unfused graph and the user's frames, and must recompile
    to the plan's IR."""
    process = plan.process
    if not isinstance(process, _PartitionReduce):
        raise TypeError(f"replay needs a plan built by aggregate_plan, not one whose process is {process!r}")
    task = next((t for t in plan.tasks if t.key == key), None)
    if task is None:
        raise ValueError(f"the plan has no task with key {key!r}")
    if not outputs or bytes(compile_ir(outputs[0].session, *outputs).ir) != process.ir:
        raise ValueError(
            "these outputs do not recompile to the plan's IR: pass the outputs given to aggregate_plan, in order"
        )
    return Replay(process, task, outputs)
