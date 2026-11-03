"""Replay one task of an ``aggregate_plan`` run on the driver, one operation at a time.

This is opt_level=0.

``replay(plan, task)`` re-executes the task with the input it read — from the plan's capture root
(``aggregate_plan(store=)``) when the run kept it, else by re-reading its partition — one operation
at a time, each carrying the user's line: the members of the plan's fused stages, or the recorded
graph when the plan's ``outputs`` are given. A failing step raises the ``StageError`` M6's ``run``
would; ``diff()`` compares the replayed partial with the recorded one.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from graphed import Array
from graphed.aggregate import _compile_at, _PartitionReduce, _slot_of, resolve_backend
from graphed.core import DurablePlanV2, GraphStore, LocalResources
from graphed.core.execution import Plan, Task
from graphed.execute import (
    CompiledGraph,
    Key,
    _unbound_external,
    _unbound_source,
    compile_ir,
    external_key,
)

from .errors import SourceFrame
from .lowering import LoweredGraph, LoweredOp, LoweredStage, lower
from .runner import _stage_error


@dataclass(frozen=True)
class Step:
    """One replayed operation: its lowered op, its value, and how long it took.

    The lowered op carries the user's frame.
    """

    node: LoweredOp
    value: Any
    seconds: float


@dataclass(frozen=True)
class ReplayDiff:
    """The replayed partial against the reference.

    ``"recorded"`` (the run's captured output) or ``"re-evaluated"`` (the plan's graph run again on
    the same input).
    """

    reference: str
    equal: bool
    recorded: Any
    replayed: Any


def _decode(store: Any, blob: str) -> Any:
    from graphed.checkpoint import PickleCodec

    data = store.get(blob)
    if data is None:
        raise FileNotFoundError(f"capture blob {blob} is missing or corrupt")
    return PickleCodec().decode(data)


def _equal(a: Any, b: Any) -> bool:
    """Compare by value.

    Dicts by key, sequences by position, leaves by ``numpy.array_equal`` (which dispatches to
    awkward and answers False on a length or shape mismatch). NaN is unequal.
    """
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(map(_equal, a, b))
    import numpy

    return bool(numpy.array_equal(a, b))


def _unfused(
    process: _PartitionReduce[Any],
) -> tuple[list[dict[str, Any]], LoweredGraph, list[int]]:
    """Return the plan's IR one operation per node, its lowering and its outputs' new ids.

    A stage's members are the operations it runs and the plan's frames name each, so this needs
    nothing the plan does not already ship.
    """
    arena = GraphStore.deserialize(process.ir)
    frames = dict(process.frames)
    boundary_ops = resolve_backend(process.backend_factory).boundary_ops()
    nodes: list[dict[str, Any]] = []
    ops: list[LoweredOp] = []
    value_of: dict[int, int] = {}

    def add(node: Mapping[str, Any], key: Key, inputs: list[int]) -> None:
        new = len(nodes)
        nodes.append({**node, "id": new, "inputs": inputs})
        frame = SourceFrame(*frames.get(key, ("<no recorded line>", 0)))
        kind = node["kind"]
        name = node["name"] or kind  # an External has no name
        lowered_kind = "op" if kind == "reduction" else kind
        boundary = lowered_kind != "op" or name in boundary_ops
        ops.append(LoweredOp(new, name, lowered_kind, frame, tuple(inputs), "", boundary))
        value_of[key[0]] = new

    for node in arena.nodes():
        ins = [value_of[i] for i in node["inputs"]]
        if node["kind"] != "stage":
            add(node, (node["id"], None), ins)
            continue
        base = len(nodes)
        for index, member in enumerate(node["members"]):
            inputs = [ins[i] if tag == "input" else base + i for tag, i in member["inputs"]]
            add(member, (node["id"], index), inputs)
    stages = tuple(LoweredStage(i, (op,)) for i, op in enumerate(ops))
    outputs = [value_of[o] for o in arena.outputs()]
    return nodes, LoweredGraph(0, stages, outputs[-1]), outputs


class Replay:
    """One task of a run, ready to replay.

    ``steps()`` is lazy; ``value`` runs it once and caches.
    """

    def __init__(
        self,
        process: _PartitionReduce[Any],
        task: Task,
        outputs: tuple[Array, ...] = (),
        compiled: CompiledGraph | None = None,
    ) -> None:
        """Prepare ``task`` of a run for replay, over ``outputs`` when they are given."""
        self.task = task
        self._process = process
        self._outputs = outputs
        self._lowered: LoweredGraph | None = None
        if outputs:
            self._session = outputs[0].session
            if compiled is None:
                compiled = _compile_at(process.opt_level, self._session, outputs)
            slot, first = _slot_of(compiled), dict[int, int]()
            for o in outputs:
                first.setdefault(slot(o), o.node_id)
            # one value per IR output, in its order, as the run's `reduce` receives them
            self._reduced_ids = [first[s] for s in sorted(first)]
            ops = (op for o in outputs for op in lower(self._session, o, opt_level=0).ops)
            cone = {op.node_id: op for op in ops}
            self._cone = [cone[i] for i in sorted(cone)]
            # the unfused IR keeps record ids; it is the whole arena, so only the cone is evaluated
            arena = GraphStore.deserialize(
                bytes(compile_ir(self._session, *outputs, optimize=False).ir)
            )
            self._nodes = [n for n in arena.nodes() if n["id"] in cone]
        else:
            self._nodes, self._lowered, self._reduced_ids = _unfused(process)
            self._cone = list(self._lowered.ops)
        self._store: Any = None
        self._captured: dict[str, str] = {}
        if process.store is not None:
            self._store = process.open_store()
            cid = process.capture_id(task.partition)
            done = self._store.completed()
            self._captured = {
                k: done[f"{cid}:{k}"].blob for k in ("input", "output") if f"{cid}:{k}" in done
            }
        self.input_source = "store" if "input" in self._captured else "re-read"

    @cached_property
    def _chunk(self) -> Any:
        if "input" in self._captured:
            return _decode(self._store, self._captured["input"])
        return self._process.reader.read_partition(
            self.task.partition, self._process.columns, LocalResources()
        )

    def steps(self) -> Iterator[Step]:
        """Evaluate the task's opt_level=0 cone node by node, yielding each :class:`Step`.

        The input is read before the first step, and a failed read raises as the run's own read
        would.
        """
        # workers import graphed.debug
        from graphed.preserve.interpreter import iter_ir

        chunk = self._chunk
        source_name = self._process.source_name
        externals = dict(self._process.externals)

        # bound as evaluate_ir binds them for the run, so an unbound name fails replay as it failed
        # the task
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
                with self._process.floating_point():
                    _nid, value = next(values)
            except Exception as exc:
                failing = self._lowered
                if failing is None:
                    proxy = self._session._proxies([op.node_id], self._outputs)[0]
                    failing = lower(self._session, proxy, opt_level=0)
                raise _stage_error(failing, op.node_id, str(self.task.partition), exc) from exc
            yield Step(op, value, time.perf_counter() - start)

    @cached_property
    def value(self) -> Any:
        """The replayed partial.

        The plan's ``reduce`` over one replayed value per output of the plan's IR, as the run's
        ``reduce`` receives them.
        """
        out = {
            s.node.node_id: s.value for s in self.steps() if s.node.node_id in self._reduced_ids
        }
        return self._process.reduce([out[i] for i in self._reduced_ids])

    def diff(self) -> ReplayDiff:
        """Compare the replayed partial with the run's recorded one.

        When the run kept no output for this task, compare with the plan's graph evaluated on the
        same input.
        """
        replayed = self.value
        if "output" in self._captured:
            reference, recorded = "recorded", _decode(self._store, self._captured["output"])
        else:
            evaluated = self._process.evaluate(self._chunk, self.task.partition)
            reference, recorded = "re-evaluated", self._process.reduce(evaluated)
        return ReplayDiff(reference, _equal(recorded, replayed), recorded, replayed)


def replay(plan: Plan[Any] | DurablePlanV2, task: int, *outputs: Array) -> Replay:
    """Replay the task keyed ``task`` of an ``aggregate_plan`` plan.

    Each step is one operation the plan runs. ``outputs``, the Arrays the plan was built from in
    order, replay the recorded graph instead, op for op as written; they must recompile to the
    plan's IR. A plan over a join runs in stages and is refused.
    """
    if not isinstance(plan, Plan):
        raise TypeError("replay needs a plan with fixed tasks; this one runs in stages")
    process = plan.process
    if not isinstance(process, _PartitionReduce):
        raise TypeError(
            f"replay needs a plan built by aggregate_plan, not one whose process is {process!r}"
        )
    if process.writes:
        raise TypeError(
            "replay does not re-run a plan's writes; replay a plan built without writes="
        )
    found = next((t for t in plan.tasks if t.key == task), None)
    if found is None:
        raise ValueError(f"the plan has no task with key {task!r}")
    if not outputs:
        return Replay(process, found)
    for output in outputs:
        if not isinstance(output, Array):
            raise TypeError(
                f"replay's outputs are deferred Arrays, not {type(output).__name__}; "
                "gd.replay(plan, task) takes them from the plan"
            )
    compiled = _compile_at(process.opt_level, outputs[0].session, outputs)
    if bytes(compiled.ir) != process.ir:
        raise ValueError(
            "these outputs do not recompile to the plan's IR: pass the outputs given to "
            "aggregate_plan, in order, or none to take them from the plan"
        )
    return Replay(process, found, outputs, compiled)
