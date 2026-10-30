"""Evaluate recorded results in this process, reading only the source columns they use."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, cast

from graphed.core import Partition
from graphed.core.execution import LocalResources, Plan, SequentialRunner, WorkerResources
from graphed.core.plan import DurablePlanV2

from .aggregate import (
    _PartitionReduce,
    _planned_arrays,
    aggregate_plan,
    attribute_failures,
    external_evaluators,
)
from .array import Array
from .execute import CompiledGraph, Frame, Key, OnFailure, _dispatch, compile_ir, evaluate_ir
from .projection import read_columns
from .session import Session
from .varied import Varied, members_of
from .write import PartitionedSource, PartWrite, _concat_paths, declared_columns, sources_read

#: where a failure outside any one partition says it ran
_WHOLE = "whole dataset"


@dataclass(frozen=True)
class PlanPieces:
    """A deferred object's share of one :func:`graphed.aggregate_plan`.

    A class whose instances evaluate through a plan defines ``__graphed_pieces__()`` returning
    these, and :func:`materialize` plans them beside the call's other results. ``reduce`` gets
    the whole value list, so it reads ``outputs`` at the slots ``on_compiled`` records from the
    :class:`CompiledGraph`; what ``on_compiled`` returns joins the plan's variation labels.
    ``finalize`` turns the combined value into the object's result; with no ``outputs`` it is
    given ``empty()`` and nothing runs. Pieces belong to one plan, so ``__graphed_pieces__`` builds
    fresh ones each call.
    """

    outputs: tuple[Array, ...]
    reduce: Callable[[list[Any]], Any]
    combine: Callable[[Any, Any], Any]
    empty: Callable[[], Any]
    finalize: Callable[[Any], Any]
    on_compiled: Callable[[CompiledGraph], Any]
    externals: Mapping[str, Callable[..., object]] = field(default_factory=dict)


def _deferred(item: Any) -> bool:
    return isinstance(item, Array | PartWrite | Plan | DurablePlanV2) or hasattr(
        type(item), "__graphed_pieces__"
    )


def _replace_items(item: Any, leaf: Callable[[Any], Any]) -> Any:
    """Return ``item`` with each deferred result replaced by ``leaf(result)``.

    Through lists, tuples (named ones keep their type), mappings and a ``Varied``'s universes,
    which become a ``{label: result}`` mapping; other items pass through.
    """
    if _deferred(item):
        return leaf(item)
    if isinstance(item, Varied):
        return {label: _replace_items(v, leaf) for label, v in members_of(item).items()}
    if isinstance(item, Mapping):
        return {key: _replace_items(v, leaf) for key, v in item.items()}
    if isinstance(item, list):
        return [_replace_items(v, leaf) for v in item]
    if isinstance(item, tuple):
        values = [_replace_items(v, leaf) for v in item]
        return type(item)(*values) if hasattr(item, "_fields") else tuple(values)
    return item


@dataclass
class _Joint:
    """Pieces and writes evaluated together: their one value is a list, pieces first."""

    pieces: dict[int, PlanPieces] = field(default_factory=dict)
    writes: list[PartWrite] = field(default_factory=list)

    def outputs(self) -> list[Array]:
        return [array for piece in self.pieces.values() for array in piece.outputs]

    def reduce(self, values: list[Any]) -> list[Any]:
        paths = values[len(values) - len(self.writes) :]
        return [*(p.reduce(values) for p in self.pieces.values()), *([path] for path in paths)]

    def combine(self, a: list[Any], b: list[Any]) -> list[Any]:
        fns = [*(p.combine for p in self.pieces.values()), *[_concat_paths] * len(self.writes)]
        return [fn(x, y) for fn, x, y in zip(fns, a, b, strict=True)]

    def empty(self) -> list[Any]:
        return [*(p.empty() for p in self.pieces.values()), *([] for _ in self.writes)]

    def finalize(self, value: list[Any]) -> list[Any]:
        fns = [*(p.finalize for p in self.pieces.values()), *[list] * len(self.writes)]
        return [fn(v) for fn, v in zip(fns, value, strict=True)]

    def on_compiled(self, compiled: CompiledGraph) -> tuple[Any, ...] | None:
        """Run each piece's hook; a key several pieces label carries the union of their labels."""
        entries: dict[Key, tuple[set[str], Any]] = {}
        for piece in self.pieces.values():
            for key, (labels, frame) in piece.on_compiled(compiled) or ():
                entries.setdefault(key, (set(), frame))[0].update(labels)
        return tuple((k, (tuple(sorted(v[0])), v[1])) for k, v in entries.items()) or None

    def externals(self) -> dict[str, Callable[..., object]]:
        return {k: fn for piece in self.pieces.values() for k, fn in piece.externals.items()}

    def by_item(self, value: list[Any]) -> dict[int, Any]:
        """Each result in ``value``, by the ``id`` of the object that asked for it."""
        return dict(zip([*self.pieces, *map(id, self.writes)], value, strict=True))


@dataclass(frozen=True)
class _Kept:
    """A reader keeping each chunk it reads, for the Arrays sharing the pass."""

    source: PartitionedSource
    chunks: list[object]

    def read_partition(
        self, partition: Partition, columns: Sequence[str] | None, resources: WorkerResources
    ) -> object:
        chunk = self.source.read_partition(partition, columns, resources)
        self.chunks.append(chunk)
        return chunk


def _columns(data: object, arrays: Sequence[Array], nid: int) -> tuple[str, ...] | None:
    declared = declared_columns(data, arrays)
    return read_columns(arrays, nid) if declared is None else declared


def _joined(session: Session, chunks: Sequence[object]) -> object | None:
    """``chunks`` as one chunk; ``None`` when there is none or the backend cannot make one."""
    if len(chunks) == 1:
        return chunks[0]
    concat = getattr(session.backend, "concat", None)
    return None if concat is None or not chunks else concat(chunks)


def _read(session: Session, arrays: Sequence[Array], reads: dict[int, object]) -> None:
    """Add to ``reads`` each source ``arrays`` read, with the columns a plan over them would read.

    A source whose read list is every column, or whose partitions the backend cannot concatenate,
    is left to its lazy loader.
    """
    if not arrays:
        return
    resources = LocalResources()
    try:
        for nid, data in sources_read(arrays).items():
            if nid in reads or not isinstance(data, PartitionedSource):
                continue
            columns = _columns(data, arrays, nid)
            if columns is None:
                continue
            partitions = data.partitions(1)
            if len(partitions) == 1 or hasattr(session.backend, "concat"):
                chunks = [data.read_partition(p, columns, resources) for p in partitions]
                joined = _joined(session, chunks)
                if joined is not None:
                    reads[nid] = joined
    finally:
        resources.close()


def _run_pass(
    session: Session, nid: int, joint: _Joint, arrays: Sequence[Array], reads: dict[int, object]
) -> dict[int, Any]:
    """Run ``joint`` as one plan over source ``nid``; add the read ``arrays`` share to ``reads``.

    Those of ``arrays`` that read the source widen the plan's columns to theirs.
    """
    plan = aggregate_plan(
        *joint.outputs(),
        reduce=joint.reduce,
        combine=joint.combine,
        empty=joint.empty,
        externals=joint.externals(),
        on_compiled=joint.on_compiled,
        writes=joint.writes,
        finalize=joint.finalize,
    )
    sharing = [a for a in arrays if nid in session.reachable_sources(a.node_id)]
    chunks: list[object] = []
    if sharing:
        process = cast("_PartitionReduce[Any]", plan.process)
        planned = _planned_arrays(joint.outputs(), tuple(joint.writes), store=None, opt_level=1)
        plan = replace(
            plan,
            process=replace(
                process,
                reader=cast("PartitionedSource", _Kept(process.reader, chunks)),
                columns=_columns(process.reader, [*planned, *sharing], nid),
            ),
        )
    value = joint.by_item(SequentialRunner().run(plan).value)
    joined = _joined(session, chunks)
    if joined is not None:
        reads[nid] = joined
    return value


def _run_whole(session: Session, joint: _Joint, reads: Mapping[int, object]) -> dict[int, Any]:
    """Evaluate ``joint`` once over whole sources, as no one partitioned source covers it."""
    if joint.writes:
        raise TypeError("a write needs exactly one partitioned source; it writes per partition")
    outputs = joint.outputs()
    compiled = compile_ir(session, *outputs)
    labels = joint.on_compiled(compiled)
    sources = {
        session.source_name(nid): reads[nid]
        if nid in reads
        else partial(session.source_value, nid)
        for nid in sources_read(outputs)
    }
    values = evaluate_ir(
        compiled,
        session.backend,
        sources,
        externals={**external_evaluators(session, compiled), **joint.externals()},
        on_failure=attribute_failures(compiled.correspondence.frames, _WHOLE, labels),
    )
    return joint.by_item(joint.finalize(joint.reduce(values)))


def _on_failure(session: Session) -> OnFailure | None:
    """Return the attribution of a walk over ``session``'s recorded nodes."""
    frames: list[tuple[Key, Frame]] = [
        (
            (nid, None),
            (str(p["filename"]), cast("int", p["lineno"]), str(p["function"]), str(p["source"])),
        )
        for nid, p in session.sourcemap().items()
    ]
    return attribute_failures(frames, _WHOLE, opt_level=0)


def _value(
    session: Session, array: Array, memo: dict[int, object], on_failure: OnFailure | None
) -> object:
    """``array``'s value; walks sharing ``memo`` evaluate each shared node once."""

    def op(nid: int, name: str, ins: list[object], params: Mapping[str, object]) -> object:
        run = partial(session.backend.eval_stage, name, ins, params)
        return _dispatch(run, name, ins, (nid, None), on_failure)

    def external(nid: int, fn: Callable[..., object], ins: list[object]) -> object:
        return _dispatch(partial(fn, *ins), "external", ins, (nid, None), on_failure)

    return session.walk(array, source=session.source_value, op=op, external=external, memo=memo)


@dataclass
class _Work:
    """What one call asks of one Session."""

    arrays: list[Array] = field(default_factory=list)
    #: by the one partitioned source each reads; ``None`` holds what reads anything else
    joints: dict[int | None, _Joint] = field(default_factory=dict)

    def joint(self, arrays: Sequence[Array]) -> _Joint:
        sources = sources_read(arrays)
        partitioned = len(sources) == 1 and isinstance(
            next(iter(sources.values())), PartitionedSource
        )
        return self.joints.setdefault(next(iter(sources)) if partitioned else None, _Joint())


def _evaluated(items: Sequence[Any]) -> dict[int, Any]:
    """Return each item's value by its ``id``; a Session's results share a pass over a source."""
    values: dict[int, Any] = {}
    works: dict[Session, _Work] = {}
    for item in {id(i): i for i in items}.values():
        if isinstance(item, Plan | DurablePlanV2):
            values[id(item)] = SequentialRunner().run(item).value
        elif isinstance(item, Array):
            works.setdefault(item.session, _Work()).arrays.append(item)
        elif isinstance(item, PartWrite):
            work = works.setdefault(item.array.session, _Work())
            work.joint([item.array]).writes.append(item)
        else:
            pieces: PlanPieces = item.__graphed_pieces__()
            if not pieces.outputs:
                values[id(item)] = pieces.finalize(pieces.empty())
                continue
            work = works.setdefault(pieces.outputs[0].session, _Work())
            work.joint(pieces.outputs).pieces[id(item)] = pieces
    for session, work in works.items():
        reads: dict[int, object] = {}
        for nid, joint in work.joints.items():
            if nid is not None:
                values.update(_run_pass(session, nid, joint, work.arrays, reads))
        whole = work.joints.get(None)
        _read(session, [*work.arrays, *(whole.outputs() if whole else ())], reads)
        if whole:
            values.update(_run_whole(session, whole, reads))
        on_failure = _on_failure(session)
        for array in work.arrays:
            values[id(array)] = _value(session, array, reads, on_failure)
    return values


def materialize(arrays: Any) -> Any:
    """Evaluate anything deferred in this process.

    An :class:`Array`, a :class:`~graphed.write.PartWrite`, a plan, an object whose class defines
    ``__graphed_pieces__`` (a deferred histogram; see :class:`PlanPieces`), or any nesting of them
    in lists, tuples (named ones keep their type) and mappings, recorded in any number of Sessions.
    Returns the same structure with each replaced by its value: an Array's value, a write's part
    paths, a plan's ``run(plan).value``, the object's own result, and a
    :class:`~graphed.Varied` becomes ``{label: value}``. Other items pass through.

    One Session's results share one read of each source, of the columns they use together. Its
    pieces and writes over a source are one :func:`graphed.aggregate_plan`, evaluating what they
    share once; its Arrays evaluate every node they share once, over the chunks that plan read. A
    plan passed in is its own pass. Pieces reading anything but one partitioned source evaluate
    once over whole sources. A failing operation raises a :class:`~graphed.errors.StageError` at
    the line that recorded it.

    Evaluation only: running on an executor is a plan's job
    (:func:`graphed.aggregate_plan`, :func:`graphed.collate`).
    """
    items: list[Any] = []
    _replace_items(arrays, items.append)
    values = _evaluated(items)
    return _replace_items(arrays, lambda item: values[id(item)])
