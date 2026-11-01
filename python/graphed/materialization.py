"""Evaluate recorded results in this process, reading only the source columns they use."""

from __future__ import annotations

from collections.abc import Callable, Container, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from functools import partial
from operator import add
from typing import Any

from graphed.core.execution import LocalResources, Plan, SequentialRunner
from graphed.core.plan import DurablePlan, DurablePlanV2

from .aggregate import (
    _planned_arrays,
    _slot_of,
    aggregate_plan,
    attribute_failures,
    external_evaluators,
)
from .array import Array
from .execute import CompiledGraph, Key, _compile_cone, evaluate_ir
from .projection import read_columns
from .session import Session
from .varied import Varied, members_of
from .write import PartitionedSource, PartWrite, _concat_paths, declared_columns, sources_read

#: where a failure outside any one partition says it ran
_WHOLE = "whole dataset"

_PLANS = (Plan, DurablePlan, DurablePlanV2)

#: node kinds whose chunks do not concatenate to their whole value
_BOUNDARIES = ("reduction", "exchange", "join")


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
    return isinstance(item, (Array, PartWrite, *_PLANS)) or hasattr(
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

    def on_compiled(self, compiled: CompiledGraph) -> tuple[Any, ...]:
        """Run each piece's hook; a key several pieces label carries the union of their labels."""
        entries: dict[Key, tuple[set[str], Any]] = {}
        for piece in self.pieces.values():
            for key, (labels, frame) in piece.on_compiled(compiled) or ():
                entries.setdefault(key, (set(), frame))[0].update(labels)
        return tuple((k, (tuple(sorted(v[0])), v[1])) for k, v in entries.items())

    def externals(self) -> dict[str, Callable[..., object]]:
        return {k: fn for piece in self.pieces.values() for k, fn in piece.externals.items()}

    def by_item(self, value: list[Any]) -> dict[int, Any]:
        """Each result in ``value``, by the ``id`` of the object that asked for it."""
        return dict(zip([*self.pieces, *map(id, self.writes)], value, strict=True))


def _columns(data: object, arrays: Sequence[Array], nid: int) -> tuple[str, ...] | None:
    declared = declared_columns(data, arrays)
    return read_columns(arrays, nid) if declared is None else declared


def _keep(session: Session, nid: int, chunks: Sequence[object], reads: dict[int, object]) -> None:
    """Add ``chunks`` to ``reads`` as one chunk, unless the backend cannot make one."""
    concat = getattr(session.backend, "concat", None)
    if len(chunks) == 1:
        reads[nid] = chunks[0]
    elif chunks and concat is not None:
        reads[nid] = concat(chunks)


def _read(
    session: Session, arrays: Sequence[Array], passed: Container[int], reads: dict[int, object]
) -> None:
    """Add to ``reads`` each source ``arrays`` read, with the columns a plan over them would read.

    A source in ``passed`` had its pass. One whose read list is every column, or whose partitions
    the backend cannot concatenate, is left to its lazy loader.
    """
    resources = LocalResources()
    try:
        for nid, data in sources_read(arrays).items():
            if nid in passed or not isinstance(data, PartitionedSource):
                continue
            columns = _columns(data, arrays, nid)
            if columns is None:
                continue
            partitions = data.partitions(1)
            if len(partitions) == 1 or hasattr(session.backend, "concat"):
                chunks = [data.read_partition(p, columns, resources) for p in partitions]
                _keep(session, nid, chunks, reads)
    finally:
        resources.close()


def _frontier(session: Session, arrays: Sequence[Array], nid: int, past: Set[int]) -> list[Array]:
    """Return the nodes ``arrays`` take from a pass over source ``nid``.

    Those not ``past`` a reduction, whose chunks concatenate to their whole value; whatever
    consumes one evaluates on that concatenation.
    """
    nodes: dict[int, None] = {}

    def consumed(ins: Sequence[Any]) -> None:
        nodes.update(dict.fromkeys(i for i in ins if i is not None))

    def onward(node: int, ins: list[Any]) -> int | None:
        if ins and None not in ins and node not in past:
            return node
        consumed(ins)
        return None

    memo: dict[int, object] = {}
    for array in arrays:
        end = session.walk(
            array,
            source=lambda n: n if n == nid else None,
            op=lambda node, _name, ins, _params: onward(node, ins),
            external=lambda node, _fn, ins: onward(node, ins),
            memo=memo,
        )
        consumed([end])
    return session._proxies(nodes, arrays)


def _past_boundary(session: Session) -> set[int]:
    """Return the nodes at or downstream of a boundary; a node's inputs precede it in the store."""
    past: set[int] = set()
    for nid, nd in enumerate(session._nodes()):
        if nd["kind"] in _BOUNDARIES or not past.isdisjoint(nd["inputs"]):
            past.add(nid)
    return past


def _carrying(frontier: Sequence[Array]) -> PlanPieces:
    """Return a piece whose result is ``frontier``'s values, one tuple per partition."""
    slots: list[int] = []

    def on_compiled(compiled: CompiledGraph) -> None:
        slots[:] = map(_slot_of(compiled), frontier)

    return PlanPieces(
        outputs=tuple(frontier),
        reduce=lambda values: [tuple(values[slot] for slot in slots)],
        combine=add,
        empty=list,
        finalize=list,
        on_compiled=on_compiled,
    )


def _run_pass(
    session: Session,
    nid: int,
    joint: _Joint,
    arrays: Sequence[Array],
    past: Set[int],
    reads: dict[int, object],
) -> dict[int, Any]:
    """Run ``joint`` as one plan over source ``nid``, computing what ``arrays`` take from it.

    Those of ``arrays`` that read the source add their columns to the plan's; ``reads`` gets the
    whole value of each node they take.
    """
    sharing = [a for a in arrays if nid in session.reachable_sources(a.node_id)]
    read_by = _planned_arrays(joint.outputs(), tuple(joint.writes), store=None, opt_level=1)
    frontier = _frontier(session, sharing, nid, past)
    carried = _carrying(frontier)
    joint.pieces[id(carried)] = carried
    plan = aggregate_plan(
        *joint.outputs(),
        reduce=joint.reduce,
        combine=joint.combine,
        empty=joint.empty,
        externals=joint.externals(),
        on_compiled=joint.on_compiled,
        writes=joint.writes,
        finalize=joint.finalize,
        # this process's own backend: nothing ships, so nothing need rebuild it
        backend=lambda: session.backend,
        read_by=[*read_by, *sharing],
    )
    value = joint.by_item(SequentialRunner().run(plan).value)
    # popped, so each column's chunks go as soon as they are joined
    columns = list(zip(*value.pop(id(carried)), strict=True))
    while columns:
        _keep(session, frontier[len(columns) - 1].node_id, columns.pop(), reads)
    return value


def _run_whole(
    session: Session, joint: _Joint, arrays: Sequence[Array], reads: Mapping[int, object]
) -> dict[int, Any]:
    """Evaluate ``joint`` and ``arrays`` in one graph over whole sources, cut at ``reads``."""
    outputs = [*joint.outputs(), *arrays]
    compiled = _compile_cone(session, *outputs)
    labels = joint.on_compiled(compiled)
    landed = compiled.correspondence.node_map
    values = evaluate_ir(
        compiled,
        session.backend,
        {
            session.source_name(nid): partial(session.source_value, nid)
            for nid in sources_read(outputs)
            if nid not in reads
        },
        externals={**external_evaluators(session, compiled), **joint.externals()},
        on_failure=attribute_failures(compiled.correspondence.frames, _WHOLE, labels, opt_level=0),
        given={landed[nid][0]: value for nid, value in reads.items() if nid in landed},
    )
    slot = _slot_of(compiled)
    return {
        **joint.by_item(joint.finalize(joint.combine(joint.empty(), joint.reduce(values)))),
        **{id(array): values[slot(array)] for array in arrays},
    }


@dataclass
class _Work:
    """What one call asks of one Session."""

    arrays: list[Array] = field(default_factory=list)
    #: pieces by the ``id`` of their object, until every one has recorded what it needs
    pieces: dict[int, PlanPieces] = field(default_factory=dict)
    #: by the one partitioned source each reads
    passes: dict[int, _Joint] = field(default_factory=dict)
    #: what reads anything else
    whole: _Joint = field(default_factory=_Joint)

    def joint(self, arrays: Sequence[Array]) -> _Joint:
        sources = sources_read(arrays)
        if len(sources) != 1:
            return self.whole
        ((nid, data),) = sources.items()
        if isinstance(data, PartitionedSource):
            return self.passes.setdefault(nid, _Joint())
        return self.whole


def _evaluated(items: Iterable[Any]) -> dict[int, Any]:
    """Return each item's value by its ``id``; a Session's results share a pass over a source."""
    values: dict[int, Any] = {}
    works: dict[Session, _Work] = {}
    plans: list[Any] = []
    for item in items:
        if isinstance(item, _PLANS):
            plans.append(item)
        elif isinstance(item, Array):
            works.setdefault(item.session, _Work()).arrays.append(item)
        elif isinstance(item, PartWrite):
            work = works.setdefault(item.array.session, _Work())
            joint = work.joint([item.array])
            if joint is work.whole:
                raise TypeError(
                    "a write needs exactly one partitioned source; it writes per partition"
                )
            joint.writes.append(item)
        else:
            pieces: PlanPieces = item.__graphed_pieces__()
            if not pieces.outputs:
                values[id(item)] = pieces.finalize(pieces.empty())
                continue
            works.setdefault(pieces.outputs[0].session, _Work()).pieces[id(item)] = pieces
    for plan in plans:
        run = plan.to_plan() if isinstance(plan, DurablePlan) else plan
        values[id(plan)] = SequentialRunner().run(run).value
    for session, work in works.items():
        past = _past_boundary(session)
        for key, pieces in work.pieces.items():
            # past a boundary is whole, where a reduction has its one value
            whole = not past.isdisjoint(a.node_id for a in pieces.outputs)
            (work.whole if whole else work.joint(pieces.outputs)).pieces[key] = pieces
        reads: dict[int, object] = {}
        beside_passes = [*work.arrays, *work.whole.outputs()]
        for nid, joint in work.passes.items():
            values.update(_run_pass(session, nid, joint, beside_passes, past, reads))
        if beside_passes:
            _read(session, beside_passes, work.passes, reads)
            values.update(_run_whole(session, work.whole, work.arrays, reads))
    return values


def materialize(arrays: Any) -> Any:
    """Evaluate anything deferred in this process.

    An :class:`Array`, a :class:`~graphed.write.PartWrite`, a plan, an object whose class defines
    ``__graphed_pieces__`` (a deferred histogram; see :class:`PlanPieces`), or any nesting of them
    in lists, tuples (named ones keep their type) and mappings, recorded in any number of Sessions.
    Returns the same structure with each replaced by its value: an Array's value, a write's part
    paths, a plan's ``run(plan).value``, the object's own result, and a
    :class:`~graphed.Varied` becomes ``{label: value}``. Other items pass through.

    One Session's results share one read of each source, of the columns they use together, and
    evaluate each node they share once. Its pieces and writes over a source are one
    :func:`graphed.aggregate_plan`, which also computes everything its Arrays need that is not
    downstream of a reduction; the rest evaluates once over the concatenated chunks, so an Array
    is always its whole-dataset value. As in any chunked run, that takes behavior methods, named
    types' overloads and Externals to be row-wise, and an operation pairing two row sets raises
    in that pass, though it broadcasts alone. A piece downstream of a reduction, or reading
    anything but one partitioned source, evaluates with the Arrays over the whole dataset, where
    that reduction has its one correct value. A write consuming one is refused, as it is in a
    plan. A plan passed in is its own pass. A failing operation raises a
    :class:`~graphed.errors.StageError` at the line that recorded it.

    Evaluation only: running on an executor is a plan's job
    (:func:`graphed.aggregate_plan`, :func:`graphed.collate`).
    """
    # values are found by id, and a Mapping may build its values anew on each read
    items: dict[int, Any] = {}
    held = _replace_items(arrays, lambda item: items.setdefault(id(item), item))
    values = _evaluated(items.values())
    return _replace_items(held, lambda item: values[id(item)])
