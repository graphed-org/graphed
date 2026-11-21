"""Evaluate recorded results in this process, reading only the source columns they use."""

from __future__ import annotations

from collections.abc import Callable, Container, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from functools import cache, partial
from operator import add, index
from typing import Any

from graphed.core import Partition
from graphed.core.execution import LocalResources, Plan, SequentialRunner, WorkerResources
from graphed.core.plan import DurablePlan, DurablePlanV2

from .aggregate import (
    _planned_arrays,
    _refuse_slot_readers,
    _slot_of,
    aggregate_plan,
    attribute_failures,
    external_evaluators,
    read_label,
)
from .array import Array
from .execute import CompiledGraph, Key, _compile_cone, evaluate_ir, resolved_partition
from .projection import read_columns
from .session import Session, wants_partition
from .varied import Varied, members_of, refuse_container
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
    the whole value list, one value per output of the plan, so it reads each of ``outputs`` at its
    first index in the :class:`CompiledGraph`'s ``requested``, which ``on_compiled`` records;
    what ``on_compiled`` returns joins the plan's variation labels.
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
        outputs, landed = self.outputs(), compiled.correspondence.node_map
        if len({landed[a.node_id][0] for a in outputs}) < len(outputs):
            _refuse_slot_readers(
                *(f for p in self.pieces.values() for f in (p.reduce, p.on_compiled))
            )
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


def _downstream(
    nodes: Sequence[Mapping[str, Any]], seed: Callable[[int, Mapping[str, Any]], bool]
) -> set[int]:
    """Return the nodes ``seed`` picks and those downstream; a node's inputs precede it."""
    found: set[int] = set()
    for nid, nd in enumerate(nodes):
        if not found.isdisjoint(nd["inputs"]) or seed(nid, nd):
            found.add(nid)
    return found


def _paired_by_chunk(
    session: Session, nodes: Sequence[Mapping[str, Any]], among: Container[int] | None = None
) -> set[int]:
    """Return the nodes at or downstream of an op the backend pairs by chunk, not by row.

    With ``among``, only of such an op in ``among``.
    """
    pairs = getattr(session.backend, "pairs_by_chunk", None)
    if pairs is None:
        return set()
    return _downstream(
        nodes,
        lambda nid, nd: (
            nd["kind"] == "op"
            and (among is None or nid in among)
            and pairs(nd["name"], nd["params"])
        ),
    )


def _partition_fed(session: Session, arrays: Sequence[Array]) -> frozenset[int]:
    """Return the sources ``arrays`` read through a function that reads its input partition."""
    readers = _downstream(session._nodes(), lambda _, nd: wants_partition(nd.get("params") or {}))
    return frozenset[int]().union(
        *(session.reachable_sources(a.node_id) for a in arrays if a.node_id in readers)
    )


def _chunks_join(session: Session, data: PartitionedSource) -> bool:
    """Return whether ``data``'s partitions make one chunk under this session's backend."""
    return hasattr(session.backend, "concat") or len(data.partitions(1)) == 1


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
            if _chunks_join(session, data):
                chunks = [data.read_partition(p, columns, resources) for p in data.partitions(1)]
                _keep(session, nid, chunks, reads)
    finally:
        resources.close()


def _first(value: Any, n: int) -> tuple[Any, int]:
    """Return ``value``'s first ``n`` rows and how many that is.

    A Mapping is a backend's columns; a value at most ``n`` long stays itself, as slicing a chunk
    can reach the placeholders of its unread columns.
    """
    if isinstance(value, Mapping):
        columns = {name: column[:n] for name, column in value.items()}
        return columns, max(map(len, columns.values()), default=0)
    rows = len(value)
    return (value, rows) if rows <= n else (value[:n], n)


def _read_first(
    data: PartitionedSource,
    columns: Sequence[str] | None,
    entries: int,
    resources: WorkerResources,
) -> tuple[list[object], int, list[Partition]]:
    """Read partitions in order until ``entries`` rows are read; return chunks, rows and parts."""
    chunks: list[object] = []
    parts: list[Partition] = []
    rows = 0
    for partition in data.partitions(1):
        wanted = entries - rows
        parts.append(partition.first(wanted))
        chunk, read = _first(data.read_partition(parts[-1], columns, resources), wanted)
        chunks.append(chunk)
        rows += read
        if rows == entries:
            break
    return chunks, rows, parts


def _first_entries(
    session: Session, arrays: Sequence[Array], entries: int, load: Callable[[int], object]
) -> tuple[dict[int, object], bool, Callable[[], Partition] | None]:
    """Return each source's first ``entries`` rows, whether every source ran out, and a partition.

    The partition is the one a ``partition=True`` function's source was read from, where exactly
    one partition feeds such functions.

    ``load`` gives a source read whole: one that is not partitioned, or whose chunks do not join.
    """
    reads: dict[int, object] = {}
    exhausted = True
    first: Callable[[], Partition] | None = None
    paired = cache(lambda: _paired_by_chunk(session, session._nodes()))
    fed = cache(lambda: _partition_fed(session, arrays))
    resources = LocalResources()
    try:
        for nid, data in sources_read(arrays).items():
            if isinstance(data, PartitionedSource) and _chunks_join(session, data):
                columns = _columns(data, arrays, nid)
                chunks, rows, parts = _read_first(data, columns, entries, resources)
                if len(chunks) > 1 and (
                    nid in fed()
                    or not paired().isdisjoint(
                        a.node_id for a in arrays if nid in session.reachable_sources(a.node_id)
                    )
                ):
                    raise ValueError(
                        f"the first {entries} entries span {len(chunks)} partitions, and these "
                        "results take an op that pairs its operands chunk by chunk or a function "
                        "that reads its input partition, so one evaluation over them is wrong; "
                        "materialize them whole, or one chunk with partition="
                    )
                _keep(session, nid, chunks, reads)
                if nid in fed():  # one chunk, else refused above
                    first = cache(partial(resolved_partition, data, parts[0]))
            else:
                reads[nid], rows = _first(load(nid), entries)
            exhausted = exhausted and rows < entries
    finally:
        resources.close()
    return reads, exhausted, first if len(fed()) == 1 else None


def _read_chunk(
    data: PartitionedSource, partition: Partition, columns: Sequence[str] | None
) -> object:
    resources = LocalResources()
    try:
        return data.read_partition(partition, columns, resources)
    finally:
        resources.close()


def _partition_chunk(
    arrays: Sequence[Array], partition: Partition
) -> tuple[int, Callable[[], object], Callable[[], str], Callable[[], Partition]]:
    """Return the source holding ``partition``, its chunk's read and label, and its partition.

    The partition is the one a task hands a ``partition=True`` function; the label is the one a
    plan's task gives the chunk, worked out only when asked for.
    """
    place = (partition.uri, partition.tree)
    found = [
        (nid, data)
        for nid, data in sources_read(arrays).items()
        if isinstance(data, PartitionedSource)
        and any((p.uri, p.tree) == place for p in data.partitions(1))
    ]
    if not found:
        raise ValueError(f"no source of these results holds {partition}")
    nid, data = found[0]
    read: list[object] = []

    def chunk() -> object:
        read.append(_read_chunk(data, partition, _columns(data, arrays, nid)))
        return read[-1]

    def label() -> str:
        return read_label(data, partition, read[-1] if read else None)

    return nid, chunk, label, cache(partial(resolved_partition, data, partition))


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


def _past_boundary(session: Session, nodes: Sequence[Mapping[str, Any]] | None = None) -> set[int]:
    """Return the nodes at or downstream of a boundary."""
    return _downstream(
        session._nodes() if nodes is None else nodes, lambda _, nd: nd["kind"] in _BOUNDARIES
    )


def _positions(compiled: CompiledGraph) -> Callable[[int], int]:
    """Return where a node's value sits among one value per output asked, by its record id."""
    first: dict[int, int] = {}
    for i, nid in enumerate(compiled.requested):
        first.setdefault(nid, i)
    return first.__getitem__


def _carrying(frontier: Sequence[Array]) -> PlanPieces:
    """Return a piece whose result is ``frontier``'s values, one tuple per partition."""
    slots: list[int] = []

    def on_compiled(compiled: CompiledGraph) -> None:
        slots[:] = map(_positions(compiled), (a.node_id for a in frontier))

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
    outputs, writes = joint.outputs(), tuple(joint.writes)
    read_by = (
        _planned_arrays(outputs, writes, store=None, opt_level=1) if outputs or writes else []
    )
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
    session: Session,
    joint: _Joint,
    arrays: Sequence[Array],
    reads: Mapping[int, object],
    where: str | Callable[[], str] = _WHOLE,
    *,
    chunked: bool = False,
    partition: Callable[[], Partition] | None = None,
) -> dict[int, Any]:
    """Evaluate ``joint`` and ``arrays`` in one graph over whole sources, cut at ``reads``.

    ``where`` is what a failure says ran; ``chunked`` and ``partition`` are :func:`evaluate_ir`'s.
    """
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
        on_failure=attribute_failures(compiled.correspondence.frames, where, labels, opt_level=0),
        given={landed[nid][0]: value for nid, value in reads.items() if nid in landed},
        chunked=chunked,
        partition=partition,
    )
    slot = _slot_of(compiled)
    per_output = [values[slot(a)] for a in joint.outputs()]
    return {
        **joint.by_item(joint.finalize(joint.combine(joint.empty(), joint.reduce(per_output)))),
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


@dataclass(frozen=True)
class _Sample:
    """What one call evaluates in place of its whole sources."""

    #: each source's stand-in, by node id, for ``(session, arrays)``, and the one partition they
    #: came from, as a task hands it to a ``partition=True`` function
    reads: Callable[
        [Session, Sequence[Array]], tuple[dict[int, object], Callable[[], Partition] | None]
    ]
    #: what a failure says ran
    where: str | Callable[[], str]
    chunked: bool = False


def _evaluated(items: Iterable[Any], sample: _Sample | None = None) -> dict[int, Any]:
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
        if sample is not None:
            joint = _Joint(pieces=work.pieces)
            stand_ins, part = sample.reads(session, [*work.arrays, *joint.outputs()])
            ran = _run_whole(
                session,
                joint,
                work.arrays,
                stand_ins,
                sample.where,
                chunked=sample.chunked,
                partition=part,
            )
            values.update(ran)
            continue
        nodes = session._nodes()
        paired = _paired_by_chunk(session, nodes)
        # a pass over each partitioned source of a paired Array evaluates it chunk by chunk
        chunk_paired = [a for a in work.arrays if a.node_id in paired]
        for nid, data in (sources_read(chunk_paired) if chunk_paired else {}).items():
            if isinstance(data, PartitionedSource):
                work.passes.setdefault(nid, _Joint())
        past = _past_boundary(session, nodes) if work.pieces or work.passes else frozenset[int]()
        stranded = _paired_by_chunk(session, nodes, past) if paired else set()
        if any(a.node_id in stranded for a in [*work.arrays, *_Joint(work.pieces).outputs()]):
            raise ValueError(
                "an op the backend pairs by chunk is downstream of a reduction, exchange or join "
                "here, so materialize has no chunk to pair its operands by"
            )
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


def _first_label(entries: int) -> str:
    return f"first {entries} entries"


def _first_sample(entries: int, caller: str) -> _Sample:
    if type(entries) is not int or entries < 1:
        raise ValueError(f"{caller}(entries=) is a positive int, not {entries!r}")

    def reads(
        session: Session, arrays: Sequence[Array]
    ) -> tuple[dict[int, object], Callable[[], Partition] | None]:
        values, _, part = _first_entries(session, arrays, entries, session.source_value)
        return values, part

    return _Sample(reads, _first_label(entries))


def _sample_of(entries: int | None, partition: Partition | None) -> _Sample | None:
    """Return what ``entries=`` or ``partition=`` asks for, refusing a bad request."""
    if entries is not None and partition is not None:
        raise ValueError("materialize takes entries= or partition=, not both")
    if partition is not None:
        last: list[Callable[[], str]] = []

        def reads(
            session: Session, arrays: Sequence[Array]
        ) -> tuple[dict[int, object], Callable[[], Partition] | None]:
            nid, chunk, label, part = _partition_chunk(arrays, partition)
            last.append(label)
            # another source is read whole, so a partition=True function over it has no partition
            return {nid: chunk()}, part if _partition_fed(session, arrays) <= {nid} else None

        return _Sample(reads, lambda: last[-1](), chunked=True)
    return None if entries is None else _first_sample(entries, "materialize")


def materialize(
    arrays: Any, *, entries: int | None = None, partition: Partition | None = None
) -> Any:
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
    in that pass, though it broadcasts alone. An Array taking an op its backend's
    ``pairs_by_chunk`` names evaluates in a per-partition pass instead, and a sample of
    ``entries=`` spanning partitions refuses it. A piece downstream of a reduction, or reading
    anything but one partitioned source, evaluates with the Arrays over the whole dataset, where
    that reduction has its one correct value. A write consuming one is refused, as it is in a
    plan. A plan passed in is its own pass. A failing operation raises a
    :class:`~graphed.errors.StageError` at the line that recorded it.

    ``entries=N`` evaluates over each source's first ``N`` input events instead, reading only those
    partitions' used columns: a reduction, a histogram or a cut sees those events as its dataset.
    ``partition=`` does the same over one chunk, such as a plan's ``plan.tasks[3].partition``, the
    way a task runs it, so a task's failure raises here too. A plan or a write runs whole, so
    neither takes them. A sample of ``entries=`` is not checked as one row set: an operand cut down
    to one row broadcasts here, where the whole dataset raises.

    Evaluation only: running on an executor is a plan's job
    (:func:`graphed.aggregate_plan`, :func:`graphed.collate`).

    >>> import awkward as ak
    >>> import graphed
    >>> from graphed.awkward import from_awkward
    >>> x = from_awkward("x", ak.Array([[1.0, 2.0], [], [3.0]]))
    >>> graphed.materialize(x[x > 1.5]).tolist()
    [[2.0], [], [3.0]]
    """
    # values are found by id, and a Mapping may build its values anew on each read
    items: dict[int, Any] = {}
    held = _replace_items(arrays, lambda item: items.setdefault(id(item), item))
    sample = _sample_of(entries, partition)
    if sample is not None and any(isinstance(i, (PartWrite, *_PLANS)) for i in items.values()):
        raise TypeError("a plan or a write runs whole; entries= and partition= cannot sample it")
    values = _evaluated(items.values(), sample)
    return _replace_items(held, lambda item: values[id(item)])


def head(array: Array, n: int = 5) -> Any:
    """Return the first ``n`` rows of ``array``'s value, reading as little as that takes.

    Reads the used columns of the first entries, widening until ``n`` rows survive the cuts or the
    data ends. Rows downstream of a reduction need the whole dataset, so they read all of it.
    """
    refuse_container("graphed.head", array)
    n = index(n)
    if n < 0:
        raise ValueError(f"head(n=) is at least 0, not {n}")
    session = array.session
    if array.node_id in _past_boundary(session):
        return materialize(array[:n])
    entries = max(n, 1)
    load = cache(session.source_value)
    while True:
        reads, exhausted, part = _first_entries(session, [array], entries, load)
        # nothing here is past a boundary, so every operand is the prefix's rows, and a length-1
        # operand is a row set that must not broadcast
        where = _first_label(entries)
        value = _run_whole(session, _Joint(), [array], reads, where, chunked=True, partition=part)[
            id(array)
        ]
        rows, found = _first(value, n)
        if exhausted or found == n:
            return rows
        entries *= 2
