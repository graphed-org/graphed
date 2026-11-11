"""The repartition/join frontend surface + the multi-stage plan builders.

Plan M39 §3.1-§3.3, m68c.

Two recording entry points, both backend-agnostic (graphed imports no numpy/awkward):

- :func:`repartition` — the neutral MODULE verb for a KEYED repartition (``by=`` a field). A keyed
  shuffle is neither an awkward nor a numpy idiom, so per the factorization rule it is a module
  function, not an ``Array`` method. It records a hash ``Exchange``. Count/size rebalancing is
  *physical* (moves rows, no idiom) and stays on ``Array.repartition``, which delegates here.
- :func:`join` — the neutral relational join verb, recorded as ``pack_key`` → hash ``Exchange`` per
  side → a two-input ``Join`` boundary.

The builders :func:`shuffle_plan` and :func:`join_plan` serialize such a graph as a multi-stage
:class:`~graphed.core.DurablePlanV2` whose stage processes evaluate the plan's one compiled IR on
either side of the barrier (the first ``Exchange`` of a shuffle, the ``Join`` of a join): a
``map_write`` stage per source reads each partition, evaluates up to its barrier input, and routes
every row to a dest; a ``gather``/``gather_join`` stage concatenates (or joins) each dest's blocks
and evaluates what was recorded after the barrier; an optional one-task ``reduce`` stage folds the
per-dest values. Every stage follows the ``process(task, inputs, resources) -> bytes`` convention
of ``graphed.checkpoint.run_shuffle_resumable``; ``SequentialRunner`` and graphed-executors'
``SubmitRunner`` run the same plans. A map-write payload is ``pickle({dest: wire block})``
(:func:`split`/:func:`pick`), every other payload the pickled value.

A size-driven repartition (``target_bytes=``) fixes its dest count from sizes measured at run time,
which a plan's build-time task list cannot express: it runs through graphed-executors'
``run_repartition_by_size``, the block engine over the ``ShuffleBackend`` protocol.
"""

from __future__ import annotations

import functools
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, TypeVar

from graphed.core import DurablePlanV2, GraphStore, OpSpec, Partition, StageSpec, Task
from graphed.core.execution import WorkerResources

from .aggregate import (
    _adopting,
    _partitioned,
    attribute_failures,
    external_evaluators,
    plan_services,
    read_label,
    resolve_backend,
    worker_backend,
)
from .array import Array
from .backend import ParamValue
from .errors import GraphedError
from .execute import (
    CompiledGraph,
    Frame,
    Key,
    compile_ir,
    evaluate_ir,
    external_key,
    ir_cone,
    refuse_chunk_partials,
)
from .projection import read_columns
from .services import Bindable, Resolvable, bind_externals
from .session import Session
from .varied import refuse_boundary
from .write import PartitionedSource, declared_columns, steps_for

V = TypeVar("V")
#: a plan's ``(reduce, combine, empty)``
_Monoid = tuple[Callable[[list[Any]], Any], Callable[[Any, Any], Any], Callable[[], Any]]


def _scheme_params(
    *, by: str | None, n: int | None, target_bytes: int | None
) -> dict[str, ParamValue]:
    """Map the user's intent to an ``Exchange`` scheme (its structural identity).

    A key -> hash route, a byte target -> coalesce, a partition count -> count.
    """
    if by is not None:
        return {"scheme": "hash", "key": by}
    if target_bytes is not None:
        return {"scheme": "coalesce", "target_bytes": target_bytes}
    if n is not None:
        return {"scheme": "count", "parts": n}
    raise TypeError("repartition needs one of by=, n=, or target_bytes=")


def repartition(
    array: Array,
    *,
    by: str | None = None,
    n: int | None = None,
    target_bytes: int | None = None,
) -> Array:
    """Repartition ``array`` (plan §3.1).

    ``by=<field>`` records a hash ``Exchange`` keyed on that field (the join/keyed-shuffle case — a
    neutral module verb, NOT an ``Array`` method); ``target_bytes=`` coalesces by measured size;
    ``n=`` sets a target partition count.
    """
    refuse_boundary("graphed.repartition", array)
    return array.session.record_exchange(
        array, _scheme_params(by=by, n=n, target_bytes=target_bytes)
    )


JOINKEY = "__joinkey__"


def pack_key(array: Array, *, on: Sequence[str]) -> Array:
    """Record a neutral ``pack_key`` op that adds a flat unsigned-64 ``__joinkey__`` column.

    The column is derived from the ``on`` fields (plan §2.1/§3.3, spec Impl Target 8). A fusible
    ``Op`` (not a boundary), so it folds into the map stage; the backend computes it by big-endian
    integer bit-ops (never Python ``hash()``). It is public so a caller can pre-key a source.
    """
    refuse_boundary("graphed.pack_key", array)
    return array.session.record_op("pack_key", [array], {"on": ",".join(on)})


def route_key(array: Array, on: Sequence[str]) -> Array:
    """Record ``pack_key`` for a join's routing: the match compares the real fields too.

    So a value too wide for its field only shares a route, and is not refused.
    """
    return array.session.record_op("pack_key", [array], {"on": ",".join(on), "route": True})


def join(left: Array, right: Array, *, on: Sequence[str] = (), how: str = "inner") -> Array:
    """Return the neutral ``graphed.join`` MODULE verb (plan §3.1).

    A relational, SQL-*duplicating* join of two arrays on ``on`` (a probe row with k build matches
    ⇒ k output rows). A join is neither an awkward nor a numpy idiom, so — like :func:`repartition`
    — it is a module function, not an ``Array`` method. It records ``pack_key`` → hash ``Exchange``
    on each side (co-partitioning them on the shared ``__joinkey__``) then a two-input ``Join``
    boundary; the flat output record is the union of both sides' fields, the shared key columns
    COALESCED (a left/right/outer miss keeps the present side's key, never null). ``how`` ∈ {inner,
    left, right, outer} (SQL/pandas relational semantics).

    ``how="positional"`` takes no ``on``: row i pairs with row i, for two sources holding the same
    events in the same order. It reads no key and moves no rows between partitions; a differing
    row count is refused when it runs, so a one-row side never broadcasts.
    """
    refuse_boundary("graphed.join", left, right)
    if left.session is not right.session:
        raise GraphedError(
            "join: left and right must belong to the same Session; record both into one (read the "
            "right side inside `with left.session:`, or pass the reader session=)"
        )
    session = left.session
    if (how == "positional") == bool(on):
        raise TypeError("graphed.join takes on=[...], or how='positional' and no on=")
    if how == "positional":
        return session.record_join(left, right, {"on": "", "how": how})
    lk = route_key(left, on)
    rk = route_key(right, on)
    le = session.record_exchange(lk, {"scheme": "hash", "key": JOINKEY})
    re = session.record_exchange(rk, {"scheme": "hash", "key": JOINKEY})
    # Match + coalesce on the full key set — the user's ``on`` fields AND the packed
    # ``__joinkey__`` (comma-joined; IR params carry only scalars). Carrying the real keys lets
    # ``merge_records`` COALESCE them on a left/right/outer miss row (the present side's key
    # survives, never null), and makes the match disambiguate any ``__joinkey__`` collision by the
    # real fields.
    return session.record_join(le, re, {"on": ",".join([*on, JOINKEY]), "how": how})


def join_blocks(
    backend: Any, left: Any, right: Any, *, on: Sequence[str] = (JOINKEY,), how: str = "inner"
) -> Any:
    """Return the generic radix-hash join KERNEL over a ``JoinBackend`` (plan §3.3b).

    Match co-partitioned blocks, gather both sides by the aligned indices, merge to one flat
    relational record. Backend- agnostic — it calls ONLY ``JoinBackend`` primitives, so the same
    kernel drives every backend and no awkward/numpy leaks into ``graphed``. Used by the reference
    ``eval_stage("join")`` and by the two-phase executor's gather-join.
    """
    if how == "positional":
        if len(left) != len(right):
            raise ValueError(f"how='positional' got {len(left)} rows against {len(right)}")
        return backend.merge_records(left, right, on=[])
    build_idx, probe_idx = backend.match_indices(left, right, on=list(on), how=how)
    return backend.merge_records(
        backend.take(left, build_idx), backend.take(right, probe_idx), on=list(on)
    )


def _backend_identity(session: Session, backend: Callable[[], Any] | str | None) -> str:
    """Return the backend's shuffle-format ``identity`` token (folded into the V2 task ids, §7.2).

    Defaults to the session backend; a factory/class/``"module:attr"`` ref is resolved to an
    instance.
    """
    be = session.backend if backend is None else resolve_backend(backend)
    return str(getattr(be, "identity", "unknown/0"))


def split(payload: bytes) -> dict[int, bytes]:
    """Return a map-write payload as ``{dest: wire block}``."""
    mapping: dict[int, bytes] = pickle.loads(payload)
    return mapping


def pick(mapping: Mapping[int, bytes], dest: int) -> bytes:
    """Return the payload a gather of ``dest`` reads from one map task.

    That dest's block alone, in the map-write payload format, and nothing from a ``position`` map
    task of another dest.
    """
    return pickle.dumps({dest: mapping[dest]} if dest in mapping else {}, protocol=5)


@dataclass(frozen=True)
class _MapWrite:
    """A ``map_write`` task.

    Read one partition of ``source_name``, evaluate the IR up to ``target`` (this side's barrier
    input), and route the value to every dest — a zero-row block of its type where none of its rows
    go. ``route`` is ``(scheme, key, parts)``: ``hash`` routes rows by ``key``, ``count`` sends the
    whole block to dest ``task.key % parts``. ``position`` sends it to dest ``task.key`` alone: the
    plan has one dest per task, so a block for every other would be as many empties as tasks.
    """

    checkpointable: ClassVar[bool] = True
    checkpoint_ignore: ClassVar[tuple[str, ...]] = ("frames", "variation_labels")
    checkpoint_resolve: ClassVar[Mapping[str, Callable[[Any], Any]]] = {
        "backend_factory": resolve_backend
    }

    ir: bytes
    target: int
    source_name: str
    reader: PartitionedSource
    columns: tuple[str, ...] | None
    backend_factory: Callable[[], Any] | str
    externals: tuple[tuple[str, Callable[..., object]], ...]
    frames: tuple[tuple[Key, Frame], ...]
    route: tuple[str, str, int]
    variation_labels: tuple[Any, ...] | None

    def __call__(self, task: Task, inputs: Sequence[bytes], resources: WorkerResources) -> bytes:
        backend = resolve_backend(self.backend_factory)
        chunk = self.reader.read_partition(task.partition, self.columns, resources)
        (value,) = evaluate_ir(
            self.ir,
            backend,
            {self.source_name: chunk},
            externals=dict(self.externals),
            on_failure=attribute_failures(
                self.frames,
                functools.partial(read_label, self.reader, task.partition, chunk),
                self.variation_labels,
            ),
            outputs=(self.target,),
            chunked=True,
        )
        scheme, key, parts = self.route
        if scheme == "position":
            return pickle.dumps({task.key: backend.to_wire(value)}, protocol=5)
        if scheme == "hash":
            blocks = tuple(backend.partition(value, key, parts))
        else:
            empty = backend.slice_rows(value, 0, 0)
            blocks = tuple(value if d == task.key % parts else empty for d in range(parts))
        return pickle.dumps({d: backend.to_wire(b) for d, b in enumerate(blocks)}, protocol=5)

    def bind_services(self, endpoints: Mapping[str, str]) -> _MapWrite:
        return replace(self, externals=bind_externals(self.externals, endpoints))


@dataclass(frozen=True)
class _Gather:
    """A ``gather``/``gather_join`` task for dest ``task.key``.

    Concatenate that dest's blocks per side (the first ``n_left`` inputs are a join's left side),
    then evaluate the IR from the barrier's inputs on — the barrier itself (an exchange is the
    identity, a join the backend's join kernel) and everything recorded after it. The payload is
    ``reduce(values)``, or the one output's value.
    """

    checkpointable: ClassVar[bool] = True
    checkpoint_ignore: ClassVar[tuple[str, ...]] = ("frames", "variation_labels")
    checkpoint_resolve: ClassVar[Mapping[str, Callable[[Any], Any]]] = {
        "backend_factory": resolve_backend
    }

    ir: bytes
    barrier_inputs: tuple[int, ...]
    backend_factory: Callable[[], Any] | str
    externals: tuple[tuple[str, Callable[..., object]], ...]
    frames: tuple[tuple[Key, Frame], ...]
    n_left: int | None
    reduce: Callable[[list[Any]], Any] | None
    variation_labels: tuple[Any, ...] | None

    def __call__(self, task: Task, inputs: Sequence[bytes], resources: WorkerResources) -> bytes:
        backend = resolve_backend(self.backend_factory)
        mappings = [split(p) for p in inputs]
        blocks = [backend.from_wire(m[task.key]) for m in mappings if task.key in m]
        sides = [blocks] if self.n_left is None else [blocks[: self.n_left], blocks[self.n_left :]]
        values = evaluate_ir(
            self.ir,
            backend,
            {},
            externals=dict(self.externals),
            on_failure=attribute_failures(self.frames, f"dest {task.key}", self.variation_labels),
            given={
                nid: backend.concat(side)
                for nid, side in zip(self.barrier_inputs, sides, strict=True)
            },
            chunked=True,
        )
        value = values[0] if self.reduce is None else self.reduce(values)
        adopt = getattr(backend, "adopt", None)
        return pickle.dumps(value if adopt is None else adopt(value), protocol=5)

    def bind_services(self, endpoints: Mapping[str, str]) -> _Gather:
        reduce = (
            self.reduce.bind_services(endpoints)
            if isinstance(self.reduce, Bindable)
            else self.reduce
        )
        return replace(self, externals=bind_externals(self.externals, endpoints), reduce=reduce)

    def resolve_services(self, value: Any) -> Any:
        return (
            self.reduce.resolve_services(value) if isinstance(self.reduce, Resolvable) else value
        )


@dataclass(frozen=True)
class _Fold:
    """The one-task ``reduce`` stage.

    ``combine`` the decoded gather payloads, in dest order, onto ``empty()``.
    """

    checkpointable: ClassVar[bool] = True

    combine: Callable[[Any, Any], Any]
    empty: Callable[[], Any]

    def __call__(self, task: Task, inputs: Sequence[bytes], resources: WorkerResources) -> bytes:
        return pickle.dumps(
            functools.reduce(self.combine, map(pickle.loads, inputs), self.empty()), protocol=5
        )


def shuffle_plan(
    output: Array,
    *,
    reduce: Callable[[list[Any]], V],
    combine: Callable[[V, V], V],
    empty: Callable[[], V],
    backend: Callable[[], Any] | str | None = None,
    steps_per_file: int | None = None,
    services: Sequence[str] | None = None,
) -> DurablePlanV2:
    """Build the :class:`~graphed.core.DurablePlanV2` of a repartition (plan §3.2, §4.4).

    ``map_write`` over the session's partitioned source, ``gather`` per dest (``reduce`` over its
    output values), and a one-task ``reduce`` stage that folds the dests with
    ``combine``/``empty``. The barrier is the graph's first ``Exchange``; a ``target_bytes=`` one
    is refused (see the module docstring). ``services`` names services beyond those the graph's
    External nodes name, and ``steps_per_file`` splits the source's files, as in
    ``aggregate_plan``.
    """
    refuse_boundary("graphed.shuffle_plan", output)
    fold = (reduce, combine, empty)
    return _build(
        "shuffle_plan",
        (output,),
        fold,
        backend=backend,
        steps_per_file=steps_per_file,
        names=services,
    )


def broadcast_join_choice(build_size: int, probe_size: int, n: int) -> bool:
    """Return the pinned broadcast-vs-shuffle cost rule (plan §3.3, theme (c); E5/F6).

    Broadcast the build side IFF replicating it ``n``-fold is cheaper than shuffling BOTH sides —
    ``|build|·n < |build|+|probe|``. The single source of truth for the rule: :func:`join_plan`
    calls it at plan-build time with a PLAN-STABLE ``n`` (each side's own partition count —
    typetracer forms carry no row count, so a byte estimate isn't available pre-execution, R7.9)
    and freezes the result into the durable plan; ``graphed_exec_local.shuffle`` re-exports this
    same function for its ``run_join(broadcast=None)`` auto-choice (also keyed on ``parts``, never
    the runtime worker count — the F6 bug was recomputing this from the live worker pool).
    """
    return build_size * n < build_size + probe_size


def join_plan(
    *outputs: Array,
    backend: Callable[[], Any] | str | None = None,
    steps_per_file: int | None = None,
    reduce: Callable[[list[Any]], V] | None = None,
    combine: Callable[[V, V], V] | None = None,
    empty: Callable[[], V] | None = None,
    services: Sequence[str] | None = None,
) -> DurablePlanV2:
    """Build the :class:`~graphed.core.DurablePlanV2` of a two-source ``graphed.join``.

    Plan §3.2, contract E1/target 13: one ``map_write`` stage per join input (stage ``i`` reads
    input ``i``'s source), co-partitioned on ``__joinkey__``, then a ``gather_join`` over both
    (``inputs=(0, 1)``) that joins each dest and runs what was recorded after the join. Given
    ``reduce``/``combine``/``empty`` (all or none), the gather reduces its values and a one-task
    ``reduce`` stage folds the dests; without them the plan's value is the tuple of per-dest
    outputs. The recorded ``broadcast`` routing field is the block engines' choice; a plan always
    hash-routes both sides. ``steps_per_file`` splits each source's files as in ``aggregate_plan``
    and, absent a repartition, the larger side's steps per file is the dest count. A positional
    join routes no key: dest i pairs block i of each side, so the dest count is their partition
    count.

    Several ``outputs`` need the fold: ``reduce`` gets their values, as in ``aggregate_plan``.
    """
    refuse_boundary("graphed.join_plan", *outputs)
    fold: _Monoid | None = None
    if reduce is not None and combine is not None and empty is not None:
        fold = (reduce, combine, empty)
    elif reduce is not None or combine is not None or empty is not None:
        raise TypeError("join_plan takes reduce, combine and empty together, or none of them")
    if not outputs or (len(outputs) > 1 and fold is None):
        raise TypeError("join_plan takes one output, or several with reduce, combine and empty")
    return _build(
        "join_plan", outputs, fold, backend=backend, steps_per_file=steps_per_file, names=services
    )


def _barrier_sides(
    builder: str, outputs: Sequence[Array], store: GraphStore
) -> tuple[int, tuple[int, ...], list[tuple[str, int, PartitionedSource]], set[int]]:
    """Return the barrier node, its inputs, the source each input reads, and the cone after it.

    Refuses a graph ``builder`` cannot run.
    """
    is_join = builder == "join_plan"
    session = outputs[0].session
    nodes = store.nodes()
    barriers = [i for i, n in enumerate(nodes) if n["kind"] == ("join" if is_join else "exchange")]
    if not barriers:
        raise TypeError(
            "join_plan needs a Join boundary in the graph (use graphed.join)"
            if is_join
            else "shuffle_plan needs a repartition Exchange in the graph (use graphed.repartition)"
        )
    partitioned = _partitioned(outputs)
    if len(partitioned) != (2 if is_join else 1):
        raise TypeError(
            f"{builder} needs exactly {'two' if is_join else 'one'} partitioned source"
            f"{'s' if is_join else ''}; the output reads {len(partitioned)}"
        )
    barrier = barriers[0]
    params = dict(nodes[barrier]["params"])
    if params.get("scheme") == "coalesce":
        raise TypeError(
            "shuffle_plan cannot run repartition(target_bytes=): its dest count comes from sizes "
            "measured at run time; run it with graphed-executors' run_repartition_by_size"
        )
    how = str(params.get("how", "inner"))
    if params.get("grouped") and how in ("right", "outer"):
        raise TypeError(
            f"join_plan cannot run gak.join(grouped=True, how={how!r}) per dest: its unmatched "
            "build rows regroup across dests; use how='inner' or 'left', or grouped=False"
        )
    post = ir_cone(nodes, store.outputs(), stop=(barrier,))
    if any(nodes[i]["kind"] == "source" for i in post):
        raise TypeError(
            f"{builder}: an operation after the {nodes[barrier]['kind']} reads a source directly, "
            "so its rows would not line up with the barrier's; compute it before the barrier"
        )
    by_name = {session.source_name(nid): (nid, data) for nid, data in partitioned.items()}
    targets = tuple(nodes[barrier]["inputs"])
    sides: list[tuple[str, int, PartitionedSource]] = []
    for target in targets:
        read = [
            nodes[i]["name"] for i in ir_cone(nodes, (target,)) if nodes[i]["kind"] == "source"
        ]
        if len(read) != 1 or read[0] not in by_name:
            raise TypeError(
                f"{builder}: each side of the {nodes[barrier]['kind']} must read exactly one "
                f"partitioned source; one reads {sorted(read)}"
            )
        sides.append((read[0], *by_name[read[0]]))
    return barrier, targets, sides, post


def _routing(
    is_join: bool,
    nodes: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    steps_per_file: int,
    backend_id: str,
) -> tuple[int, tuple[str, str, int], dict[str, Any]]:
    """Return the destination count, the map-write route, and the map stages' routing record."""
    if is_join:
        exchanges = [n for n in nodes if n["kind"] == "exchange"]
        parts = (
            int(dict(exchanges[0]["params"]).get("parts", steps_per_file))
            if exchanges
            else steps_per_file
        )
        routing: dict[str, Any] = {
            "scheme": "hash",
            "key": JOINKEY,
            "parts": parts,
            "backend_id": backend_id,
        }
        return parts, ("hash", JOINKEY, parts), routing
    parts = int(params.get("parts", steps_per_file))
    route = (str(params["scheme"]), str(params.get("key", "")), parts)
    return parts, route, {**params, "parts": parts, "backend_id": backend_id}


def _build(
    builder: str,
    outputs: Sequence[Array],
    fold: _Monoid | None,
    *,
    backend: Callable[[], Any] | str | None,
    steps_per_file: int | None,
    names: Sequence[str] | None,
    compiled: CompiledGraph | None = None,
    externals: Mapping[str, Callable[..., object]] | None = None,
    on_compiled: Callable[[CompiledGraph], Any] | None = None,
    finalize: Callable[[Any], Any] | None = None,
) -> DurablePlanV2:
    is_join = builder == "join_plan"
    session = outputs[0].session
    compiled = compile_ir(session, *outputs) if compiled is None else compiled
    ir = bytes(compiled.ir)
    store = GraphStore.deserialize(ir)
    nodes = store.nodes()
    barrier, targets, sides, post = _barrier_sides(builder, outputs, store)
    params = dict(nodes[barrier]["params"])
    refuse_chunk_partials(compiled, as_outputs=fold is None)

    backend_id = _backend_identity(session, backend)
    factory = worker_backend(session, backend)
    wired = external_evaluators(session, compiled) | dict(externals or {})
    frames = compiled.correspondence.frames
    steps = [steps_for(data, steps_per_file) for _, _, data in sides]
    side_parts = [data.partitions(n) for n, (_, _, data) in zip(steps, sides, strict=True)]
    count = max(steps)
    if params.get("how") == "positional":
        # block i pairs with block i, and a differing row count in one is refused when it runs
        count, right_count = map(len, side_parts)
        if count != right_count:
            raise TypeError(
                "join_plan: how='positional' needs one partition per partition, and "
                f"{sides[0][0]!r} has {count} and {sides[1][0]!r} {right_count}"
            )
        parts, route, routing = _routing(False, nodes, {"scheme": "position"}, count, backend_id)
    else:
        parts, route, routing = _routing(is_join, nodes, params, count, backend_id)

    # past the refusals, since pieces take one call, and before `reduce`, which it fills, pickles
    variation_labels = None if on_compiled is None else on_compiled(compiled)
    map_stages = []
    for target, (name, nid, data), partitions in zip(targets, sides, side_parts, strict=True):
        declared = declared_columns(data, outputs)
        process = _MapWrite(
            ir=ir,
            target=target,
            source_name=name,
            reader=data,
            columns=read_columns(outputs, nid) if declared is None else declared,
            backend_factory=factory,
            externals=_cone_externals(nodes, ir_cone(nodes, (target,)), wired),
            frames=frames,
            route=route,
            variation_labels=variation_labels,
        )
        tasks = tuple(Task(i, p) for i, p in enumerate(partitions))
        map_stages.append(
            StageSpec(
                kind="map_write",
                process=OpSpec.from_callable(process),
                routing=routing,
                tasks=tasks,
            )
        )
    gather = _Gather(
        ir=ir,
        barrier_inputs=targets,
        backend_factory=factory,
        externals=_cone_externals(nodes, post, wired),
        frames=frames,
        # a `position` dest gets one block a side
        n_left=None if not is_join else 1 if route[0] == "position" else len(map_stages[0].tasks),
        reduce=None if fold is None else fold[0],
        variation_labels=variation_labels,
    )
    stages = [
        *map_stages,
        StageSpec(
            kind="gather_join" if is_join else "gather",
            inputs=tuple(range(len(map_stages))),  # the barrier edge: every map-write stage
            process=OpSpec.from_callable(gather),
            routing=_gather_routing(params, parts, backend_id, map_stages, is_join=is_join),
            tasks=tuple(Task(d, Partition("dest", "p", d, d + 1)) for d in range(parts)),
        ),
    ]
    if fold is not None:
        stages.append(_reduce_stage(fold, factory, backend_id, after=len(map_stages)))
    return DurablePlanV2(
        ir=ir,
        stages=tuple(stages),
        services=plan_services(session, compiled, names),
        finalize=None if finalize is None else OpSpec.from_callable(finalize),
    )


def _gather_routing(
    params: Mapping[str, Any],
    parts: int,
    backend_id: str,
    map_stages: Sequence[StageSpec],
    *,
    is_join: bool,
) -> dict[str, Any]:
    """Return the gather stage's routing: its parts, and for a join its how and broadcast."""
    routing: dict[str, Any] = {"parts": parts, "backend_id": backend_id}
    if is_join:
        # E5/F6: the broadcast-vs-shuffle choice is a PLAN property, keyed on each side's partition
        # count (typetracer forms carry no row count, R7.9) — never the runtime worker pool.
        broadcast = broadcast_join_choice(
            len(map_stages[0].tasks), len(map_stages[1].tasks), parts
        )
        routing.update(how=str(params.get("how", "inner")), broadcast=broadcast)
    return routing


def _reduce_stage(
    fold: _Monoid, factory: Callable[[], Any], backend_id: str, *, after: int
) -> StageSpec:
    """Return the one-task stage folding the gather stage ``after``'s outputs."""
    return StageSpec(
        kind="reduce",
        inputs=(after,),
        process=OpSpec.from_callable(
            _Fold(_adopting(fold[1], factory), _adopting(fold[2], factory))
        ),
        routing={"backend_id": backend_id},
        tasks=(Task(0, Partition("reduce", "", 0, 1)),),
    )


def _cone_externals(
    nodes: Sequence[Mapping[str, Any]], cone: set[int], wired: Mapping[str, Callable[..., object]]
) -> tuple[tuple[str, Callable[..., object]], ...]:
    """Return the wired External evaluators whose nodes lie in ``cone``."""
    keys = {external_key(nodes[i]) for i in cone if nodes[i]["kind"] == "external"}
    return tuple((key, fn) for key, fn in wired.items() if key in keys)
