"""The repartition/join frontend surface + the multi-stage plan builders (plan M39 §3.1-§3.3, m68c).

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
every row to a dest; a ``gather``/``gather_join`` stage concatenates (or joins) each dest's blocks and
evaluates what was recorded after the barrier; an optional one-task ``reduce`` stage folds the
per-dest values. Every stage follows the ``process(task, inputs, resources) -> bytes`` convention of
``graphed.checkpoint.run_shuffle_resumable``; ``SequentialRunner`` and graphed-executors'
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
from typing import Any, Generic, TypeVar

from graphed.core import DurablePlanV2, GraphStore, OpSpec, Partition, StageSpec, Task
from graphed.core.execution import WorkerResources

from .aggregate import attribute_failures, external_evaluators, plan_services, resolve_backend
from .array import Array
from .backend import ParamValue
from .errors import GraphedError
from .execute import Frame, Key, compile_ir, evaluate_ir, external_key, ir_cone, refuse_chunk_partials
from .projection import read_columns
from .services import Bindable, Resolvable, bind_externals
from .session import Session
from .varied import refuse_boundary
from .write import PartitionedSource, declared_columns

V = TypeVar("V")
#: a plan's ``(reduce, combine, empty)``
_Monoid = tuple[Callable[[list[Any]], Any], Callable[[Any, Any], Any], Callable[[], Any]]


@dataclass(frozen=True)
class _GatherReduce(Generic[V]):
    """A picklable reduction monoid over blocks: ``reduce`` each block, ``combine`` the results onto
    ``empty()``. No builder emits it since the V2 stages evaluate the IR (m68c)."""

    reduce: Callable[[list[Any]], V]
    combine: Callable[[V, V], V]
    empty: Callable[[], V]

    def __call__(self, blocks: Sequence[Any]) -> V:
        return functools.reduce(self.combine, (self.reduce([b]) for b in blocks), self.empty())


def _scheme_params(*, by: str | None, n: int | None, target_bytes: int | None) -> dict[str, ParamValue]:
    """Map the user's intent to an ``Exchange`` scheme (its structural identity): a key -> hash route,
    a byte target -> coalesce, a partition count -> count."""
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
    """Repartition ``array`` (plan §3.1). ``by=<field>`` records a hash ``Exchange`` keyed on that
    field (the join/keyed-shuffle case — a neutral module verb, NOT an ``Array`` method);
    ``target_bytes=`` coalesces by measured size; ``n=`` sets a target partition count."""
    refuse_boundary("graphed.repartition", array)
    return array.session.record_exchange(array, _scheme_params(by=by, n=n, target_bytes=target_bytes))


JOINKEY = "__joinkey__"


def pack_key(array: Array, *, on: Sequence[str]) -> Array:
    """Record a neutral ``pack_key`` op that adds a flat unsigned-64 ``__joinkey__`` column derived
    from the ``on`` fields (plan §2.1/§3.3, spec Impl Target 8). A fusible ``Op`` (not a boundary), so
    it folds into the map stage; the backend computes it by big-endian integer bit-ops (never Python
    ``hash()``). :func:`join` uses it internally; it is also public so a caller can pre-key a source."""
    refuse_boundary("graphed.pack_key", array)
    return array.session.record_op("pack_key", [array], {"on": ",".join(on)})


def join(left: Array, right: Array, *, on: Sequence[str], how: str = "inner") -> Array:
    """The neutral ``graphed.join`` MODULE verb (plan §3.1): a relational, SQL-*duplicating* join of
    two arrays on ``on`` (a probe row with k build matches ⇒ k output rows). A join is neither an
    awkward nor a numpy idiom, so — like :func:`repartition` — it is a module function, not an
    ``Array`` method. It records ``pack_key`` → hash ``Exchange`` on each side (co-partitioning them on
    the shared ``__joinkey__``) then a two-input ``Join`` boundary; the flat output record is the union
    of both sides' fields, the shared key columns COALESCED (a left/right/outer miss keeps the present
    side's key, never null). ``how`` ∈ {inner, left, right, outer} (SQL/pandas relational semantics)."""
    refuse_boundary("graphed.join", left, right)
    if left.session is not right.session:
        raise GraphedError("join: left and right must belong to the same Session")
    session = left.session
    lk = pack_key(left, on=on)
    rk = pack_key(right, on=on)
    le = session.record_exchange(lk, {"scheme": "hash", "key": JOINKEY})
    re = session.record_exchange(rk, {"scheme": "hash", "key": JOINKEY})
    # Match + coalesce on the full key set — the user's ``on`` fields AND the packed ``__joinkey__``
    # (comma-joined; IR params carry only scalars). Carrying the real keys lets ``merge_records``
    # COALESCE them on a left/right/outer miss row (the present side's key survives, never null), and
    # makes the match disambiguate any ``__joinkey__`` collision by the real fields.
    return session.record_join(le, re, {"on": ",".join([*on, JOINKEY]), "how": how})


def join_blocks(
    backend: Any, left: Any, right: Any, *, on: Sequence[str] = (JOINKEY,), how: str = "inner"
) -> Any:
    """The generic radix-hash join KERNEL over a ``JoinBackend`` (plan §3.3b): match co-partitioned
    blocks, gather both sides by the aligned indices, merge to one flat relational record. Backend-
    agnostic — it calls ONLY ``JoinBackend`` primitives, so the same kernel drives every backend and
    no awkward/numpy leaks into ``graphed``. Used by the reference ``eval_stage("join")`` and by the
    two-phase executor's gather-join."""
    build_idx, probe_idx = backend.match_indices(left, right, on=list(on), how=how)
    return backend.merge_records(backend.take(left, build_idx), backend.take(right, probe_idx), on=list(on))


def partition_block(
    backend: Any, block: Any, *, parts: int, salt: int = 0, boundaries: object = None
) -> tuple[Any, ...]:
    """Route a block's rows to ``parts`` sub-blocks by the pinned hash of ``__joinkey__`` (a
    ``ShuffleBackend`` primitive)."""
    return backend.partition(block, JOINKEY, parts, salt=salt, boundaries=boundaries)  # type: ignore[no-any-return]


def _backend_identity(session: Session, backend: Callable[[], Any] | str | None) -> str:
    """The backend's shuffle-format ``identity`` token (folded into the V2 task ids, §7.2). Defaults
    to the session backend; a factory/class/``"module:attr"`` ref is resolved to an instance."""
    be = session.backend if backend is None else resolve_backend(backend)
    return str(getattr(be, "identity", "unknown/0"))


def split(payload: bytes) -> dict[int, bytes]:
    """A map-write payload as ``{dest: wire block}``."""
    mapping: dict[int, bytes] = pickle.loads(payload)
    return mapping


def pick(mapping: Mapping[int, bytes], dest: int) -> bytes:
    """The payload a gather of ``dest`` reads from one map task: that dest's block alone, in the
    map-write payload format."""
    return pickle.dumps({dest: mapping[dest]}, protocol=5)


@dataclass(frozen=True)
class _MapWrite:
    """A ``map_write`` task: read one partition of ``source_name``, evaluate the IR up to ``target``
    (this side's barrier input), and route the value to every dest — a zero-row block of its type where
    none of its rows go. ``route`` is ``(scheme, key, parts)``: ``hash`` routes rows by ``key``,
    ``count`` sends the whole block to dest ``task.key % parts``."""

    ir: bytes
    target: int
    source_name: str
    reader: PartitionedSource
    columns: tuple[str, ...] | None
    backend_factory: Callable[[], Any] | str
    externals: tuple[tuple[str, Callable[..., object]], ...]
    frames: tuple[tuple[Key, Frame], ...]
    route: tuple[str, str, int]

    def __call__(self, task: Task, inputs: Sequence[bytes], resources: WorkerResources) -> bytes:
        backend = resolve_backend(self.backend_factory)
        chunk = self.reader.read_partition(task.partition, self.columns, resources)
        (value,) = evaluate_ir(
            self.ir,
            backend,
            {self.source_name: chunk},
            externals=dict(self.externals),
            on_failure=attribute_failures(self.frames, str(task.partition)),
            outputs=(self.target,),
        )
        scheme, key, parts = self.route
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
    """A ``gather``/``gather_join`` task for dest ``task.key``: concatenate that dest's blocks per side
    (the first ``n_left`` inputs are a join's left side), then evaluate the IR from the barrier's
    inputs on — the barrier itself (an exchange is the identity, a join the backend's join kernel) and
    everything recorded after it. The payload is ``reduce(values)``, or the one output's value."""

    ir: bytes
    barrier_inputs: tuple[int, ...]
    backend_factory: Callable[[], Any] | str
    externals: tuple[tuple[str, Callable[..., object]], ...]
    frames: tuple[tuple[Key, Frame], ...]
    n_left: int | None
    reduce: Callable[[list[Any]], Any] | None

    def __call__(self, task: Task, inputs: Sequence[bytes], resources: WorkerResources) -> bytes:
        backend = resolve_backend(self.backend_factory)
        blocks = [backend.from_wire(split(p)[task.key]) for p in inputs]
        sides = [blocks] if self.n_left is None else [blocks[: self.n_left], blocks[self.n_left :]]
        values = evaluate_ir(
            self.ir,
            backend,
            {},
            externals=dict(self.externals),
            on_failure=attribute_failures(self.frames, f"dest {task.key}"),
            given={nid: backend.concat(side) for nid, side in zip(self.barrier_inputs, sides, strict=True)},
        )
        return pickle.dumps(values[0] if self.reduce is None else self.reduce(values), protocol=5)

    def bind_services(self, endpoints: Mapping[str, str]) -> _Gather:
        reduce = self.reduce.bind_services(endpoints) if isinstance(self.reduce, Bindable) else self.reduce
        return replace(self, externals=bind_externals(self.externals, endpoints), reduce=reduce)

    def resolve_services(self, value: Any) -> Any:
        return self.reduce.resolve_services(value) if isinstance(self.reduce, Resolvable) else value


@dataclass(frozen=True)
class _Fold:
    """The one-task ``reduce`` stage: ``combine`` the decoded gather payloads, in dest order, onto
    ``empty()``."""

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
    steps_per_file: int = 1,
    services: Sequence[str] | None = None,
) -> DurablePlanV2:
    """Build the :class:`~graphed.core.DurablePlanV2` of a repartition (plan §3.2, §4.4): ``map_write``
    over the session's partitioned source, ``gather`` per dest (``reduce`` over its output values), and
    a one-task ``reduce`` stage that folds the dests with ``combine``/``empty``. The barrier is the
    graph's first ``Exchange``; a ``target_bytes=`` one is refused (see the module docstring).
    ``services`` names services beyond those the graph's External nodes name, as in ``aggregate_plan``."""
    refuse_boundary("graphed.shuffle_plan", output)
    fold = (reduce, combine, empty)
    return _build(
        "shuffle_plan", output, fold, backend=backend, steps_per_file=steps_per_file, names=services
    )


def broadcast_join_choice(build_size: int, probe_size: int, n: int) -> bool:
    """The pinned broadcast-vs-shuffle cost rule (plan §3.3, theme (c); E5/F6): broadcast the build
    side IFF replicating it ``n``-fold is cheaper than shuffling BOTH sides —
    ``|build|·n < |build|+|probe|``. The single source of truth for the rule: :func:`join_plan` calls
    it at plan-build time with a PLAN-STABLE ``n`` (each side's own partition count — typetracer forms
    carry no row count, so a byte estimate isn't available pre-execution, R7.9) and freezes the result
    into the durable plan; ``graphed_exec_local.shuffle`` re-exports this same function for its
    ``run_join(broadcast=None)`` auto-choice (also keyed on ``parts``, never the runtime worker count —
    the F6 bug was recomputing this from the live worker pool)."""
    return build_size * n < build_size + probe_size


def join_plan(
    output: Array,
    *,
    backend: Callable[[], Any] | str | None = None,
    steps_per_file: int = 1,
    reduce: Callable[[list[Any]], V] | None = None,
    combine: Callable[[V, V], V] | None = None,
    empty: Callable[[], V] | None = None,
    services: Sequence[str] | None = None,
) -> DurablePlanV2:
    """Build the :class:`~graphed.core.DurablePlanV2` of a two-source ``graphed.join`` (plan §3.2,
    contract E1/target 13): one ``map_write`` stage per join input (stage ``i`` reads input ``i``'s
    source), co-partitioned on ``__joinkey__``, then a ``gather_join`` over both (``inputs=(0, 1)``) that
    joins each dest and runs what was recorded after the join. Given ``reduce``/``combine``/``empty``
    (all or none), the gather reduces its values and a one-task ``reduce`` stage folds the dests;
    without them the plan's value is the tuple of per-dest outputs. The recorded ``broadcast`` routing
    field is the block engines' choice; a plan always hash-routes both sides."""
    refuse_boundary("graphed.join_plan", output)
    fold: _Monoid | None = None
    if reduce is not None and combine is not None and empty is not None:
        fold = (reduce, combine, empty)
    elif reduce is not None or combine is not None or empty is not None:
        raise TypeError("join_plan takes reduce, combine and empty together, or none of them")
    return _build("join_plan", output, fold, backend=backend, steps_per_file=steps_per_file, names=services)


def _build(
    builder: str,
    output: Array,
    fold: _Monoid | None,
    *,
    backend: Callable[[], Any] | str | None,
    steps_per_file: int,
    names: Sequence[str] | None,
) -> DurablePlanV2:
    is_join = builder == "join_plan"
    session = output.session
    compiled = compile_ir(session, output)
    ir = bytes(compiled.ir)
    store = GraphStore.deserialize(ir)
    nodes = store.nodes()
    barriers = [i for i, n in enumerate(nodes) if n["kind"] == ("join" if is_join else "exchange")]
    if not barriers:
        raise TypeError(
            "join_plan needs a Join boundary in the graph (use graphed.join)"
            if is_join
            else "shuffle_plan needs a repartition Exchange in the graph (use graphed.repartition)"
        )
    partitioned = {nid: d for nid, d in session.sources().items() if isinstance(d, PartitionedSource)}
    if len(partitioned) != (2 if is_join else 1):
        raise TypeError(
            f"{builder} needs exactly {'two' if is_join else 'one'} partitioned source"
            f"{'s' if is_join else ''}; this session has {len(partitioned)}"
        )
    barrier = barriers[0]
    params = dict(nodes[barrier]["params"])
    if params.get("scheme") == "coalesce":
        raise TypeError(
            "shuffle_plan cannot run repartition(target_bytes=): its dest count comes from sizes measured"
            " at run time; run it with graphed-executors' run_repartition_by_size"
        )
    how = str(params.get("how", "inner"))
    if params.get("grouped") and how in ("right", "outer"):
        raise TypeError(
            f"join_plan cannot run gak.join(grouped=True, how={how!r}) per dest: its unmatched build rows"
            " regroup across dests; use how='inner' or 'left', or grouped=False"
        )
    post = ir_cone(nodes, store.outputs(), stop=(barrier,))
    if any(nodes[i]["kind"] == "source" for i in post):
        raise TypeError(
            f"{builder}: an operation after the {nodes[barrier]['kind']} reads a source directly, so its"
            " rows would not line up with the barrier's; compute it before the barrier"
        )
    by_name = {session.source_name(nid): (nid, data) for nid, data in partitioned.items()}
    targets = tuple(nodes[barrier]["inputs"])
    sides: list[tuple[str, int, PartitionedSource]] = []
    for target in targets:
        read = [nodes[i]["name"] for i in ir_cone(nodes, (target,)) if nodes[i]["kind"] == "source"]
        if len(read) != 1 or read[0] not in by_name:
            raise TypeError(
                f"{builder}: each side of the {nodes[barrier]['kind']} must read exactly one partitioned"
                f" source; one reads {sorted(read)}"
            )
        sides.append((read[0], *by_name[read[0]]))
    refuse_chunk_partials(compiled, as_outputs=fold is None)

    backend_id = _backend_identity(session, backend)
    factory = backend if backend is not None else type(session.backend)
    wired = external_evaluators(session, compiled)
    frames = compiled.correspondence.frames
    if is_join:
        exchanges = [n for n in nodes if n["kind"] == "exchange"]
        parts = (
            int(dict(exchanges[0]["params"]).get("parts", steps_per_file)) if exchanges else steps_per_file
        )
        route = ("hash", JOINKEY, parts)
        routing: dict[str, Any] = {"scheme": "hash", "key": JOINKEY, "parts": parts, "backend_id": backend_id}
    else:
        parts = int(params.get("parts", steps_per_file))
        route = (str(params["scheme"]), str(params.get("key", "")), parts)
        routing = {**params, "parts": parts, "backend_id": backend_id}

    map_stages = []
    for target, (name, nid, data) in zip(targets, sides, strict=True):
        declared = declared_columns(data, [output])
        process = _MapWrite(
            ir=ir,
            target=target,
            source_name=name,
            reader=data,
            columns=read_columns([output], nid) if declared is None else declared,
            backend_factory=factory,
            externals=_cone_externals(nodes, ir_cone(nodes, (target,)), wired),
            frames=frames,
            route=route,
        )
        tasks = tuple(Task(i, p) for i, p in enumerate(data.partitions(steps_per_file)))
        map_stages.append(
            StageSpec(kind="map_write", process=OpSpec.from_callable(process), routing=routing, tasks=tasks)
        )
    gather = _Gather(
        ir=ir,
        barrier_inputs=targets,
        backend_factory=factory,
        externals=_cone_externals(nodes, post, wired),
        frames=frames,
        n_left=len(map_stages[0].tasks) if is_join else None,
        reduce=None if fold is None else fold[0],
    )
    gather_routing: dict[str, Any] = {"parts": parts, "backend_id": backend_id}
    if is_join:
        # E5/F6: the broadcast-vs-shuffle choice is a PLAN property, keyed on each side's partition
        # count (typetracer forms carry no row count, R7.9) — never the runtime worker pool.
        broadcast = broadcast_join_choice(len(map_stages[0].tasks), len(map_stages[1].tasks), parts)
        gather_routing.update(how=how, broadcast=broadcast)
    stages = [
        *map_stages,
        StageSpec(
            kind="gather_join" if is_join else "gather",
            inputs=tuple(range(len(map_stages))),  # the barrier edge: every map-write stage
            process=OpSpec.from_callable(gather),
            routing=gather_routing,
            tasks=tuple(Task(d, Partition("dest", "p", d, d + 1)) for d in range(parts)),
        ),
    ]
    if fold is not None:
        stages.append(
            StageSpec(
                kind="reduce",
                inputs=(len(map_stages),),
                process=OpSpec.from_callable(_Fold(fold[1], fold[2])),
                routing={"backend_id": backend_id},
                tasks=(Task(0, Partition("reduce", "", 0, 1)),),
            )
        )
    return DurablePlanV2(ir=ir, stages=tuple(stages), services=plan_services(session, compiled, names))


def _cone_externals(
    nodes: Sequence[Mapping[str, Any]], cone: set[int], wired: Mapping[str, Callable[..., object]]
) -> tuple[tuple[str, Callable[..., object]], ...]:
    """The wired External evaluators whose nodes lie in ``cone``."""
    keys = {external_key(nodes[i]) for i in cone if nodes[i]["kind"] == "external"}
    return tuple((key, fn) for key, fn in wired.items() if key in keys)
