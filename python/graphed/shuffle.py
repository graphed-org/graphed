"""The repartition frontend surface + the multi-stage shuffle plan builder (plan M39 §3.1-§3.3).

Two recording entry points, both backend-agnostic (graphed imports no numpy/awkward):

- :func:`repartition` — the neutral MODULE verb for a KEYED repartition (``by=`` a field). A keyed
  shuffle is neither an awkward nor a numpy idiom, so per the factorization rule it is a module
  function, not an ``Array`` method. It records a hash ``Exchange``. Count/size rebalancing is
  *physical* (moves rows, no idiom) and stays on ``Array.repartition``, which delegates here.
- :func:`shuffle_plan` — the multi-source/multi-stage plan builder. The single-source
  ``aggregate_plan`` raises on a repartition boundary, so a shuffle serializes as a multi-stage
  :class:`~graphed.core.DurablePlanV2`: a stage-1 map-write (route + coalesce) that a stage-2 gather
  depends on.

The generic radix-split / coalesce-split / deterministic-merge ENGINE over the ``ShuffleBackend``
protocol is the two-phase executor ``graphed_exec_local.shuffle`` (``run_repartition`` /
``run_repartition_by_size``): it is backend-agnostic there (it deals only in opaque ``ShuffleBackend``
blocks) and its backend-agnosticism is witnessed by EXECUTION over BOTH real backends (the a-BI
theme). It lives with the executor rather than here because graphed's frozen suite records/plans but
never executes blocks — so hosting the block engine here would leave it uncovered by graphed's own
gate; the ShuffleBackend seam keeps it backend-neutral either way.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from graphed.core import DurablePlanV2, GraphStore, OpSpec, Partition, StageSpec, Task

from .aggregate import resolve_backend
from .array import Array
from .backend import ParamValue
from .errors import GraphedError
from .execute import compile_ir
from .session import Session
from .write import PartitionedSource

V = TypeVar("V")


@dataclass(frozen=True)
class _GatherReduce(Generic[V]):
    """The stage-2 gather's reduction monoid (``reduce`` per gathered block, ``combine`` across them,
    ``empty`` as the identity). Picklable (a frozen dataclass) so it rides in the durable plan."""

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
    return array.session.record_exchange(array, _scheme_params(by=by, n=n, target_bytes=target_bytes))


JOINKEY = "__joinkey__"


def pack_key(array: Array, *, on: Sequence[str]) -> Array:
    """Record a neutral ``pack_key`` op that adds a flat unsigned-64 ``__joinkey__`` column derived
    from the ``on`` fields (plan §2.1/§3.3, spec Impl Target 8). A fusible ``Op`` (not a boundary), so
    it folds into the map stage; the backend computes it by big-endian integer bit-ops (never Python
    ``hash()``). :func:`join` uses it internally; it is also public so a caller can pre-key a source."""
    return array.session.record_op("pack_key", [array], {"on": ",".join(on)})


def join(left: Array, right: Array, *, on: Sequence[str], how: str = "inner") -> Array:
    """The neutral ``graphed.join`` MODULE verb (plan §3.1): a relational, SQL-*duplicating* join of
    two arrays on ``on`` (a probe row with k build matches ⇒ k output rows). A join is neither an
    awkward nor a numpy idiom, so — like :func:`repartition` — it is a module function, not an
    ``Array`` method. It records ``pack_key`` → hash ``Exchange`` on each side (co-partitioning them on
    the shared ``__joinkey__``) then a two-input ``Join`` boundary; the flat output record is the union
    of both sides' fields, the shared key columns COALESCED (a left/right/outer miss keeps the present
    side's key, never null). ``how`` ∈ {inner, left, right, outer} (SQL/pandas relational semantics)."""
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
    """The stage-1 map-write routing kernel (plan §3.3b / §4): route a block's rows to ``parts``
    sub-blocks by the pinned hash of ``__joinkey__`` (a ``ShuffleBackend`` primitive). Module-level so
    a durable plan references it by import path, not by value."""
    return backend.partition(block, JOINKEY, parts, salt=salt, boundaries=boundaries)  # type: ignore[no-any-return]


def _backend_identity(session: Session, backend: Callable[[], Any] | str | None) -> str:
    """The backend's shuffle-format ``identity`` token (folded into the V2 task ids, §7.2). Defaults
    to the session backend; a factory/class/``"module:attr"`` ref is resolved to an instance."""
    be = session.backend if backend is None else resolve_backend(backend)
    return str(getattr(be, "identity", "unknown/0"))


def shuffle_plan(
    output: Array,
    *,
    reduce: Callable[[list[Any]], V],
    combine: Callable[[V, V], V],
    empty: Callable[[], V],
    backend: Callable[[], Any] | str | None = None,
    steps_per_file: int = 1,
) -> DurablePlanV2:
    """Build a multi-stage :class:`~graphed.core.DurablePlanV2` for a repartition/shuffle (plan §3.2,
    §4.4). Mirrors ``aggregate_plan``'s signature. The recorded graph must carry a repartition
    ``Exchange`` (the barrier); the plan is a stage-1 map-write (route + coalesce over the session's
    partitioned source) that a stage-2 gather depends on — the real intra-run barrier edge."""
    session = output.session
    compiled = compile_ir(session, output)
    store = GraphStore.deserialize(bytes(compiled.ir))
    exchanges = [n for n in store.nodes() if n["kind"] == "exchange"]
    if not exchanges:
        raise TypeError("shuffle_plan needs a repartition Exchange in the graph (use graphed.repartition)")

    partitioned = {nid: d for nid, d in session.sources().items() if isinstance(d, PartitionedSource)}
    if len(partitioned) != 1:
        raise TypeError(
            f"shuffle_plan needs exactly one partitioned source; this session has {len(partitioned)}"
        )
    ((_nid, data),) = partitioned.items()

    backend_id = _backend_identity(session, backend)
    scheme = dict(exchanges[0]["params"])
    parts = int(scheme.get("parts", steps_per_file))
    routing: dict[str, Any] = {**scheme, "parts": parts, "backend_id": backend_id}

    src_partitions = data.partitions(steps_per_file)
    map_tasks = tuple(Task(i, p) for i, p in enumerate(src_partitions))
    gather_tasks = tuple(Task(d, Partition("dest", "p", d, d + 1)) for d in range(parts))
    stages = (
        StageSpec(
            kind="map_write",
            inputs=(),
            process=OpSpec.from_callable(reduce),
            routing=routing,
            tasks=map_tasks,
        ),
        StageSpec(
            kind="gather",
            inputs=(0,),  # the barrier edge: the gather depends on the map-write stage
            process=OpSpec.from_callable(_GatherReduce(reduce, combine, empty)),
            routing={"parts": parts, "backend_id": backend_id},
            tasks=gather_tasks,
        ),
    )
    return DurablePlanV2(ir=bytes(compiled.ir), stages=stages)


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
) -> DurablePlanV2:
    """Build the multi-stage :class:`~graphed.core.DurablePlanV2` for a two-source ``graphed.join``
    (plan §3.2, contract E1/target 13). A join has TWO independently-partitioned sources, so the
    single-source ``shuffle_plan`` guard cannot express it: this emits ONE ``map_write`` stage per side
    (route + coalesce on ``__joinkey__``) that a single ``kind=\"gather_join\"`` stage — ``inputs`` over
    both map-writes — depends on (the two-input barrier edge). Cross-process execution is the
    executor's ``run_join``; this builder is the durable, byte-deterministic plan artifact."""
    session = output.session
    compiled = compile_ir(session, output)
    store = GraphStore.deserialize(bytes(compiled.ir))
    if not any(n["kind"] == "join" for n in store.nodes()):
        raise TypeError("join_plan needs a Join boundary in the graph (use graphed.join)")

    partitioned = {nid: d for nid, d in session.sources().items() if isinstance(d, PartitionedSource)}
    if len(partitioned) != 2:
        raise TypeError(
            f"join_plan needs exactly two partitioned sources; this session has {len(partitioned)}"
        )

    join_params = dict(next(n for n in store.nodes() if n["kind"] == "join")["params"])
    how = str(join_params.get("how", "inner"))
    exchanges = [n for n in store.nodes() if n["kind"] == "exchange"]
    parts = int(dict(exchanges[0]["params"]).get("parts", steps_per_file)) if exchanges else steps_per_file
    backend_id = _backend_identity(session, backend)
    routing: dict[str, Any] = {"scheme": "hash", "key": JOINKEY, "parts": parts, "backend_id": backend_id}

    # one map-write per side, in deterministic source order (by node id) -> the gather_join at the end
    sides = [data for _nid, data in sorted(partitioned.items())]
    map_stages = tuple(
        StageSpec(
            kind="map_write",
            inputs=(),
            process=OpSpec.from_callable(partition_block),
            routing=routing,
            tasks=tuple(Task(i, p) for i, p in enumerate(data.partitions(steps_per_file))),
        )
        for data in sides
    )
    # E5/F6: the broadcast-vs-shuffle choice is a PLAN property. Each side's own partition count is
    # the deterministic, no-file-opened size proxy (typetracer forms carry no row count, R7.9) — never
    # the runtime worker pool, which is the F6 bug the executor's auto-choice still had.
    build_n, probe_n = len(map_stages[0].tasks), len(map_stages[1].tasks)
    broadcast = broadcast_join_choice(build_n, probe_n, parts)
    gather = StageSpec(
        kind="gather_join",
        inputs=tuple(range(len(map_stages))),  # (0, 1): depends on BOTH map-writes (the barrier edge)
        process=OpSpec.from_callable(join_blocks),
        routing={"parts": parts, "backend_id": backend_id, "how": how, "broadcast": broadcast},
        tasks=tuple(Task(d, Partition("dest", "p", d, d + 1)) for d in range(parts)),
    )
    return DurablePlanV2(ir=bytes(compiled.ir), stages=(*map_stages, gather))
