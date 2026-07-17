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
