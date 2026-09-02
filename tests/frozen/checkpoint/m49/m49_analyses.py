"""Module-level closure operands for the m49 checkpoint suite (§7.3, §7.4, §8.2(i)).

Everything a `DurablePlan` embeds BY VALUE lives here, at module scope: a `__main__`-defined
`PartitionedSource` or reduction callable is cloudpickled by value and digests differently across
`PYTHONHASHSEED` values, which would make the plan-byte determinism anchor red against a correct
implementation (§7.3).

The basename carries the `m49_` prefix because `tests/frozen/checkpoint` and `tests/extra/checkpoint`
collect in ONE pytest process and `checkpoint/m8/analyses.py` is additionally on `pythonpath`, so the
bare name `analyses` is bound repo-wide.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import graphed
from graphed import Array, Session, aggregate_plan
from graphed.core import DurablePlan, GraphStore, OpSpec, Partition
from graphed.core.execution import Plan
from graphed.numpy import NumpyBackend, NumpyForm

#: the 1-D payload every partition is a contiguous slice of
VECTOR = np.arange(1.0, 25.0)
FAMILY = "jes"
N_PARTITIONS = 6

#: whole-partition reads, appended by `VectorSource.read_partition`. A module-level list (not
#: instance state) so it survives the cloudpickle round trip `DurablePlan.process.resolve()` makes:
#: `VectorSource` is importable, so it travels by class reference and the method body binds THIS list.
READS: list[tuple[int, int]] = []


@dataclass
class VectorSource:
    """A `PartitionedSource` over `VECTOR` (`aggregate_plan`'s single-partitioned-source precondition)."""

    data: np.ndarray = field(default_factory=lambda: VECTOR)

    def __call__(self) -> np.ndarray:
        raise AssertionError("the whole-dataset loader must never run during a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        edges = np.linspace(0, len(self.data), steps_per_file + 1, dtype=int)
        return tuple(
            Partition("toy://vector", "", int(edges[i]), int(edges[i + 1])) for i in range(steps_per_file)
        )

    def read_partition(self, partition: Any, columns: Any, resources: Any) -> np.ndarray:
        READS.append((partition.entry_start, partition.entry_stop))
        return self.data[partition.entry_start : partition.entry_stop]


# ---- the reduction spec (module-level, picklable, deterministic) --------------------------------
def sums(values: list[Any]) -> np.ndarray:
    """One partition's composite: the per-universe sums, in marked-output order."""
    return np.array([np.asarray(v).sum() for v in values], dtype=np.float64)


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def zeros() -> np.ndarray:
    return np.zeros(1 + len(_TAGS), dtype=np.float64)


_TAGS = ("up", "down")


# ---- the varied program -------------------------------------------------------------------------
def varied_program(*, poison: bool) -> tuple[Session, VectorSource, list[Array], tuple[str, ...]]:
    """§2.1(a)'s loose primitive over a partitioned numpy source, lowered to sibling outputs.

    `poison=True` replaces the `jes_up` member with the m6 `numpy_mismatch` idiom — a boolean-mask
    getitem added back to its unmasked operand — which raises exactly on the partitions the mask
    shortens. `VECTOR`'s first partition is the only one holding a value under the threshold, so the
    poison is confined to one partition while every universe of every other partition evaluates.
    """
    s = Session(NumpyBackend())
    src = VectorSource()
    x = s.source("x", form=NumpyForm(VECTOR.dtype, shape=(None,)), data=src)
    base = x * 2.0
    up = (base[base > 5.0] + base) if poison else base * 1.5
    v = graphed.vary(base, FAMILY, up=up, down=base * 0.5)
    tags = tuple(graphed.labels(v))
    return s, src, [graphed.universe(v, label) for label in tags], tags


def build_plan(
    *, poison: bool = False, on_compiled: Callable[[Any], Any] | None = None
) -> tuple[Plan[Any], Any, tuple[str, ...]]:
    """The `Plan` `aggregate_plan` actually returns, plus the `CompiledGraph` the (alpha) seam exposes."""
    seen: list[Any] = []

    def hook(compiled: Any) -> Any:
        seen.append(compiled)
        return None if on_compiled is None else on_compiled(compiled)

    _s, _src, outputs, tags = varied_program(poison=poison)
    plan = aggregate_plan(
        *outputs,
        reduce=sums,
        combine=add,
        empty=zeros,
        steps_per_file=N_PARTITIONS,
        on_compiled=hook,
    )
    return plan, seen[0], tags


def durable(plan: Plan[Any], compiled: Any, partitions: Sequence[Partition] | None = None) -> DurablePlan:
    """§7.3's bound construction: BY VALUE over the plan `aggregate_plan` returned.

    `OpSpec.from_ref` would keep the closure's fields — `variation_labels` among them — out of the
    plan bytes entirely, so the §8.2(i) determinism anchor could not see them.
    """
    parts = tuple(t.partition for t in plan.tasks) if partitions is None else tuple(partitions)
    return DurablePlan(
        ir=bytes(compiled.ir),
        process=OpSpec.from_callable(plan.process),
        combine=OpSpec.from_callable(add),
        empty=OpSpec.from_callable(zeros),
        partitions=parts,
    )


# ---- §8.2(i) payloads: producer-SHAPED, keyed on the compiled reduced store's node ids ----------
def _entries(compiled: Any) -> list[tuple[tuple[int, int | None], tuple[tuple[str, ...], tuple[Any, ...]]]]:
    nodes = GraphStore.deserialize(bytes(compiled.ir)).nodes()
    out = []
    for n in nodes:
        nid = int(n["id"])
        for member_index in (None, 0, 1, 2):
            labels = () if member_index is None else tuple(sorted(f"{FAMILY}_{t}" for t in _TAGS))
            frame = ("m49_analyses.py", 10 * nid + (member_index or 0), "varied_program", "base * 2.0")
            out.append(((nid, member_index), (labels, frame)))
    return out


def sorted_payload(compiled: Any) -> tuple[Any, ...]:
    """§8.2(i)'s ordering rule: `(reduced_node_id, -1 if member_index is None else member_index)`."""
    return tuple(sorted(_entries(compiled), key=lambda e: (e[0][0], -1 if e[0][1] is None else e[0][1])))


def frozenset_payload(compiled: Any) -> frozenset[Any]:
    """The shape §8.2(i) BANS — the live control for the determinism anchor's instrument."""
    return frozenset(_entries(compiled))


PAYLOADS: dict[str, Callable[[Any], Any]] = {"sorted": sorted_payload, "frozenset": frozenset_payload}


def emit_plan_digest(kind: str) -> None:
    """Child-process entry point: the plan-bytes digest and every `task_id`, one per line.

    Run under a chosen `PYTHONHASHSEED`; `kind` selects the (beta) payload shape.
    """
    plan, compiled, _tags = build_plan(on_compiled=PAYLOADS[kind])
    dp = durable(plan, compiled)
    print(hashlib.sha256(dp.to_bytes()).hexdigest())
    for part in dp.partitions:
        print(dp.task_id(part))
