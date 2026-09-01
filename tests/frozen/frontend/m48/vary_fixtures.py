"""Awkward-free fixtures for the m48 `vary` frontend suite.

ci.yml's REQUIRED free-threaded job collects `tests/frozen/frontend` WHOLE with only
`pytest hypothesis numpy` installed, so nothing reachable from this tree may import
`graphed.awkward` (or pyarrow/hist/pandas) — §10/m48 partition rule (3). Every fixture here is
numpy-idiom and every varied program is built on §2.1(a)'s loose primitive.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import graphed
from graphed import Array, Session
from graphed.core import Partition
from graphed.numpy import NumpyBackend, NumpyForm, from_array, from_record

#: the 1-D VECTOR payload every recording fixture reads
VECTOR = np.arange(1.0, 13.0)


def vector_source() -> tuple[Session, Array]:
    """A numpy-idiom session over a 1-D VECTOR partitioned source.

    Rank and kind are both load-bearing for §2.2's property dispositions: `NumpyArray.T` — the
    recording representative — raises `GraphedTypeError` on a >=2-D partitioned form and on any
    record form, so the node-count measurement that classifies it cannot run on `record_source`.
    """
    s = Session(NumpyBackend())
    return s, from_array(s, "x", VECTOR)


def record_source() -> tuple[Session, Array]:
    """A RECORD source carrying a literal `node_id` FIELD.

    §2.2 makes `node_id`/`session` raise on a `Varied`; the rule is a PROPERTY rule, so string
    getitem must still read a field of that name (a blanket `__getattr__` refusal would eat it).
    """
    s = Session(NumpyBackend())
    return s, from_record(s, "r", node_id=np.arange(4), pt=np.arange(1.0, 5.0))


def loose_varied(x: Array, name: str = "jes", *, tags: Sequence[str] = ("up", "down")) -> graphed.Varied:
    """§2.1(a): the loose primitive on a plain `Array` target, `x` itself as `"nominal"`."""
    factors = (1.5, 0.5, 2.5, 0.25)
    return graphed.vary(x, name, **{t: x * f for t, f in zip(tags, factors, strict=False)})


def sibling_outputs(v: graphed.Varied) -> list[Array]:
    """The §1.2 sibling lowering as `graphed` sees it: one marked output per label, in
    `graphed.labels` order (`compile_ir` refuses a `Varied`, §2.3d)."""
    return [graphed.universe(v, label) for label in graphed.labels(v)]


class ToyBackend:
    """The m5 backend shape (`tests/frozen/frontend/m5/test_aggregate_plan.py`), reused verbatim so
    the §7.2 seam anchor extends that call shape rather than a new one."""

    def op_form(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> str:
        return op

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        if op == "add":
            return [a + b for a, b in zip(inputs[0], inputs[1], strict=True)]  # type: ignore[call-overload]
        return inputs[0]

    def boundary_ops(self) -> frozenset[str]:
        return frozenset({"source"})

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        return used

    def external_payload(self, op: str, params: Mapping[str, object]) -> None:
        return None


@dataclass
class ListSource:
    """A `PartitionedSource` over an in-memory list, counting partition reads (the m5 shape)."""

    data: list[int]
    reads: list[tuple[int, int]] = field(default_factory=list)

    def __call__(self) -> list[int]:
        raise AssertionError("the whole-dataset loader must never run during a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("toy://list", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition: Any, columns: Any, resources: Any) -> list[int]:
        part = partition.resolve(len(self.data))
        self.reads.append((part.entry_start, part.entry_stop))
        return list(self.data[part.entry_start : part.entry_stop])


@dataclass
class ArraySource:
    """A numpy-idiom `PartitionedSource` over `VECTOR`, so a VARIED program can reach a real
    `Plan`/`ExecResult` through the §1.2 sibling lowering (one marked output per label)."""

    data: np.ndarray
    reads: list[tuple[int, int]] = field(default_factory=list)

    def __call__(self) -> np.ndarray:
        raise AssertionError("the whole-dataset loader must never run during a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("toy://vector", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition: Any, columns: Any, resources: Any) -> np.ndarray:
        part = partition.resolve(len(self.data))
        self.reads.append((part.entry_start, part.entry_stop))
        return self.data[part.entry_start : part.entry_stop]


def partitioned_vector_source() -> tuple[Session, Array, ArraySource]:
    """A numpy-idiom session whose single source is partitioned (`aggregate_plan`'s precondition)."""
    s = Session(NumpyBackend())
    src = ArraySource(VECTOR)
    x = s.source("x", form=NumpyForm(VECTOR.dtype, shape=(None,)), data=src)
    return s, x, src


def sum_per_output(values: list[Any]) -> list[float]:
    return [float(np.asarray(v).sum()) for v in values]


def add_per_output(a: list[float], b: list[float]) -> list[float]:
    return [p + q for p, q in zip(a, b, strict=True)]


def sum_each(values: list[Any]) -> list[int]:
    return [sum(v) for v in values]


def add_pairs(a: list[int], b: list[int]) -> list[int]:
    return [x + y for x, y in zip(a, b, strict=True)]


def m5_two_outputs(s: Session) -> tuple[Array, Array, ListSource]:
    """m5's own two-output program: `out1 = 2x`, `out2 = 3x` over one partitioned source."""
    src = ListSource(list(range(1, 13)))
    x = s.source("x", form="f", data=src)
    shared = x + x
    return shared, shared + x, src
