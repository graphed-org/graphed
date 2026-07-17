"""``aggregate_plan``: the multi-output, one-pass-over-a-shared-sub-graph engine. Several outputs
sharing a sub-expression compile into ONE IR and read+evaluate the partition ONCE (not once per
output); the caller supplies the per-output reduce/combine/empty. A NEW frozen file (test-authoring
deliverable). The bug this guards: building a plan per output recomputes a shared sub-graph N times.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pytest

from graphed import Array, Session, aggregate_plan, resolve_backend
from graphed.core import Partition
from graphed.core.execution import SequentialRunner


class _Backend:
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
    """A PartitionedSource over an in-memory list, counting partition reads (the efficiency witness)."""

    data: list
    reads: list = field(default_factory=list)

    def __call__(self) -> list:
        raise AssertionError("the whole-dataset loader must never run during a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("toy://list", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition, columns, resources) -> list:  # type: ignore[no-untyped-def]
        part = partition.resolve(len(self.data))
        self.reads.append((part.entry_start, part.entry_stop))
        return list(self.data[part.entry_start : part.entry_stop])


def _sum_each(values: list) -> list[int]:
    return [sum(v) for v in values]


def _add_pairs(a: list[int], b: list[int]) -> list[int]:
    return [x + y for x, y in zip(a, b, strict=True)]


def _two_outputs(s: Session) -> tuple[Array, Array, Array, ListSource]:
    src = ListSource(list(range(1, 13)))  # 1..12
    x = s.source("x", form="f", data=src)
    shared = x + x  # the shared sub-expression (2x), feeding both outputs
    return shared, shared + x, x, src  # out1 = 2x, out2 = 3x


def test_shared_subgraph_read_once_and_reduced_correctly() -> None:
    s = Session(_Backend())
    out1, out2, _x, src = _two_outputs(s)
    plan = aggregate_plan(
        out1, out2, reduce=_sum_each, combine=_add_pairs, empty=lambda: [0, 0], steps_per_file=4
    )
    value = SequentialRunner().run(plan).value
    assert value == [sum(2 * v for v in range(1, 13)), sum(3 * v for v in range(1, 13))]  # [156, 234]
    assert len(src.reads) == 4  # witness: 4 partitions read ONCE each, not once per output (would be 8)


def test_single_output() -> None:
    s = Session(_Backend())
    out1, _out2, _x, _src = _two_outputs(s)
    plan = aggregate_plan(
        out1, reduce=lambda vs: sum(vs[0]), combine=lambda a, b: a + b, empty=lambda: 0, steps_per_file=3
    )
    assert SequentialRunner().run(plan).value == sum(2 * v for v in range(1, 13))  # 156


def test_validation() -> None:
    s = Session(_Backend())
    with pytest.raises(ValueError, match="at least one output"):
        aggregate_plan(reduce=_sum_each, combine=_add_pairs, empty=lambda: [])
    # two distinct partitioned sources in one graph -> cannot aggregate one plan
    a = s.source("a", form="f", data=ListSource([1, 2]))
    b = s.source("b", form="f", data=ListSource([3, 4]))
    with pytest.raises(TypeError, match="partitioned source"):
        aggregate_plan(a + b, reduce=_sum_each, combine=_add_pairs, empty=lambda: [0])


def test_resolve_backend_forms() -> None:
    assert resolve_backend(lambda: "factory-made") == "factory-made"  # zero-arg factory
    assert isinstance(resolve_backend("collections:OrderedDict"), OrderedDict)  # importable callable
    assert resolve_backend("math:pi") == math.pi  # importable non-callable attr returned as-is
