"""What a task keys: a behavior built by iterating a set, as coffea's registration is."""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import Any

import numpy as np

from graphed.awkward import AwkwardBackend
from graphed.core import (
    DurablePlan,
    DurablePlanV2,
    OpSpec,
    Partition,
    Plan,
    StageSpec,
    Task,
)


def overload(a: Any, b: Any) -> Any:
    return a


BEHAVIOR: dict[Any, Any] = {(np.add, n, n): overload for n in {f"M{i}" for i in range(16)}}
BEHAVIOR[np.add, "M0", numbers.Real] = overload


def backend() -> AwkwardBackend:
    return AwkwardBackend(behavior=BEHAVIOR)


def fold(values: list[Any]) -> float:
    return sum(float(v) for v in values)


def add(a: float, b: float) -> float:
    return a + b


def zero() -> float:
    return 0.0


def start(partition: Partition, resources: Any) -> float:
    return float(partition.entry_start)


def stop(partition: Partition, resources: Any) -> float:
    return float(partition.entry_stop)


def stage_start(task: Task, inputs: Sequence[bytes], resources: Any) -> bytes:
    return str(task.partition.entry_start).encode()


def parts(*starts: int) -> tuple[Partition, ...]:
    return tuple(Partition("f.root", "t", s, s + 10) for s in starts)


def runtime(process: Any, partitions: Sequence[Partition]) -> Plan[float]:
    tasks = tuple(Task(i, p) for i, p in enumerate(partitions))
    return Plan(process=process, combine=add, empty=zero, tasks=tasks)


def durable(ir: bytes, partitions: Sequence[Partition]) -> DurablePlan:
    return DurablePlan(
        ir=ir,
        process=OpSpec.from_callable(start),
        combine=OpSpec.from_callable(add),
        empty=OpSpec.from_callable(zero),
        partitions=tuple(partitions),
    )


def staged(ir: bytes, partitions: Sequence[Partition]) -> DurablePlanV2:
    tasks = tuple(Task(i, p) for i, p in enumerate(partitions))
    stage = StageSpec(kind="map_write", process=OpSpec.from_callable(stage_start), tasks=tasks)
    return DurablePlanV2(ir=ir, stages=(stage,))
