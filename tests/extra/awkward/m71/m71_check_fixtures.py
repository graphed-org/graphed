"""Importable (spawn-safe) pieces for the runtime declared-type check suite."""

from __future__ import annotations

import sys
from typing import Any

import awkward as ak
import numpy as np

from graphed.core import Partition
from graphed.core.execution import LocalResources

X = np.array([0.5, 1.5, 2.5, 3.5])


class Chunks:
    """An in-memory partitioned source of the events ``{x: X}``."""

    def __init__(self) -> None:
        self.data = ak.Array({"x": X})

    def partitions(self, steps: int) -> tuple[Partition, ...]:
        return tuple(Partition.blind("mem://m71", "", s, steps) for s in range(steps))

    def read_partition(self, p: Partition, columns: Any, resources: Any) -> Any:
        r = p.resolve(len(self.data))
        return self.data[r.entry_start : r.entry_stop]


def score(x: Any) -> Any:
    """A classifier score: float64, though a caller may declare it a mask."""
    return 1.0 / (1.0 + np.exp(-x))


def cut(x: Any) -> Any:
    return x > 1


def total(values: list[Any]) -> float:
    return float(np.sum(values[0]))


def add(a: float, b: float) -> float:
    return a + b


def zero() -> float:
    return 0.0


def run_task(process: Any, partition: Partition) -> Any:
    """One plan task in a worker process, as an executor runs it."""
    return process(partition, LocalResources())


def here() -> int:
    """The caller's line."""
    return sys._getframe(1).f_lineno
