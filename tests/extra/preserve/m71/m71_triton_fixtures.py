"""A served classifier over an in-memory Triton transport: it answers booleans, which the Triton
plugin returns as float64 event scores."""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np

from graphed.core import Partition

ENDPOINT = "http://svc-host:8000"


class _Result:
    def __init__(self, y: np.ndarray) -> None:
        self.y = y

    def as_numpy(self, name: str) -> np.ndarray:
        return self.y


class Client:
    def infer(self, model: str, inputs: list[Any], outputs: Any = None) -> _Result:
        return _Result(inputs[0].data[:, 0] > 1)

    def close(self) -> None:
        pass


def transport(params: Any) -> Client:
    return Client()


class InferInput:
    def __init__(self, name: str, shape: list[int], dt: str) -> None:
        self.data: Any = None

    def set_data_from_numpy(self, a: Any) -> None:
        self.data = np.asarray(a)


class InferRequestedOutput:
    def __init__(self, name: str) -> None:
        self.name = name


class Chunks:
    def __init__(self) -> None:
        self.data = ak.Array({"x": np.array([0.5, 2.0, 5.0, 8.0])})

    def partitions(self, steps: int) -> tuple[Partition, ...]:
        return tuple(Partition.blind("mem://m71", "", s, steps) for s in range(steps))

    def read_partition(self, p: Partition, columns: Any, resources: Any) -> Any:
        r = p.resolve(len(self.data))
        return self.data[r.entry_start : r.entry_stop]


def total(values: list[Any]) -> float:
    return float(np.sum(values[0]))


def add(a: float, b: float) -> float:
    return a + b


def zero() -> float:
    return 0.0
