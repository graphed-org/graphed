"""Awkward-free fixtures for the m69b frontend suite (the free-threaded job collects
`tests/frozen/frontend` with only numpy installed). Everything a plan ships is module level, so a
pickled plan names it by reference."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from graphed import Array, Session, aggregate_plan
from graphed.core import Partition
from graphed.numpy import NumpyBackend, NumpyForm

COLUMNS = {
    "x": np.array([1.0, 2.0, 3.0, 4.0]),
    "w": np.array([0.5, 1.0, 1.5, 2.0]),
    "z": np.array([7.0, 8.0, 9.0, 10.0]),
}
FORM = NumpyForm(np.dtype(object), kind="record", fields=tuple((c, "<f8") for c in COLUMNS))


@dataclass(frozen=True)
class Events:
    """A `PartitionedSource` over `COLUMNS` that returns only the columns it is asked for."""

    def __call__(self) -> Any:
        raise AssertionError("the whole-dataset loader must never run inside a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("mem://events", "t", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> dict[str, np.ndarray]:
        part = partition.resolve(len(COLUMNS["x"]))
        names = COLUMNS if columns is None else columns
        return {c: COLUMNS[c][part.entry_start : part.entry_stop] for c in names}


def record() -> tuple[Session, Array]:
    """The fixture session. Before any output it records, unmarked, a node over `z` (no output
    reads it) and a node that raises on the data (no `x` exceeds 100)."""
    session = Session(NumpyBackend())
    events = session.source("events", form=FORM, data=Events())
    _unread = events.z * 3.0
    _raising = events.x[events.x > 100.0][0]
    return session, events


def merged_pair(events: Array) -> tuple[Array, Array]:
    """Two marked outputs the optimizer merges into one."""
    return events.w, events.w * 1.0


def failing_op(events: Array) -> tuple[Array, int]:
    """An op that raises on the data, and the line recording it."""
    return events.w[events.w > 100.0][0], sys._getframe().f_lineno


def sums(values: list[Any]) -> np.ndarray:
    """One total per value `reduce` receives, so the result's length is the reduce's arity."""
    return np.array([float(np.sum(v)) for v in values])


def arity(values: list[Any]) -> list[int]:
    return [len(values)]


def add(a: Any, b: Any) -> Any:
    return a + b


def zero() -> float:
    return 0.0


def no_arities() -> list[int]:
    return []


def by_step(p: Partition) -> str:
    return f"part{p.blind_step}.json"


def json_codec(value: object, path: str, kv: Mapping[str, str] | None) -> None:
    with open(path, "w") as handle:
        json.dump({"kv": kv, "values": np.asarray(value).tolist()}, handle, sort_keys=True)


def emit_opt0_digest() -> None:
    """One line: the sha256 of an opt-0 plan's IR and of the pickled plan, then this
    interpreter's `str` hash salt."""
    _session, events = record()
    plan = aggregate_plan(
        *merged_pair(events), reduce=sums, combine=add, empty=zero, steps_per_file=2, opt_level=0
    )
    ir = hashlib.sha256(bytes(plan.process.ir)).hexdigest()
    print(ir, hashlib.sha256(pickle.dumps(plan)).hexdigest(), hash("graphed"))
