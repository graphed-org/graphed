"""A capturing plan keeps a projected read's input as read, and its replay agrees with the run (#54)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

pytest.importorskip("pyarrow")

from awkward._nplikes.placeholder import PlaceholderArray

import graphed.debug as gd
from graphed import Session
from graphed.aggregate import _PartitionReduce, aggregate_plan
from graphed.awkward import AwkwardBackend, from_parquet, gak
from graphed.core import Partition, Plan, WorkerResources
from graphed.core.execution import SequentialRunner
from graphed.write import PartitionedSource

N = 100
FORM = ak.forms.from_dict(
    {
        "class": "RecordArray",
        "fields": ["x", "jets"],
        "contents": [
            {"class": "NumpyArray", "primitive": "float64", "form_key": "x"},
            {
                "class": "ListOffsetArray",
                "offsets": "i64",
                "form_key": "j",
                "content": {"class": "NumpyArray", "primitive": "float64", "form_key": "jc"},
            },
        ],
    }
)
#: a behavior holding a lambda, as vector's does, which plain pickle refuses
BEHAVIOR = {("__typestr__", "Event"): "Event", "unpicklable": lambda: None}


def _chunk(projected: bool) -> ak.Array:
    jets = (
        {"j-offsets": PlaceholderArray(np, (N + 1,), np.int64), "jc-data": PlaceholderArray(np, (N,), np.float64)}
        if projected
        else {"j-offsets": np.arange(N + 1, dtype=np.int64), "jc-data": np.ones(N)}
    )
    return ak.from_buffers(FORM, N, {"x-data": np.arange(float(N)), **jets}, behavior=BEHAVIOR)


@dataclasses.dataclass(frozen=True)
class _Reader:
    """The recorded source, handing every task the chunk a read restricted to ``x`` would."""

    source: PartitionedSource
    projected: bool

    def partitions(self, steps_per_file: int) -> tuple[Partition, ...]:
        return self.source.partitions(steps_per_file)

    def read_partition(self, partition: Partition, columns: Any, resources: WorkerResources) -> ak.Array:
        return _chunk(self.projected)


def _first(v: list[Any]) -> float:
    return float(v[0])


def _add(a: float, b: float) -> float:
    return a + b


def _zero() -> float:
    return 0.0


@pytest.mark.parametrize("projected", [True, False], ids=["placeholder-buffers", "lambda-behavior"])
def test_a_captured_projected_read_replays_as_the_run(tmp_path: Path, projected: bool) -> None:
    s = Session(AwkwardBackend())
    ak.to_parquet(_chunk(projected=False), tmp_path / "d.parquet")
    ev = from_parquet(s, "events", str(tmp_path / "d.parquet"))
    y = gak.sum(ev.x, axis=None)
    built = aggregate_plan(y, reduce=_first, combine=_add, empty=_zero, store=tmp_path / "capture")
    assert isinstance(built.process, _PartitionReduce)
    plan: Plan[float] = dataclasses.replace(
        built, process=dataclasses.replace(built.process, reader=_Reader(built.process.reader, projected))
    )

    total = SequentialRunner().run(plan).value

    r = gd.replay(plan, 0, y)
    assert r.input_source == "store"
    assert r.value == total == float(np.arange(float(N)).sum())
    d = r.diff()
    assert (d.reference, d.equal) == ("recorded", True)
    stored = r._chunk
    assert stored.type == _chunk(projected).type
    assert stored.behavior.keys() == BEHAVIOR.keys()
    jets = stored.layout.content("jets")
    assert isinstance(jets.offsets.data, PlaceholderArray) is projected
    assert isinstance(jets.content.data, PlaceholderArray) is projected
