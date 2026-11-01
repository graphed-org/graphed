"""What a graphed-built plan's repr says, without reading a file."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_parquet, parquet_write
from graphed.core.execution import SequentialRunner

pytest.importorskip("pyarrow")


def _reduce(values: list[Any]) -> int:
    return len(values[0])


class _Named:
    """A ``reduce`` naming its outputs for the summary."""

    def __call__(self, values: list[Any]) -> int:
        return len(values[0])

    def output_summaries(self) -> list[str]:
        return [f"h{i}" for i in range(8)]


@pytest.fixture
def events(tmp_path: Path) -> Any:
    path = tmp_path / "events.parquet"
    ak.to_parquet(ak.Array({"x": [1.0, 2.0], "y": [3.0, 4.0]}), path)
    return from_parquet("e", [str(path)], session=Session(AwkwardBackend()), steps_per_file=2)


def _plan(array: Any, **kwargs: Any) -> Any:
    return graphed.aggregate_plan(array, combine=int.__add__, empty=int, **kwargs)


def test_it_counts_the_columns_and_reads_nothing(events: Any, tmp_path: Path) -> None:
    plan = _plan(events.x + events.y, reduce=_reduce)
    (tmp_path / "events.parquet").unlink()
    assert repr(plan) == "Plan: 2 tasks over 1 file (events.parquet), reads 2 columns"


def test_it_names_outputs_a_reduce_describes(events: Any) -> None:
    text = repr(_plan(events.x, reduce=_Named()))
    assert "outputs: h0, h1, h2, h3, h4, h5, +2 more, reads 1 column" in text


def test_it_says_when_it_writes(events: Any, tmp_path: Path) -> None:
    write = parquet_write(
        events.x, str(tmp_path / "out"), name=lambda p: f"{p.blind_step}.parquet"
    )
    assert repr(_plan(events.x, reduce=_reduce, writes=[write])).endswith(", writes 1 dataset")


def test_every_column_is_said_so(events: Any) -> None:
    plan = _plan(events, reduce=_reduce)
    assert repr(plan).endswith("reads every column")


def test_a_collated_plan_names_its_plans(tmp_path: Path, events: Any) -> None:
    other = tmp_path / "other.parquet"
    ak.to_parquet(ak.Array({"x": [1.0]}), other)
    second = from_parquet("o", [str(other)], session=Session(AwkwardBackend()))
    plan = graphed.collate(
        {"DY": _plan(events.x, reduce=_reduce), "TT": _plan(second.x, reduce=_reduce)}
    )
    assert repr(plan) == "Plan: 3 tasks over 2 files, outputs: DY, TT"
    assert SequentialRunner().run(pickle.loads(pickle.dumps(plan))).value == {"DY": 2, "TT": 1}


def test_a_collated_plan_lists_the_outputs_its_plans_describe(tmp_path: Path, events: Any) -> None:
    other = tmp_path / "other.parquet"
    ak.to_parquet(ak.Array({"x": [1.0]}), other)
    second = from_parquet("o", [str(other)], session=Session(AwkwardBackend()))
    plan = graphed.collate(
        {"TT": _plan(second.x, reduce=_reduce), "DY": _plan(events.x, reduce=_Named())}
    )
    assert repr(plan) == "Plan: 3 tasks over 2 files, outputs: TT, h0, h1, h2, h3, h4, +3 more"


def test_a_saved_graphed_plan_runs_alike(events: Any, tmp_path: Path) -> None:
    path = tmp_path / "p.plan"
    plan = _plan(events.x, reduce=_reduce)
    plan.save(path)
    assert SequentialRunner().run(type(plan).load(path)).value == 2
