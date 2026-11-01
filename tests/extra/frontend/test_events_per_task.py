"""``events_per_task`` chunks a plan by entries, not by steps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import awkward as ak
import pytest

import graphed
import graphed.awkward as ga
import graphed.awkward.functions as gak
from graphed.aggregate import _event_partitions
from graphed.core import Partition, Plan, SequentialRunner


def _add(a: float, b: float) -> float:
    return a + b


def _zero() -> float:
    return 0.0


def _first(values: list[Any]) -> Any:
    return values[0]


def _events(tmp_path: Path, *, open_files: bool = False) -> Any:
    pytest.importorskip("pyarrow")
    paths = []
    for name, n in (("a", 25), ("b", 10)):
        paths.append(str(tmp_path / f"{name}.parquet"))
        ak.to_parquet(ak.Array({"x": [1.0] * n}), paths[-1])
    session = graphed.Session(ga.AwkwardBackend())
    return ga.from_parquet("events", paths, open_files=open_files, session=session)


def _plan(events: Any, **kw: Any) -> Plan[float]:
    total = gak.sum(events.x, axis=None)
    return graphed.aggregate_plan(total, reduce=_first, combine=_add, empty=_zero, **kw)


@pytest.mark.parametrize("open_files", [False, True], ids=["blind", "opened"])
def test_tasks_hold_the_asked_entries(tmp_path: Path, open_files: bool) -> None:
    plan = _plan(_events(tmp_path, open_files=open_files), events_per_task=10)
    parts = [t.partition for t in plan.tasks]
    ranges = [(Path(p.uri).stem, p.entry_start, p.entry_stop) for p in parts]
    assert ranges == [("a", 0, 10), ("a", 10, 20), ("a", 20, 25), ("b", 0, 10)]
    assert SequentialRunner().run(plan).value == 35.0


def test_a_budget_larger_than_a_file_is_one_task_per_file(tmp_path: Path) -> None:
    assert len(_plan(_events(tmp_path), events_per_task=1000).tasks) == 2


@pytest.mark.parametrize("other", [{"steps_per_file": 2}, {"partitions": ()}], ids=str)
def test_a_second_chunking_is_refused(tmp_path: Path, other: dict[str, Any]) -> None:
    with pytest.raises(TypeError, match="omit steps_per_file= and partitions="):
        _plan(_events(tmp_path), events_per_task=10, **other)


def test_a_non_positive_count_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="events_per_task must be >= 1, got 0"):
        _plan(_events(tmp_path), events_per_task=0)


class Uncounted:
    def partitions(self, steps_per_file: int) -> tuple[Partition, ...]:
        return (Partition.blind("f.root", "Events", 0, 1),)

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> object:
        raise AssertionError("planning reads nothing")


def test_a_source_that_cannot_count_is_refused() -> None:
    with pytest.raises(TypeError, match="Uncounted cannot count its entries"):
        _event_partitions(Uncounted(), 10)


class Counted(Uncounted):
    opened = 0

    def num_entries(self, partition: Partition) -> int:
        Counted.opened += 1
        return 7


def test_a_counted_source_is_asked_once_per_file() -> None:
    parts = _event_partitions(Counted(), 3)
    assert Counted.opened == 1
    assert [(p.entry_start, p.entry_stop) for p in parts] == [(0, 3), (3, 6), (6, 7)]


class Ranged(Uncounted):
    def partitions(self, steps_per_file: int) -> tuple[Partition, ...]:
        return (Partition("f.root", "Events", 0, 5),)


def test_a_source_with_ranges_needs_no_count() -> None:
    parts = _event_partitions(Ranged(), 3)
    assert [(p.entry_start, p.entry_stop) for p in parts] == [(0, 3), (3, 5)]
