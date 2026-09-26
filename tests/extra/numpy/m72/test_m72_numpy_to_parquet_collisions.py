"""numpy `to_parquet` plans name parts `part-<index>` from 0, so two collated into one destination collide."""

from __future__ import annotations

from pathlib import Path

import pytest

import graphed
from graphed import Session
from graphed.core.execution import Plan, SequentialRunner
from graphed.numpy import NumpyBackend
from graphed.numpy.io import from_parquet, to_parquet

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _write_plan(src: Path, rows: list[float], destination: Path) -> Plan[list[str]]:
    pq.write_table(pa.table({"x": rows}), src)
    ev = from_parquet(Session(NumpyBackend()), "events", str(src))
    plan = to_parquet(ev.x * 2.0, str(destination), compute=False)
    assert isinstance(plan, Plan)
    return plan


def test_collated_numpy_to_parquet_plans_into_one_destination_are_refused(tmp_path: Path) -> None:
    out = tmp_path / "out"
    a = _write_plan(tmp_path / "a.parquet", [1.0, 2.0], out)
    b = _write_plan(tmp_path / "b.parquet", [10.0, 20.0, 30.0], out)
    with pytest.raises(ValueError, match=r"write the same part .*part-00000\.parquet"):
        graphed.collate({"a": a, "b": b})
    assert not out.exists()


def test_collated_numpy_to_parquet_plans_into_distinct_destinations_write_each(tmp_path: Path) -> None:
    a = _write_plan(tmp_path / "a.parquet", [1.0, 2.0], tmp_path / "out-a")
    b = _write_plan(tmp_path / "b.parquet", [10.0, 20.0, 30.0], tmp_path / "out-b")
    value = SequentialRunner().run(graphed.collate({"a": a, "b": b})).value
    assert value == {
        "a": [str(tmp_path / "out-a" / "part-00000.parquet")],
        "b": [str(tmp_path / "out-b" / "part-00000.parquet")],
    }
    assert pq.read_table(value["a"][0]).column("data").to_pylist() == [2.0, 4.0]
    assert pq.read_table(value["b"][0]).column("data").to_pylist() == [20.0, 40.0, 60.0]
