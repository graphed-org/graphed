"""``from_parquet(steps_per_file=, open_files=)`` is the default partitioning of a plan (#109)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import Session, aggregate_plan
from graphed.core import Partition
from graphed.core.execution import SequentialRunner

pytest.importorskip("pyarrow")
import graphed.awkward.io as gio
import graphed.parquet as gpq
from graphed.awkward import AwkwardBackend, from_parquet, gak

ROWS = (10, 5)


@pytest.fixture
def files(tmp_path: Path) -> list[str]:
    out = []
    for i, n in enumerate(ROWS):
        out.append(os.path.join(tmp_path, f"f{i}.parquet"))
        ak.to_parquet(ak.Array({"x": [float(100 * i + j) for j in range(n)]}), out[-1])
    return out


def _total(paths: list[str]) -> float:
    return float(sum(ak.sum(ak.from_parquet(p).x) for p in paths))


def _sum_plan(events: Any, **kwargs: Any) -> Any:
    return aggregate_plan(
        gak.sum(events.x),
        reduce=lambda values: float(values[0]),
        combine=lambda a, b: a + b,
        empty=lambda: 0.0,
        **kwargs,
    )


def _shape(partitions: list[Partition]) -> list[tuple[Any, ...]]:
    return [
        (os.path.basename(p.uri), p.blind_step, p.blind_n_steps, p.entry_start, p.entry_stop)
        for p in partitions
    ]


def test_the_readers_steps_split_a_plan_blind(
    files: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    num_rows = gpq.num_rows

    def counted(path: str) -> int:
        opened.append(path)
        return num_rows(path)

    monkeypatch.setattr(gpq, "num_rows", counted)
    plan = _sum_plan(from_parquet(Session(AwkwardBackend()), "ev", files, steps_per_file=3))
    assert _shape([t.partition for t in plan.tasks]) == [
        (f"f{i}.parquet", s, 3, 0, 0) for i in range(2) for s in range(3)
    ]
    assert opened == []
    assert SequentialRunner().run(plan).value == _total(files)


def test_open_files_plans_exact_entry_ranges(files: list[str]) -> None:
    events = from_parquet(
        Session(AwkwardBackend()), "ev", files, steps_per_file=2, open_files=True
    )
    plan = _sum_plan(events)
    assert _shape([t.partition for t in plan.tasks]) == [
        ("f0.parquet", None, None, 0, 5),
        ("f0.parquet", None, None, 5, 10),
        ("f1.parquet", None, None, 0, 2),
        ("f1.parquet", None, None, 2, 5),
    ]
    assert SequentialRunner().run(plan).value == _total(files)


def test_a_plans_own_steps_per_file_wins(files: list[str]) -> None:
    events = from_parquet(Session(AwkwardBackend()), "ev", files, steps_per_file=3)
    plan = _sum_plan(events, steps_per_file=2)
    assert [(p.blind_step, p.blind_n_steps) for p in (t.partition for t in plan.tasks)] == [
        (s, 2) for _ in files for s in range(2)
    ]
    written = gio.to_parquet(events.x, "unused", steps_per_file=1, compute=False)
    assert isinstance(written, graphed.core.execution.Plan)
    assert len(written.tasks) == len(files)


@pytest.mark.parametrize("open_files", [False, True])
def test_to_parquet_writes_the_readers_parts_in_order(
    files: list[str], tmp_path: Path, open_files: bool
) -> None:
    events = from_parquet(
        Session(AwkwardBackend()), "ev", files, steps_per_file=3, open_files=open_files
    )
    parts = gio.to_parquet(events.x * 2.0, os.path.join(tmp_path, "out"))
    assert isinstance(parts, list)
    assert [os.path.basename(p) for p in sorted(parts)] == [
        f"part-{i:05d}.parquet" for i in range(6)
    ]
    expected = [2.0 * x for p in files for x in ak.from_parquet(p).x.to_list()]
    assert [x for p in sorted(parts) for x in ak.from_parquet(p).data.to_list()] == expected


def test_reads_differing_only_in_partitioning_do_not_collide(files: list[str]) -> None:
    irs = []
    for steps, open_files in ((1, False), (3, False), (1, True)):
        events = from_parquet(
            Session(AwkwardBackend()), "ev", files, steps_per_file=steps, open_files=open_files
        )
        irs.append((events.session.serialized_ir(events.x), _sum_plan(events).process.ir))
    assert len({ir for ir, _ in irs}) == len({plan_ir for _, plan_ir in irs}) == 3
