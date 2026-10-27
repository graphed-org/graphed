"""numpy ``from_parquet(steps_per_file=, open_files=)`` split ``to_parquet`` by default (#109)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from graphed import Session
from graphed.core.execution import Plan

pytest.importorskip("pyarrow")
import pyarrow as pa
import pyarrow.parquet as pq

import graphed.numpy.io as gio
from graphed.numpy import NumpyBackend

ROWS = (10, 5)


@pytest.fixture
def files(tmp_path: Path) -> list[str]:
    out = []
    for i, n in enumerate(ROWS):
        out.append(os.path.join(tmp_path, f"f{i}.parquet"))
        pq.write_table(pa.table({"x": np.arange(n, dtype=float) + 100 * i}), out[-1])
    return out


def _shape(plan: Any) -> list[tuple[Any, ...]]:
    assert isinstance(plan, Plan)
    return [
        (os.path.basename(p.uri), p.blind_step, p.blind_n_steps, p.entry_start, p.entry_stop)
        for p in (t.partition for t in plan.tasks)
    ]


def test_the_readers_steps_split_to_parquet_blind(files: list[str]) -> None:
    g = gio.from_parquet(Session(NumpyBackend()), "ev", files, steps_per_file=3)
    assert _shape(gio.to_parquet(g["x"], "unused", compute=False)) == [
        (f"f{i}.parquet", s, 3, 0, 0) for i in range(2) for s in range(3)
    ]


def test_open_files_splits_to_parquet_exactly(files: list[str], tmp_path: Path) -> None:
    g = gio.from_parquet(Session(NumpyBackend()), "ev", files, steps_per_file=2, open_files=True)
    assert _shape(gio.to_parquet(g["x"], "unused", compute=False)) == [
        ("f0.parquet", None, None, 0, 5),
        ("f0.parquet", None, None, 5, 10),
        ("f1.parquet", None, None, 0, 2),
        ("f1.parquet", None, None, 2, 5),
    ]
    parts = gio.to_parquet(g["x"] * 2.0, os.path.join(tmp_path, "out"))
    assert isinstance(parts, list)
    np.testing.assert_array_equal(
        np.concatenate([pq.read_table(p)["data"].to_numpy() for p in sorted(parts)]),
        2.0 * np.concatenate([pq.read_table(f)["x"].to_numpy() for f in files]),
    )


def test_to_parquets_own_steps_per_file_wins(files: list[str]) -> None:
    g = gio.from_parquet(Session(NumpyBackend()), "ev", files, steps_per_file=3)
    assert _shape(gio.to_parquet(g["x"], "unused", steps_per_file=2, compute=False)) == [
        (f"f{i}.parquet", s, 2, 0, 0) for i in range(2) for s in range(2)
    ]


def test_reads_differing_only_in_partitioning_do_not_collide(files: list[str]) -> None:
    irs = set()
    for steps, open_files in ((1, False), (3, False), (1, True)):
        g = gio.from_parquet(
            Session(NumpyBackend()), "ev", files, steps_per_file=steps, open_files=open_files
        )
        irs.add(g.session.serialized_ir(g["x"]))
    assert len(irs) == 3
