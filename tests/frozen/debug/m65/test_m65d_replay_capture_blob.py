"""M65 D frozen suite, fixup 2: a captured blob whose bytes no longer hash to its name (plan-D D-2, D-6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow")

import awkward as ak
import numpy as np

import graphed.checkpoint as gcp
import graphed.debug as gd
from graphed import Array, Session
from graphed.aggregate import aggregate_plan
from graphed.awkward import AwkwardBackend, from_parquet, gak
from graphed.core import Plan, SequentialRunner

N = 1000
ROWS = 250


def _first(v: list[Any]) -> float:
    return float(v[0])


def _add(a: float, b: float) -> float:
    return a + b


def _zero() -> float:
    return 0.0


def _captured(tmp_path: Path) -> tuple[Plan[Any], Array, gcp.Store]:
    path = tmp_path / "d.parquet"
    ak.to_parquet(ak.Array({"x": np.arange(float(N))}), path)
    ev = from_parquet(Session(AwkwardBackend()), "events", str(path))
    y = gak.sum(ev.x * 2.0, axis=None)
    root = tmp_path / "cap"
    plan = aggregate_plan(y, reduce=_first, combine=_add, empty=_zero, steps_per_file=4, store=root)
    SequentialRunner().run(plan)
    return plan, y, gcp.Store(str(root))


def _corrupt(store: gcp.Store, plan: Plan[Any], key: int, stage: str) -> str:
    part = str(next(t for t in plan.tasks if t.key == key).partition)
    (entry,) = [e for e in store.completed().values() if e.stage == stage and e.partition == part]
    blob = store.objects / entry.blob
    blob.write_bytes(blob.read_bytes() + b"corrupt")
    assert entry.blob in {e.blob for e in store.completed().values()}
    return str(entry.blob)


def test_a_corrupt_input_blob_raises_file_not_found(tmp_path: Path) -> None:
    plan, y, store = _captured(tmp_path)
    digest = _corrupt(store, plan, 1, "replay-input")
    r = gd.replay(plan, 1, y)
    assert r.input_source == "store"
    with pytest.raises(FileNotFoundError, match=digest):
        _ = r.value
    assert gd.replay(plan, 2, y).value == 2.0 * sum(range(2 * ROWS, 3 * ROWS))


def test_a_corrupt_output_blob_raises_file_not_found_from_diff(tmp_path: Path) -> None:
    plan, y, store = _captured(tmp_path)
    digest = _corrupt(store, plan, 1, "replay-output")
    r = gd.replay(plan, 1, y)
    assert r.input_source == "store"
    assert r.value == 2.0 * sum(range(ROWS, 2 * ROWS))
    with pytest.raises(FileNotFoundError, match=digest):
        r.diff()
