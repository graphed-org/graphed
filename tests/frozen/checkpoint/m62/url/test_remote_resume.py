"""m62 unit B — kill a run on a URL store, resume with a new instance, match the uninterrupted run."""

from __future__ import annotations

import sys
from pathlib import Path

import analyses
import m62_url_helpers as h
import numpy as np
import pytest

from graphed.checkpoint import FsspecStore, Store, run_resumable, run_shuffle_resumable
from graphed.checkpoint.runner import _SimulatedInterrupt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "m39"))
import shuffle_analyses as sa


def test_kill_then_resume_equals_uninterrupted(store_url) -> None:
    plan = h.hist_plan(6)
    with pytest.raises(_SimulatedInterrupt):
        run_resumable(plan, FsspecStore(store_url), _kill_after=4)
    assert len(FsspecStore(store_url).completed()) == 4

    res = run_resumable(plan, FsspecStore(store_url))
    assert res.report.skipped == 4
    assert res.report.executed == 2
    assert np.array_equal(res.value, analyses.reference())
    assert set(FsspecStore(store_url).completed()) == {plan.task_id(p) for p in plan.partitions}


def test_shuffle_kill_then_resume_equals_uninterrupted(store_url, tmp_path) -> None:
    plan = sa.build_shuffle_plan_v2(3, 2)
    reference = Store(tmp_path / "reference")
    ref = run_shuffle_resumable(plan, reference)
    with pytest.raises(_SimulatedInterrupt):
        run_shuffle_resumable(plan, FsspecStore(store_url), _kill_after=3)
    assert len(FsspecStore(store_url).completed()) == 3

    res = run_shuffle_resumable(plan, FsspecStore(store_url))
    assert res.report.skipped == 3
    assert res.report.executed == sa.n_blocks(3, 2) - 3
    assert res.value == ref.value
    assert set(FsspecStore(store_url).completed()) == set(reference.completed())
