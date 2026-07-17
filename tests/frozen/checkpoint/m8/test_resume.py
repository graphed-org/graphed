"""M8 acceptance — kill a run halfway, resume, and get the uninterrupted result with less work.

Plan M8 contract: "Kill a run halfway; re-run; final histogram equals an uninterrupted run AND the
second run does measurably less work (skipped tasks logged)."
"""

from __future__ import annotations

import analyses
import numpy as np
import pytest

from graphed_checkpoint import Store, run_resumable
from graphed_checkpoint.runner import _SimulatedInterrupt


def _plan(n_chunks: int = 6):  # type: ignore[no-untyped-def]
    return analyses.build_plan("analyses:histogram_chunk", n_chunks)


def test_uninterrupted_run_matches_single_pass(tmp_path) -> None:  # type: ignore[no-untyped-def]
    res = run_resumable(_plan(), Store(tmp_path))
    assert np.array_equal(res.value, analyses.reference())
    assert res.report.executed == 6 and res.report.skipped == 0


def test_kill_then_resume_equals_uninterrupted_and_does_less_work(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = Store(tmp_path)
    plan = _plan(6)
    # crash after 4 partitions have been committed to the store
    with pytest.raises(_SimulatedInterrupt):
        run_resumable(plan, store, _kill_after=4)
    assert len(store.completed()) == 4  # only committed work survived the "kill"

    # resume on the same store
    res = run_resumable(plan, store)
    assert res.report.skipped == 4, "completed tasks must be reused, not recomputed"
    assert res.report.executed == 2, "only the unfinished partitions run"
    assert res.report.did_less_work
    assert np.array_equal(res.value, analyses.reference()), "resumed result must match single pass"


def test_resume_after_completion_redoes_nothing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store, plan = Store(tmp_path), _plan(6)
    run_resumable(plan, store)
    again = run_resumable(plan, store)  # everything already done
    assert again.report.executed == 0 and again.report.skipped == 6
    assert np.array_equal(again.value, analyses.reference())


def test_result_is_invariant_to_partition_count(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ref = analyses.reference()
    for i, n_chunks in enumerate((1, 4, 13)):
        res = run_resumable(
            analyses.build_plan("analyses:histogram_chunk", n_chunks), Store(tmp_path / str(i))
        )
        assert np.array_equal(res.value, ref), f"{n_chunks} chunks changed the histogram"


def test_no_double_count_when_kill_lands_between_partitions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # kill at every possible boundary; each resume must still equal the single pass exactly once
    ref = analyses.reference()
    for k in range(1, 6):
        store, plan = Store(tmp_path / f"k{k}"), _plan(6)
        with pytest.raises(_SimulatedInterrupt):
            run_resumable(plan, store, _kill_after=k)
        res = run_resumable(plan, store)
        assert np.array_equal(res.value, ref)
        assert res.report.skipped == k
