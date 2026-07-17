"""M8 — checkpoint/resume on a real graphed HEP analysis (recorded + executed per chunk).

A deeper integration than the synthetic-histogram tests: each chunk records the ADL-q1 MET-pt
histogram through the graphed frontend on the awkward backend, and a kill-and-resume across chunks
must still match the single-pass histogram bit-for-bit while skipping the completed work.
"""

from __future__ import annotations

import analyses
import numpy as np
import pytest

from graphed.checkpoint import Store, run_resumable
from graphed.checkpoint.runner import _SimulatedInterrupt


def test_real_analysis_chunked_resume_matches_single_pass(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ref = analyses.met_reference()
    assert int(ref.sum()) > 0, "the realistic ladder must actually fill the histogram"

    store, plan = Store(tmp_path), analyses.build_met_plan(5)
    with pytest.raises(_SimulatedInterrupt):
        run_resumable(plan, store, _kill_after=3)
    res = run_resumable(plan, store)
    assert res.report.skipped == 3 and res.report.executed == 2
    assert np.array_equal(res.value, ref), "resumed real-analysis histogram must match single pass"


def test_real_analysis_is_partition_count_invariant(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ref = analyses.met_reference()
    for i, n_chunks in enumerate((1, 6)):
        res = run_resumable(analyses.build_met_plan(n_chunks), Store(tmp_path / str(i)))
        assert np.array_equal(res.value, ref)
