"""m62 unit B — another interpreter resumes a run given only the plan file and the store URL."""

from __future__ import annotations

import hashlib

import analyses
import m62_url_helpers as h
import pytest

from graphed.checkpoint import FsspecStore, run_resumable
from graphed.checkpoint.runner import _SimulatedInterrupt


def test_other_process_resumes_from_the_url_alone(shared_url, tmp_path) -> None:
    plan = h.hist_plan(6)
    with pytest.raises(_SimulatedInterrupt):
        run_resumable(plan, FsspecStore(shared_url), _kill_after=4)
    parent = FsspecStore(shared_url)
    assert len(parent.completed()) == 4

    executed, skipped, value = h.resume_in_child(plan, shared_url, tmp_path)
    assert (executed, skipped) == (2, 4)
    assert value == analyses.reference().tolist()

    done = parent.completed()
    assert set(done) == {plan.task_id(p) for p in plan.partitions}
    for entry in done.values():
        blob = parent.get(entry.blob)
        assert blob is not None
        assert hashlib.sha256(blob).hexdigest() == entry.blob


def test_other_process_finds_nothing_left_to_do(shared_url, tmp_path) -> None:
    plan = h.hist_plan(6)
    assert run_resumable(plan, FsspecStore(shared_url)).report.executed == 6
    executed, skipped, value = h.resume_in_child(plan, shared_url, tmp_path)
    assert (executed, skipped) == (0, 6)
    assert value == analyses.reference().tolist()
