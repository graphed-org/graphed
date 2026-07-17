"""``_kill_after`` must fail loudly when the simulated crash can never fire.

Regression guard: a checkpointing *demo* asked ``run_resumable`` to "crash after 3 committed
partitions" while the plan it was given had only one partition. The kill point was unreachable, so
the run completed uninterrupted and the demonstration silently became a no-op (no crash, no resume).
A simulated crash that can never happen is a misconfiguration; the runner now raises ``ValueError``
rather than finishing quietly.

The three cases below are mutually discriminating on the SAME single-partition plan: the kill point
is the only thing that changes.
"""

from __future__ import annotations

from pathlib import Path

import analyses
import pytest
from graphed_core import DurablePlan

from graphed_checkpoint import Store, run_resumable
from graphed_checkpoint.runner import _SimulatedInterrupt


def _one_partition_plan() -> DurablePlan:
    plan = analyses.build_plan("analyses:histogram_chunk", 1)
    assert len(plan.partitions) == 1, "the regression needs a plan smaller than the kill point"
    return plan


def test_unreachable_kill_point_raises_valueerror(tmp_path: Path) -> None:
    # the regression: kill-after-3 on a 1-partition plan can never fire -> loud, not silent.
    with pytest.raises(ValueError, match=r"_kill_after=3 never fired.*committed only 1"):
        run_resumable(_one_partition_plan(), Store(tmp_path), _kill_after=3)


def test_reachable_kill_point_still_crashes(tmp_path: Path) -> None:
    # discriminator: same 1-partition plan, but a kill point it CAN reach -> the real interrupt,
    # proving the new guard fires only when the crash is genuinely unreachable.
    store = Store(tmp_path)
    with pytest.raises(_SimulatedInterrupt):
        run_resumable(_one_partition_plan(), store, _kill_after=1)
    assert len(store.completed()) == 1, "the committed partial must survive the simulated kill"


def test_no_kill_completes_normally(tmp_path: Path) -> None:
    # discriminator: without _kill_after the very same plan completes and produces a result, so the
    # ValueError above is attributable to the unreachable kill point, not to the plan itself.
    res = run_resumable(_one_partition_plan(), Store(tmp_path))
    assert res.report.executed == 1 and res.report.skipped == 0
