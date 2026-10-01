"""A `Replay` built directly, without `replay()`'s compile, recompiles at the plan's level."""

from __future__ import annotations

from typing import Any

import m69b_opt_fixtures as fx
import pytest

from graphed import aggregate_plan
from graphed.aggregate import _PartitionReduce
from graphed.core import LocalResources
from graphed.debug.replaying import Replay


@pytest.mark.parametrize("opt_level", [0, 1])
def test_direct_replay_reduces_what_the_run_reduces(opt_level: int) -> None:
    _s, events = fx.record()
    outputs = fx.merged_pair(events)
    plan = aggregate_plan(
        *outputs, reduce=fx.sums, combine=fx.add, empty=fx.zero, steps_per_file=2, opt_level=opt_level
    )
    assert isinstance(plan.process, _PartitionReduce)
    task = plan.tasks[0]
    run: Any = plan.process(task.partition, LocalResources())
    assert len(run) == 2 - opt_level
    assert Replay(plan.process, task, outputs).value.tolist() == run.tolist()
