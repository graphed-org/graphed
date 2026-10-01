"""m68c §3.2/§3.4: an unbound V2 plan that calls a service is refused with ``UnboundService`` naming
it before any stage process runs, by ``SequentialRunner`` and by ``run_shuffle_resumable``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from m68c_services_harness import FOLD, READS, corrected, one_source, two_sources

import graphed
from graphed.checkpoint import Store, run_shuffle_resumable
from graphed.core import DurablePlanV2
from graphed.core.execution import SequentialRunner
from graphed.services import UnboundService


def _join() -> DurablePlanV2:
    ev, lu = two_sources()
    return graphed.join_plan(graphed.join(corrected(ev), lu, on=["run"]), steps_per_file=2)


def _shuffle() -> DurablePlanV2:
    return graphed.shuffle_plan(graphed.repartition(corrected(one_source()), n=2), steps_per_file=2, **FOLD)


def _sequential(plan: DurablePlanV2, tmp_path: Path) -> Any:
    return SequentialRunner().run(plan)


def _resumable(plan: DurablePlanV2, tmp_path: Path) -> Any:
    return run_shuffle_resumable(plan, Store(tmp_path / "store"))


@pytest.mark.parametrize("run", [_sequential, _resumable])
@pytest.mark.parametrize("build", [_join, _shuffle])
def test_an_unbound_v2_plan_is_refused_before_any_process_call(build: Any, run: Any, tmp_path: Path) -> None:
    plan = build()
    READS.clear()
    with pytest.raises(UnboundService, match="'sf'"):
        run(plan, tmp_path)
    assert READS == []
    if run is _resumable:
        assert Store(tmp_path / "store").completed() == {}
