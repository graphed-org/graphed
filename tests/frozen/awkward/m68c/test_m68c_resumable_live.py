"""m68c §3.4: ``run_shuffle_resumable`` runs a bound ``join_plan`` whose map stage calls a service;
``plan.value`` over its last-stage blocks equals the ``SequentialRunner`` value."""

from __future__ import annotations

from pathlib import Path

import pytest
from m68c_services_harness import FOLD, corrected, joined, rows, serve, two_sources

import graphed
from graphed.checkpoint import Store, run_shuffle_resumable
from graphed.core.execution import SequentialRunner
from graphed.services import bind_services


@pytest.mark.parametrize("fold", [True, False])
def test_resumable_bound_join_plan_equals_the_sequential_value(fold: bool, tmp_path: Path) -> None:
    ev, lu = two_sources()
    served = graphed.join(corrected(ev), lu, on=["run"], how="outer")
    plan = graphed.join_plan(served, steps_per_file=2, **(FOLD if fold else {}))
    with serve() as server:
        bound = bind_services(plan, {"sf": server.url})
        store = Store(tmp_path / "store")
        res = run_shuffle_resumable(bound, store)
        assert server.requests > 0
        assert res.report.executed == sum(len(st.tasks) for st in plan.stages)
        value = bound.value([store.get(h) for h in res.value])
        expected = SequentialRunner().run(bound).value
    assert len(value) == len(expected) == 2
    union, want = joined(value), joined(expected)
    assert str(union.type) == str(want.type)
    assert rows(union) == rows(want)
