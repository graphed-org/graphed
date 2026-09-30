"""m68c §3.2/§3.3: a server call before a repartition, on one side before a join, and after a join
runs through ``bind_services`` then ``SequentialRunner`` against a local HTTP stand-in."""

from __future__ import annotations

from typing import Any

import pytest
from m68c_services_harness import (
    FOLD,
    SF,
    by_hand,
    corrected,
    joined,
    one_source,
    rows,
    scaled,
    serve,
    two_sources,
)

import graphed
from graphed.awkward import gak
from graphed.core import DurablePlanV2
from graphed.core.execution import SequentialRunner
from graphed.services import bind_services


def _before_repartition() -> tuple[DurablePlanV2, Any]:
    ev = one_source()
    plan = graphed.shuffle_plan(graphed.repartition(corrected(ev), n=2), steps_per_file=2, **FOLD)
    return plan, ev.session.materialize(graphed.repartition(by_hand(ev), n=2))


def _before_join() -> tuple[DurablePlanV2, Any]:
    ev, lu = two_sources()
    plan = graphed.join_plan(graphed.join(corrected(ev), lu, on=["run"], how="left"), steps_per_file=2, **FOLD)
    return plan, ev.session.materialize(graphed.join(by_hand(ev), lu, on=["run"], how="left"))


def _after_join() -> tuple[DurablePlanV2, Any]:
    ev, lu = two_sources()
    j = graphed.join(ev, lu, on=["run"])
    plan = graphed.join_plan(gak.with_field(j, scaled(j.x), "x"), steps_per_file=2)
    return plan, ev.session.materialize(gak.with_field(j, j.x * SF, "x"))


@pytest.mark.parametrize("build", [_before_repartition, _before_join, _after_join])
def test_a_bound_server_call_runs_in_a_v2_plan(build: Any) -> None:
    plan, expected = build()
    with serve() as server:
        value = SequentialRunner().run(bind_services(plan, {"sf": server.url})).value
        assert server.requests > 0
    union = joined(value)
    assert str(union.type) == str(expected.type)
    assert rows(union) == rows(expected)
