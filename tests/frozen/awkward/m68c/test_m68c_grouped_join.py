"""m68c §3.3: ``gak.join(grouped=True)`` for inner and left runs through a V2 plan per dest, one dest
having no left rows, and gives ``materialize``'s grouped type and sublists."""

from __future__ import annotations

import pytest
from m68c_services_harness import FOLD, joined, sublists, two_sources

import graphed
from graphed.awkward import gak
from graphed.core.execution import SequentialRunner


@pytest.mark.parametrize("fold", [True, False])
@pytest.mark.parametrize("how", ["inner", "left"])
def test_grouped_join_plan_matches_materialize(how: str, fold: bool) -> None:
    ev, lu = two_sources()
    grouped = gak.join(ev, lu, on=["run"], how=how, grouped=True)
    expected = ev.session.materialize(grouped)
    plan = graphed.join_plan(grouped, steps_per_file=2, **(FOLD if fold else {}))
    res = SequentialRunner().run(plan)
    assert len(res.value) == 2
    assert len(res.value[1]) == 0
    union = joined(res.value)
    assert str(union.type) == str(expected.type)
    assert " * var * " in str(union.type)
    assert sublists(union) == sublists(expected)
