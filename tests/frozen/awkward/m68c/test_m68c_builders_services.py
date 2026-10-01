"""m68c §3.3 builders: ``join_plan`` and ``shuffle_plan`` carry the services their External nodes
name (plus ``services=``), and building the same plan twice gives the same bytes, services included."""

from __future__ import annotations

from m68c_services_harness import EXTRA, FOLD, SPEC, corrected, one_source, two_sources

import graphed
from graphed.core import DurablePlanV2


def test_join_plan_and_shuffle_plan_carry_the_named_services_deterministically() -> None:
    ev, lu = two_sources()
    served = graphed.join(corrected(ev), lu, on=["run"])
    plan = graphed.join_plan(served)
    assert plan.services == (SPEC,)
    assert graphed.join_plan(graphed.join(ev, lu, on=["run"])).services == ()
    again = graphed.join_plan(served)
    assert again.to_bytes() == plan.to_bytes()
    assert DurablePlanV2.from_bytes(plan.to_bytes()).services == (SPEC,)
    assert graphed.join_plan(served, services=["extra"]).services == (EXTRA, SPEC)

    rep = graphed.repartition(corrected(one_source()), n=2)
    shuffled = graphed.shuffle_plan(rep, **FOLD)
    assert shuffled.services == (SPEC,)
    assert graphed.shuffle_plan(rep, **FOLD).to_bytes() == shuffled.to_bytes()
    assert graphed.shuffle_plan(rep, services=["extra"], **FOLD).services == (EXTRA, SPEC)
