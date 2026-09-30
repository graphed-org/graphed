"""m68c §3.2/§3.3: ``resolve_services`` on a V2 plan reaches a ``Resolvable`` ``reduce`` held by its
gather stage."""

from __future__ import annotations

from m68c_services_harness import RESOLVED, ResolvingReduce, concat, nothing, one_source

import graphed
from graphed.services import resolve_services


def test_resolve_services_reaches_the_gathers_reduce() -> None:
    rep = graphed.repartition(one_source(), n=2)
    plan = graphed.shuffle_plan(rep, reduce=ResolvingReduce(), combine=concat, empty=nothing)
    RESOLVED.clear()
    assert resolve_services(plan, ["value"]) == ("resolved", ["value"])
    assert RESOLVED == [["value"]]
