"""M39 — the multi-source / multi-stage plan builder ``graphed.shuffle_plan`` (plan §3.2, §4.4).

The single-source ``aggregate_plan`` raises on >1 partitioned source, so it cannot express a shuffle.
``shuffle_plan`` emits a **multi-stage ``DurablePlanV2``**: a stage-1 map-write (route+coalesce) that
a stage-2 gather depends on. This test witnesses the plan SHAPE (kinds + dependency edge); the actual
cross-process execution is witnessed in graphed-exec-local. The single-source builder is UNCHANGED.

Pinned surface (test-author decision, mirroring ``aggregate_plan``): ``graphed.shuffle_plan(output, *,
reduce, combine, empty, backend=None, steps_per_file=1) -> DurablePlanV2``.
"""

from __future__ import annotations

import pytest
from shuffle_backends import ListSource, ToyBackend

import graphed
from graphed import Session, aggregate_plan
from graphed.core import DurablePlanV2


def _repartitioned(s: Session):  # type: ignore[no-untyped-def]
    src = s.source("x", form="f", data=ListSource([{"__joinkey__": i, "v": i} for i in range(12)]))
    return graphed.repartition(src, by="__joinkey__").reduce("sum")


def test_shuffle_plan_emits_a_multistage_durable_plan_v2() -> None:
    s = Session(ToyBackend())
    plan = graphed.shuffle_plan(
        _repartitioned(s),
        reduce=lambda vs: vs[0],
        combine=lambda a, b: a,
        empty=lambda: None,
        backend=ToyBackend,
        steps_per_file=4,
    )
    assert isinstance(plan, DurablePlanV2), "a shuffle must serialize as the multi-stage V2 plan"
    kinds = [st.kind for st in plan.stages]
    assert "map_write" in kinds, "stage 1 routes+coalesces (map-write)"
    assert any(k in kinds for k in ("gather_join", "gather", "reduce")), "stage 2 gathers"


def test_gather_stage_depends_on_the_map_write_stage() -> None:
    s = Session(ToyBackend())
    plan = graphed.shuffle_plan(
        _repartitioned(s),
        reduce=lambda vs: vs[0],
        combine=lambda a, b: a,
        empty=lambda: None,
        backend=ToyBackend,
        steps_per_file=4,
    )
    map_idx = next(i for i, st in enumerate(plan.stages) if st.kind == "map_write")
    downstream = [st for st in plan.stages if map_idx in st.inputs]
    assert downstream, "at least one later stage must depend on the map-write stage (a real barrier edge)"


def test_single_source_aggregate_plan_is_unchanged() -> None:
    # control: the single-source builder still rejects two partitioned sources (not silently widened).
    s = Session(ToyBackend())
    a = s.source("a", form="f", data=ListSource([{"__joinkey__": 1}]))
    b = s.source("b", form="f", data=ListSource([{"__joinkey__": 2}]))
    with pytest.raises(TypeError, match="partitioned source"):
        aggregate_plan(a, b, reduce=lambda vs: vs, combine=lambda x, y: x, empty=lambda: None)
