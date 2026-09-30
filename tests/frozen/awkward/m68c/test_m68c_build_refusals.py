"""m68c §3.3 build-time refusals: a size-driven repartition, a post-join read of a source, a chunk
partial consumed on either side of the barrier (or left unfolded at a join output), and a grouped
right/outer join, each refused when the plan is built."""

from __future__ import annotations

from typing import Any

import pytest
from m68c_services_harness import FOLD, one_source, two_sources

import graphed
from graphed.awkward import gak
from graphed.errors import GraphedError


def test_target_bytes_repartition_is_refused_by_shuffle_plan() -> None:
    graphed.shuffle_plan(graphed.repartition(one_source(), n=2), **FOLD)
    with pytest.raises(TypeError, match="target_bytes"):
        graphed.shuffle_plan(graphed.repartition(one_source(), target_bytes=100), **FOLD)


def test_a_post_join_operation_reading_a_source_directly_is_refused() -> None:
    ev, lu = two_sources()
    j = graphed.join(ev, lu, on=["run"])
    graphed.join_plan(gak.with_field(j, j.x, "z"))
    with pytest.raises(TypeError):
        graphed.join_plan(gak.with_field(j, ev.x, "z"))


def _join_sum_before() -> Any:
    ev, lu = two_sources()
    return graphed.join_plan(graphed.join(gak.with_field(ev, ev.x / gak.sum(ev.x), "y"), lu, on=["run"]))


def _join_sum_after() -> Any:
    ev, lu = two_sources()
    j = graphed.join(ev, lu, on=["run"])
    return graphed.join_plan(j.x / gak.sum(j.x))


def _join_sum_unfolded() -> Any:
    ev, lu = two_sources()
    return graphed.join_plan(gak.sum(graphed.join(ev, lu, on=["run"]).x))


def _shuffle_sum_before() -> Any:
    ev = one_source()
    return graphed.shuffle_plan(graphed.repartition(gak.with_field(ev, ev.x / gak.sum(ev.x), "y"), n=2), **FOLD)


def _shuffle_sum_after() -> Any:
    r = graphed.repartition(one_source(), n=2)
    return graphed.shuffle_plan(r.x / gak.sum(r.x), **FOLD)


@pytest.mark.parametrize(
    "build", [_join_sum_before, _join_sum_after, _join_sum_unfolded, _shuffle_sum_before, _shuffle_sum_after]
)
def test_a_chunk_partial_is_refused(build: Any) -> None:
    graphed.shuffle_plan(gak.sum(graphed.repartition(one_source(), n=2).x), **FOLD)
    with pytest.raises(GraphedError, match=r"'ak\.sum'"):
        build()
    ev, lu = two_sources()
    graphed.join_plan(gak.sum(graphed.join(ev, lu, on=["run"]).x), **FOLD)


@pytest.mark.parametrize("how", ["right", "outer"])
def test_grouped_right_and_outer_joins_are_refused(how: str) -> None:
    ev, lu = two_sources()
    graphed.join_plan(gak.join(ev, lu, on=["run"], how="left", grouped=True))
    with pytest.raises(TypeError, match="grouped"):
        graphed.join_plan(gak.join(ev, lu, on=["run"], how=how, grouped=True))
