"""m68c §3.3/§3.4: ``SequentialRunner`` runs builder output. The map stage reads each partition and
evaluates the IR up to the barrier, the gather joins or concatenates per dest and runs what was
recorded after the barrier, and a ``reduce`` stage folds the per-dest values in dest order."""

from __future__ import annotations

from collections import Counter
from typing import Any

import awkward as ak
import pytest
from m68c_services_harness import (
    FOLD,
    READS,
    joined,
    lumi_files,
    new_session,
    one_source,
    rows,
    source,
    two_sources,
)

import graphed
from graphed.awkward import AwkwardBackend, gak
from graphed.core.execution import SequentialRunner

HOWS = ["inner", "left", "right", "outer"]
JOIN_READS = Counter({("events", "mem://events/0"): 2, ("lumi", "mem://lumi/0"): 2, ("lumi", "mem://lumi/1"): 2})


def _joined(how: str) -> tuple[Any, Any]:
    """A join with an operation after it, and what ``materialize`` gives for it."""
    ev, lu = two_sources()
    j = graphed.join(ev, lu, on=["run"], how=how)
    post = gak.with_field(j, j.run * 10, "run10")
    assert rows(ev.session.materialize(post)) != rows(ev.session.materialize(j))
    return post, ev.session.materialize(post)


def _assert_same(blocks: Any, expected: Any) -> None:
    union = joined(blocks)
    assert str(union.type) == str(expected.type)
    assert rows(union) == rows(expected)


@pytest.mark.parametrize("how", HOWS)
def test_join_plan_with_reduce_folds_the_post_join_value_per_dest(how: str) -> None:
    post, expected = _joined(how)
    plan = graphed.join_plan(post, steps_per_file=2, **FOLD)
    assert [st.kind for st in plan.stages] == ["map_write", "map_write", "gather_join", "reduce"]
    assert plan.stages[2].routing["broadcast"] is True
    READS.clear()
    res = SequentialRunner().run(plan)
    assert Counter(READS) == JOIN_READS
    assert (res.n_partitions, res.n_combines) == (6, 3)
    assert len(res.value) == 2
    # every events run hashes to dest 0, so dest 1 holds right-side rows only
    right_only = res.value[1]
    if how in ("right", "outer"):
        assert len(right_only) > 0
        assert ak.all(ak.is_none(right_only.x))
    else:
        assert len(right_only) == 0
    _assert_same(res.value, expected)


@pytest.mark.parametrize("how", HOWS)
def test_join_plan_without_reduce_returns_the_per_dest_blocks(how: str) -> None:
    post, expected = _joined(how)
    plan = graphed.join_plan(post, steps_per_file=2)
    READS.clear()
    res = SequentialRunner().run(plan)
    assert Counter(READS) == JOIN_READS
    assert isinstance(res.value, tuple)
    assert len(res.value) == 2
    _assert_same(res.value, expected)


def test_join_plan_keeps_the_later_registered_source_on_the_left() -> None:
    ev, lu = two_sources()
    j = graphed.join(lu, ev, on=["run"], how="left")
    expected = ev.session.materialize(j)
    res = SequentialRunner().run(graphed.join_plan(j, steps_per_file=2))
    _assert_same(res.value, expected)


def test_shuffle_plan_by_key_routes_each_key_to_one_dest_and_runs_the_post_op() -> None:
    lu = source(new_session(), "lumi", *lumi_files())
    rep = graphed.repartition(lu, by="run")
    post = gak.with_field(rep, rep.w * 2, "w2")
    expected = lu.session.materialize(post)
    READS.clear()
    res = SequentialRunner().run(graphed.shuffle_plan(post, steps_per_file=2, **FOLD))
    assert Counter(READS) == Counter({("lumi", "mem://lumi/0"): 2, ("lumi", "mem://lumi/1"): 2})
    routed = AwkwardBackend().partition(expected, "run", 2)
    assert [sorted(set(b.run.tolist())) for b in res.value] == [sorted(set(b.run.tolist())) for b in routed]
    assert all(len(b) for b in res.value)
    _assert_same(res.value, expected)


def test_shuffle_plan_by_count_sends_each_map_block_whole_to_one_dest() -> None:
    ev = one_source()
    rep = graphed.repartition(ev[ev.x > 25], n=3)
    expected = ev.session.materialize(rep)
    res = SequentialRunner().run(graphed.shuffle_plan(rep, steps_per_file=2, **FOLD))
    assert [len(b) for b in res.value] == [0, 4, 0]
    assert all(str(ak.Array(b).type.content) == str(expected.type.content) for b in res.value)
    _assert_same(res.value, expected)
