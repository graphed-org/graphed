"""``aggregate_plan`` with no ``reduce``: reductions that know their own combine."""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import Session
from graphed.aggregate import fold_of
from graphed.awkward import AwkwardBackend, from_parquet, gak
from graphed.checkpoint import resumable
from graphed.core.execution import LocalResources, SequentialRunner
from graphed.numpy import NumpyBackend, from_array

pytest.importorskip("pyarrow")

N = 12


@pytest.fixture
def events(tmp_path: Path) -> Any:
    path = tmp_path / "events.parquet"
    ak.to_parquet(
        ak.Array({"x": [float(i) for i in range(N)], "j": [[1, 2]] * (N // 2) + [[3]] * (N // 2)}),
        path,
    )
    return from_parquet("events", [str(path)], session=Session(AwkwardBackend()), steps_per_file=4)


def _row_wise(values: Any) -> Any:
    return values


def _run(*outputs: Any) -> Any:
    return SequentialRunner().run(graphed.aggregate_plan(*outputs)).value


def test_several_sums_equal_materialize_after_a_pickle(events: Any) -> None:
    outputs = (gak.sum(events.x > 3), gak.count(events.j), gak.count_nonzero(events.x))
    plan = graphed.aggregate_plan(*outputs)
    assert len(plan.tasks) == 4
    value = SequentialRunner().run(pickle.loads(pickle.dumps(plan))).value
    assert value == graphed.materialize(outputs) == (8, 18, 11)


def test_one_output_is_its_value_not_a_tuple(events: Any) -> None:
    assert _run(gak.sum(events.x)) == sum(range(N))


def test_equal_outputs_read_their_one_merged_slot(events: Any) -> None:
    total = gak.sum(events.x)
    assert _run(total, gak.max(events.x), total) == (66.0, 11.0, 66.0)


@pytest.mark.parametrize(
    ("fn", "want"), [(gak.max, 11.0), (gak.min, 9.0), (gak.any, True), (gak.all, True)]
)
def test_chunks_with_no_rows_leave_the_fold_alone(events: Any, fn: Any, want: Any) -> None:
    passing = events.x[events.x > 8.5]
    assert _run(fn(passing)) == want == ak.to_list(graphed.materialize(fn(passing)))


def test_a_folded_value_ignores_which_side_is_empty(events: Any) -> None:
    assert _run(gak.max(events.x[events.x < 2.5])) == 2.0
    assert _run(gak.min(events.x[events.x < 2.5])) == 0.0


def test_no_rows_at_all_is_none_as_over_one_empty_array(events: Any) -> None:
    none = events.x[events.x > 100]
    assert _run(gak.max(none)) is None
    assert graphed.materialize(gak.max(none)) is None
    assert _run(gak.sum(none)) == 0.0


def test_a_mask_identity_max_stays_the_awkward_fill(events: Any) -> None:
    none = events.x[events.x > 100]
    assert _run(gak.max(none, mask_identity=False)) == -math.inf


@pytest.mark.parametrize("fn", [gak.max, gak.min])
@pytest.mark.parametrize("first_chunk", [True, False])
def test_nan_wins_as_it_does_in_awkward(tmp_path: Path, fn: Any, first_chunk: bool) -> None:
    path = tmp_path / "nan.parquet"
    rows = [math.nan, 1.0, 2.0, 3.0] if first_chunk else [1.0, 2.0, math.nan, 3.0]
    ak.to_parquet(ak.Array({"x": rows}), path)
    events = from_parquet("e", [str(path)], session=Session(AwkwardBackend()), steps_per_file=2)
    assert math.isnan(_run(fn(events.x)))
    assert math.isnan(graphed.materialize(fn(events.x)))


def _tree(plan: Any) -> Any:
    """Run ``plan`` as a pool does: pairwise over the partials, never seeded with ``empty()``."""
    level = [plan.process(task.partition, LocalResources()) for task in plan.tasks]
    while len(level) > 1:
        pairs = zip(level[::2], level[1::2], strict=False)
        tail = level[len(level) & ~1 :]
        level = [plan.combine(a, b) for a, b in pairs] + tail
    return plan.finalize(level[0])


@pytest.mark.parametrize(
    "fn", [gak.sum, gak.count, gak.count_nonzero, gak.any, gak.all, gak.max, gak.min]
)
def test_a_tree_and_a_fold_agree_on_no_rows(events: Any, fn: Any) -> None:
    none = events.x[events.x > 100]
    plan = graphed.aggregate_plan(fn(none, mask_identity=True))
    assert SequentialRunner().run(plan).value is None
    assert _tree(plan) is None
    assert graphed.materialize(fn(none, mask_identity=True)) is None


def test_a_tree_gives_the_folds_values(events: Any) -> None:
    outputs = (gak.sum(events.x), gak.count(events.x), gak.max(events.x), gak.min(events.x))
    plan = graphed.aggregate_plan(*outputs)
    assert _tree(plan) == SequentialRunner().run(plan).value == (66.0, 12, 11.0, 0.0)


def test_the_numpy_spelling_folds_too() -> None:
    x = from_array(Session(NumpyBackend()), "x", np.arange(6.0))
    assert fold_of(x.sum())(1.0, 2.0) == 3.0
    assert fold_of(x.max())(1.0, 2.0) == 2.0
    with pytest.raises(TypeError, match="with axis= or keepdims="):
        fold_of(x.sum(axis=0))


def test_a_per_event_output_refuses_naming_the_way_out(events: Any) -> None:
    with pytest.raises(TypeError, match=r"output 1 is per-event.*pass reduce=, combine=, empty="):
        graphed.aggregate_plan(gak.sum(events.x), events.x * 2)


def test_an_inner_axis_sum_is_per_event(events: Any) -> None:
    with pytest.raises(TypeError, match="output 0 is per-event"):
        graphed.aggregate_plan(gak.sum(events.j, axis=1))


@pytest.mark.parametrize(
    ("fn", "why"),
    [
        (gak.mean, "mean of chunk means"),
        (gak.var, "variance of chunk variances"),
        (gak.std, "std of chunk stds"),
    ],
)
def test_means_refuse_and_say_why(events: Any, fn: Any, why: str) -> None:
    with pytest.raises(TypeError, match=why):
        graphed.aggregate_plan(fn(events.x))


def test_a_reduction_with_no_known_combine_refuses(events: Any) -> None:
    with pytest.raises(TypeError, match=r"ak\.prod, which has no known combine"):
        graphed.aggregate_plan(gak.prod(events.x))


@pytest.mark.parametrize("kwargs", [{"axis": 0}, {"keepdims": True}])
def test_a_reduction_keeping_an_axis_refuses(events: Any, kwargs: dict[str, Any]) -> None:
    with pytest.raises(TypeError, match="with axis= or keepdims="):
        graphed.aggregate_plan(gak.sum(events.x, **kwargs))


def test_a_reduction_feeding_another_node_keeps_its_own_refusal(events: Any) -> None:
    with pytest.raises(graphed.GraphedError, match="feeds another node"):
        graphed.aggregate_plan(gak.sum(events.x) / gak.count(events.x))


@pytest.mark.parametrize("given", ["reduce", "combine", "empty"])
def test_a_partial_default_refuses(events: Any, given: str) -> None:
    kwargs: Any = {given: list}
    with pytest.raises(TypeError, match="together, or none"):
        graphed.aggregate_plan(gak.sum(events.x), **kwargs)


def test_writes_need_a_reduce(events: Any, tmp_path: Path) -> None:
    write = graphed.awkward.parquet_write(events.x, str(tmp_path / "out"), name=str)
    with pytest.raises(TypeError, match=r"writes=\) needs reduce="):
        graphed.aggregate_plan(gak.sum(events.x), writes=[write])


def test_a_given_reduction_is_used_as_it_was(events: Any) -> None:
    plan = graphed.aggregate_plan(
        gak.sum(events.x), reduce=lambda vs: [float(vs[0])], combine=list.__add__, empty=list
    )
    assert SequentialRunner().run(plan).value == [3.0, 12.0, 21.0, 30.0]


def test_a_finalize_takes_the_value_a_run_would_give(events: Any) -> None:
    half = graphed.aggregate_plan(gak.sum(events.x), finalize=lambda value: value / 2)
    assert SequentialRunner().run(pickle.loads(pickle.dumps(half))).value == 33.0
    both = graphed.aggregate_plan(gak.sum(events.x), gak.max(events.x), finalize=sum)
    assert SequentialRunner().run(both).value == 77.0


def test_an_external_output_is_not_called_per_event(events: Any) -> None:
    mapped = events.x.map(_row_wise, name="row_wise", output_type="float64")
    with pytest.raises(TypeError, match="output 0 is external, not a reduction"):
        graphed.aggregate_plan(mapped)


def test_an_axis_free_fold_is_public(events: Any) -> None:
    assert fold_of(gak.sum(events.x))(1, 2) == 3
    with pytest.raises(TypeError, match="is per-event"):
        fold_of(events.x)


def test_a_default_plan_resumes(events: Any, tmp_path: Path) -> None:
    plan = graphed.aggregate_plan(gak.sum(events.x), gak.max(events.x))
    store = str(tmp_path / "store")
    first = SequentialRunner().run(resumable(plan, store)).value
    assert first == SequentialRunner().run(resumable(plan, store)).value == (66.0, 11.0)
