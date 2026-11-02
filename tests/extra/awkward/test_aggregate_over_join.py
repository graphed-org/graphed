"""``aggregate_plan`` over joined arrays: several outputs, keyed or positional, in one pass."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed.aggregate import external_evaluators
from graphed.awkward import AwkwardBackend, from_parquet, gak, to_parquet
from graphed.core import DurablePlanV2, SequentialRunner
from graphed.errors import GraphedError, StageError
from graphed.execute import compile_ir
from graphed.write import PartWrite


def _sums(values: list[Any]) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def _add(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(map(operator.add, a, b))


def _zeros() -> tuple[float, float]:
    return (0.0, 0.0)


def _zero() -> tuple[float]:
    return (0.0,)


def _labelled(compiled: Any) -> tuple[Any, ...]:
    return tuple((key, (("sig_up",), frame)) for key, frame in compiled.correspondence.frames)


def _listed(value: tuple[float, ...]) -> Any:
    return list(value)


def _columns(chunks: list[ak.Array], root: Path, name: str) -> list[str]:
    paths = [str(root / f"{name}{i}.parquet") for i in range(len(chunks))]
    for chunk, path in zip(chunks, paths, strict=True):
        ak.to_parquet(chunk, path)
    return paths


def _pair(tmp_path: Path) -> tuple[Any, Any]:
    idx = np.arange(12)
    key: dict[str, Any] = {"run": np.ones(12, np.int64), "event": idx + 2**33}
    left = ak.Array({**key, "x": idx * 1.0})
    right = ak.Array({**key, "y": idx * 2.0})
    session = graphed.Session(AwkwardBackend())
    return (
        from_parquet("left", _columns([left[:6], left[6:]], tmp_path, "l"), session=session),
        from_parquet("right", _columns([right[:6], right[6:]], tmp_path, "r"), session=session),
    )


def _joined(tmp_path: Path, how: str) -> Any:
    left, right = _pair(tmp_path)
    if how == "positional":
        return graphed.join(left[["x"]], right[["y"]], how=how)
    return graphed.join(left, right, on=["run", "event"], how=how)


@pytest.mark.parametrize("how", ["positional", "inner"])
def test_several_outputs_over_a_join_are_one_plan(tmp_path: Path, how: str) -> None:
    joined = _joined(tmp_path, how)
    one, two = gak.sum(joined.x), gak.sum(joined.x + joined.y)
    plan = graphed.aggregate_plan(
        one, two, reduce=_sums, combine=_add, empty=_zeros, steps_per_file=2
    )
    assert isinstance(plan, DurablePlanV2)
    assert SequentialRunner().run(plan).value == (66.0, 198.0)
    alone = [
        SequentialRunner()
        .run(graphed.join_plan(o, steps_per_file=2, reduce=_sums, combine=_add, empty=_zero))
        .value
        for o in (one, two)
    ]
    assert alone == [(66.0,), (198.0,)]


def test_join_plan_takes_several_outputs_with_a_fold(tmp_path: Path) -> None:
    joined = _joined(tmp_path, "positional")
    plan = graphed.join_plan(
        gak.sum(joined.x), gak.sum(joined.y), reduce=_sums, combine=_add, empty=_zeros
    )
    assert SequentialRunner().run(plan).value == (66.0, 132.0)
    with pytest.raises(TypeError, match="several with reduce"):
        graphed.join_plan(joined.x, joined.y)


def test_finalize_runs_on_the_folded_value(tmp_path: Path) -> None:
    joined = _joined(tmp_path, "positional")
    plan = graphed.aggregate_plan(
        gak.sum(joined.x), reduce=_sums, combine=_add, empty=_zero, finalize=_listed
    )
    assert SequentialRunner().run(plan).value == [66.0]


def test_externals_and_on_compiled_reach_the_join_stages(tmp_path: Path) -> None:
    joined = _joined(tmp_path, "positional")
    out = gak.sum(graphed.apply(lambda v: v * 2.0, joined.x, name="double"))
    key = next(iter(external_evaluators(joined.session, compile_ir(joined.session, out))))
    seen: list[Any] = []
    plan = graphed.aggregate_plan(
        out,
        reduce=_sums,
        combine=_add,
        empty=_zero,
        externals={key: lambda v: v * 3.0},
        on_compiled=seen.append,
    )
    assert SequentialRunner().run(plan).value == (198.0,)
    assert len(seen) == 1


def test_a_failure_past_a_join_carries_its_variation_label(tmp_path: Path) -> None:
    joined = _joined(tmp_path, "positional")
    out = gak.sum(graphed.apply(lambda v: 1 / 0, joined.x, name="boom"))

    plan = graphed.aggregate_plan(
        out, reduce=_sums, combine=_add, empty=_zero, on_compiled=_labelled
    )
    with pytest.raises(StageError) as failed:
        SequentialRunner().run(plan)
    assert failed.value.variation == "sig_up"


def test_a_failure_before_a_join_carries_its_variation_label(tmp_path: Path) -> None:
    left, right = _pair(tmp_path)
    side = gak.with_field(left[["x"]], graphed.apply(lambda v: 1 / 0, left.x, name="boom"), "z")
    joined = graphed.join(side, right[["y"]], how="positional")

    plan = graphed.aggregate_plan(
        gak.sum(joined.z), reduce=_sums, combine=_add, empty=_zero, on_compiled=_labelled
    )
    with pytest.raises(StageError) as failed:
        SequentialRunner().run(plan)
    assert failed.value.variation == "sig_up"


def test_a_refused_plan_leaves_on_compiled_unfired(tmp_path: Path) -> None:
    rows = ak.Array({"x": np.arange(4.0)})
    session = graphed.Session(AwkwardBackend())
    left = from_parquet("left", _columns([rows[:2], rows[2:]], tmp_path, "l"), session=session)
    right = from_parquet("right", _columns([rows], tmp_path, "r"), session=session)
    joined = graphed.join(left[["x"]], gak.zip({"y": right.x}), how="positional")
    seen: list[Any] = []
    with pytest.raises(TypeError, match="one partition per partition"):
        graphed.aggregate_plan(
            gak.sum(joined.x), reduce=_sums, combine=_add, empty=_zeros, on_compiled=seen.append
        )
    assert seen == []


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"partitions": ()}, "partitions="),
        ({"store": "somewhere"}, "store="),
        ({"read_by": ()}, "read_by="),
        ({"opt_level": 0}, "opt_level=0"),
    ],
    ids=["partitions", "store", "read_by", "opt_level"],
)
def test_what_a_join_stage_cannot_do_is_refused(
    tmp_path: Path, kwargs: dict[str, Any], name: str
) -> None:
    joined = _joined(tmp_path, "positional")
    with pytest.raises(TypeError, match=f"over a join takes no {name}"):
        graphed.aggregate_plan(
            gak.sum(joined.x), reduce=_sums, combine=_add, empty=_zeros, **kwargs
        )


def test_writes_over_a_join_are_refused(tmp_path: Path) -> None:
    joined = _joined(tmp_path, "positional")
    write = PartWrite(joined.x, str(tmp_path / "out"), str, print)
    with pytest.raises(TypeError, match="over a join takes no writes="):
        graphed.aggregate_plan(reduce=_sums, combine=_add, empty=_zeros, writes=[write])


def test_two_sources_with_no_join_are_still_refused(tmp_path: Path) -> None:
    left, right = _pair(tmp_path)
    appended = gak.concatenate([left.x, right.y])
    with pytest.raises(TypeError, match="exactly one partitioned source"):
        graphed.aggregate_plan(appended, reduce=_sums, combine=_add, empty=_zeros)


def test_an_output_reading_a_side_directly_is_refused(tmp_path: Path) -> None:
    left, right = _pair(tmp_path)
    joined = graphed.join(left[["x"]], right[["y"]], how="positional")
    with pytest.raises(TypeError, match="reads a source directly"):
        graphed.aggregate_plan(
            gak.sum(joined.x), gak.sum(left.x), reduce=_sums, combine=_add, empty=_zeros
        )


def test_a_keyed_join_of_one_source_is_refused(tmp_path: Path) -> None:
    left, _ = _pair(tmp_path)
    joined = graphed.join(left[["run", "x"]], left[["run", "event"]], on=["run"])
    with pytest.raises(GraphedError, match="cannot join one source by key"):
        graphed.aggregate_plan(gak.sum(joined.x), reduce=_sums, combine=_add, empty=_zeros)
    with pytest.raises(GraphedError, match="cannot join one source by key"):
        to_parquet(joined, str(tmp_path / "out"), compute=False)


def test_a_positional_join_of_one_source_runs_per_partition(tmp_path: Path) -> None:
    left, _ = _pair(tmp_path)
    joined = graphed.join(left[["x"]], left[["run"]], how="positional")
    plan = graphed.aggregate_plan(gak.sum(joined.x), reduce=_sums, combine=_add, empty=_zero)
    assert SequentialRunner().run(plan).value == (66.0,)


def test_a_staged_plan_is_not_collated(tmp_path: Path) -> None:
    joined = _joined(tmp_path, "positional")
    plan = graphed.aggregate_plan(gak.sum(joined.x), reduce=_sums, combine=_add, empty=_zeros)
    with pytest.raises(TypeError, match="'staged' runs in stages"):
        graphed.collate({"staged": plan})


def test_join_plan_needs_an_output() -> None:
    with pytest.raises(TypeError, match="takes one output"):
        graphed.join_plan(reduce=_sums, combine=_add, empty=_zeros)
