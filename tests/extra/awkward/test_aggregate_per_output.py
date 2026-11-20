"""``reduce`` receives one value per output passed, in order, wherever outputs merge (#228)."""

from __future__ import annotations

import operator
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import PlanPieces
from graphed.aggregate import VALUES_PER_OUTPUT, _PerOutput
from graphed.awkward import AwkwardBackend, from_parquet, gak
from graphed.checkpoint.resume import _dumps
from graphed.core import SequentialRunner
from graphed.errors import GraphedError
from graphed.execute import CompiledGraph, compile_ir

pytest.importorskip("pyarrow")


def _floats(values: list[Any]) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def _add(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(map(operator.add, a, b)) if a else b


def _none() -> tuple[float, ...]:
    return ()


def _events(tmp_path: Path) -> Any:
    ak.to_parquet(
        ak.Array({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]}), tmp_path / "p.parquet"
    )
    return from_parquet("ev", str(tmp_path), session=graphed.Session(AwkwardBackend()))


def _merging(ev: Any) -> dict[str, Any]:
    """Five outputs the compile makes three: the same sum twice, and a sum commuted."""
    return {
        "x": gak.sum(ev.x),
        "x again": gak.sum(ev.x),
        "y": gak.sum(ev.y),
        "x+y": gak.sum(ev.x + ev.y),
        "y+x": gak.sum(ev.y + ev.x),
    }


EXPECTED = {"x": 6.0, "x again": 6.0, "y": 60.0, "x+y": 66.0, "y+x": 66.0}


@pytest.mark.parametrize("opt_level", [1, 0])
def test_each_name_pairs_with_its_own_value(tmp_path: Path, opt_level: int) -> None:
    outputs = _merging(_events(tmp_path))
    plan = graphed.aggregate_plan(
        *outputs.values(), reduce=_floats, combine=_add, empty=_none, opt_level=opt_level
    )
    assert dict(zip(outputs, SequentialRunner().run(plan).value, strict=True)) == EXPECTED


def test_the_default_fold_gives_one_value_per_output(tmp_path: Path) -> None:
    outputs = _merging(_events(tmp_path))
    value = SequentialRunner().run(graphed.aggregate_plan(*outputs.values())).value
    assert dict(zip(outputs, map(float, value), strict=True)) == EXPECTED


def _join(tmp_path: Path) -> Any:
    idx = np.arange(6)
    session = graphed.Session(AwkwardBackend())
    for name, col, scale in (("l", "x", 1.0), ("r", "y", 10.0)):
        ak.to_parquet(ak.Array({"event": idx, col: idx * scale}), tmp_path / f"{name}.parquet")
    left = from_parquet("l", str(tmp_path / "l.parquet"), session=session)
    right = from_parquet("r", str(tmp_path / "r.parquet"), session=session)
    return graphed.join(left, right, on=["event"], how="inner")


@pytest.mark.parametrize("builder", ["aggregate_plan", "join_plan"])
def test_a_join_hands_out_one_value_per_output(tmp_path: Path, builder: str) -> None:
    joined = _join(tmp_path)
    outputs = (gak.sum(joined.x), gak.sum(joined.x), gak.sum(joined.y))
    plan = getattr(graphed, builder)(*outputs, reduce=_floats, combine=_add, empty=_none)
    assert SequentialRunner().run(plan).value == (15.0, 15.0, 150.0)


class _Pieces:
    """A deferred object whose share reads each output at its position in ``requested``."""

    def __init__(self, *outputs: Any, module: str | None = None, marked: bool = False) -> None:
        self.outputs, self.positions = outputs, [0] * len(outputs)

        def on_compiled(compiled: CompiledGraph) -> None:
            first = {nid: i for i, nid in reversed(list(enumerate(compiled.requested)))}
            self.positions = [first[o.node_id] for o in outputs]

        if module is not None:
            on_compiled.__module__ = module
        if marked:
            setattr(on_compiled, VALUES_PER_OUTPUT, True)
        self.on_compiled = on_compiled

    def reduce(self, values: list[Any]) -> tuple[float, ...]:
        return tuple(float(values[p]) for p in self.positions)

    def __graphed_pieces__(self) -> PlanPieces:
        return PlanPieces(
            outputs=self.outputs,
            reduce=self.reduce,
            combine=_add,
            empty=_none,
            finalize=lambda value: value,
            on_compiled=self.on_compiled,
        )


def test_a_piece_reads_its_outputs_at_their_requested_positions(tmp_path: Path) -> None:
    ev = _events(tmp_path)
    piece = _Pieces(gak.sum(ev.y), gak.sum(ev.x), gak.sum(ev.x))
    assert graphed.materialize([gak.sum(ev.x), piece]) == [6.0, (60.0, 6.0, 6.0)]


def _hist_reduce(module: str, marked: bool) -> Any:
    class Reduce:
        def __call__(self, values: list[Any]) -> tuple[float, ...]:
            return _floats(values)

    Reduce.__module__ = module
    if marked:
        setattr(Reduce, VALUES_PER_OUTPUT, True)
    return Reduce()


@pytest.mark.parametrize("hook", ["reduce", "on_compiled"])
def test_an_unmarked_histogram_hook_is_refused_where_outputs_merge(
    tmp_path: Path, hook: str
) -> None:
    ev = _events(tmp_path)
    merged = (gak.sum(ev.x), gak.sum(ev.x))
    stand_in = _hist_reduce("graphed_histogram._combine", marked=False)
    kw: dict[str, Any] = {"reduce": stand_in if hook == "reduce" else _floats}
    if hook == "on_compiled":
        kw["on_compiled"] = _Pieces(*merged, module="graphed_histogram.boost").on_compiled
    with pytest.raises(GraphedError, match=r"upgrade graphed-histogram to 0\.0\.9"):
        graphed.aggregate_plan(*merged, combine=_add, empty=_none, **kw)


def test_an_unmarked_histogram_piece_is_refused_where_outputs_merge(tmp_path: Path) -> None:
    ev = _events(tmp_path)
    piece = _Pieces(gak.sum(ev.x), gak.sum(ev.x), module="graphed_histogram.boost")
    with pytest.raises(GraphedError, match="upgrade graphed-histogram"):
        graphed.materialize(piece)


@pytest.mark.parametrize(
    ("module", "marked", "outputs"),
    [
        ("graphed_histogram._combine", True, ("x", "x")),
        ("graphed_histogram._combine", False, ("x", "y")),
        ("graphed_histogram._combine", True, ("x", "y")),
        ("analysis", False, ("x", "x")),
    ],
    ids=["marked-merged", "unmarked-unmerged", "marked-unmerged", "other-module-merged"],
)
def test_other_hooks_pass(
    tmp_path: Path, module: str, marked: bool, outputs: tuple[str, ...]
) -> None:
    ev = _events(tmp_path)
    arrays = [gak.sum(getattr(ev, c)) for c in outputs]
    plan = graphed.aggregate_plan(
        *arrays, reduce=_hist_reduce(module, marked), combine=_add, empty=_none
    )
    want = tuple(EXPECTED[c] for c in outputs)
    assert SequentialRunner().run(plan).value == want
    piece = _Pieces(*arrays, module=module, marked=marked)
    assert graphed.materialize(piece) == want


def test_requested_keeps_the_order_and_repeats_asked(tmp_path: Path) -> None:
    ev = _events(tmp_path)
    a, b = gak.sum(ev.x), gak.sum(ev.y)
    compiled = compile_ir(ev.session, a, b, a)
    assert compiled.requested == (a.node_id, b.node_id, a.node_id)
    assert pickle.loads(pickle.dumps(compiled)).requested == compiled.requested


def test_requested_stays_out_of_a_checkpoint_key_and_an_unset_pickle(tmp_path: Path) -> None:
    ev = _events(tmp_path)
    compiled = compile_ir(ev.session, gak.sum(ev.x))
    unset = replace(compiled, requested=())
    before = object.__new__(CompiledGraph)  # the shape from before the field existed
    before.__dict__.update({k: v for k, v in vars(compiled).items() if k != "requested"})
    assert _dumps(compiled, {}, set()) == _dumps(before, {}, set())
    assert pickle.dumps(unset) == pickle.dumps(before)
    assert pickle.loads(pickle.dumps(unset)).requested == ()


class _Bound:
    def __init__(self, endpoints: Any = None) -> None:
        self.endpoints = endpoints

    def __call__(self, values: list[Any]) -> list[Any]:
        return [*values, self.endpoints]

    def bind_services(self, endpoints: Any) -> _Bound:
        return _Bound(endpoints)

    def resolve_services(self, value: Any) -> Any:
        return ("resolved", value)

    def output_summaries(self) -> list[str]:
        return ["summary"]


def test_the_per_output_reduce_forwards_the_service_hooks() -> None:
    wrapped = _PerOutput(_Bound(), (0, 0, 1))
    bound = wrapped.bind_services({"svc": "http://h"})
    assert bound([10, 20, "path"]) == [10, 10, 20, "path", {"svc": "http://h"}]
    assert wrapped.resolve_services(1) == ("resolved", 1)
    assert wrapped.output_summaries() == ["summary"]
    plain = _PerOutput(_floats, (0, 0))
    assert plain.bind_services({}) is plain
    assert plain.resolve_services(1) == 1
    assert plain.output_summaries() == []
