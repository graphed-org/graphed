"""A declared External type is checked against the value at run time and never cast (owner ruling):
a disagreeing value raises `OutputTypeError` at the declaring line, in process and in a worker."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any

import awkward as ak
import m71_check_fixtures as fx
import numpy as np
import pytest

import graphed
from graphed import GraphedTypeError, Session
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward, gak
from graphed.core.execution import Plan, SequentialRunner
from graphed.debug import StageError
from graphed.debug import run as debug_run


def _events() -> tuple[Session, Any]:
    s = Session(AwkwardBackend())
    return s, from_awkward(s, "events", ak.Array({"x": fx.X}))


def _plan_events() -> tuple[Session, Any]:
    s = Session(AwkwardBackend())
    chunks = fx.Chunks()
    form = AwkwardForm(ak.Array(chunks.data.layout.to_typetracer(forget_length=True)))
    return s, s.source("events", form=form, data=chunks)


def _plan(selected: Any) -> Any:
    return graphed.aggregate_plan(gak.sum(selected), reduce=fx.total, combine=fx.add, empty=fx.zero)


def _refused(call: Any, line: int) -> GraphedTypeError:
    with pytest.raises(GraphedTypeError) as info:
        call()
    assert type(info.value).__name__ == "OutputTypeError"
    assert info.value.provenance.lineno == line
    return info.value


def test_a_map_value_of_another_type_raises_at_the_declaring_line() -> None:
    s, ev = _events()
    mask, line = ev.x.map(fx.score, output_type="bool"), fx.here()
    err = _refused(lambda: s.materialize(ev.x[mask]), line)
    assert f"node {mask.node_id} ('score') declares output_type 'bool'; its value is 'float64'" in err.detail


def test_an_apply_value_of_another_type_raises_at_the_declaring_line() -> None:
    s, ev = _events()
    both, line = graphed.apply(np.add, ev.x, ev.x, output_type="int64"), fx.here()
    err = _refused(lambda: s.materialize(both), line)
    assert "declares output_type 'int64'; its value is 'float64'" in err.detail


def test_a_matching_value_passes_unchanged() -> None:
    s, ev = _events()
    assert ak.to_list(s.materialize(ev.x[ev.x.map(fx.cut, output_type=bool)])) == [1.5, 2.5, 3.5]


def test_only_a_declared_node_carries_a_check() -> None:
    from graphed.session import CheckedExternal  # noqa: PLC0415  (absent on the base)

    s, ev = _events()
    assert s._externals[ev.x.map(fx.cut).node_id][0] is fx.cut
    checked = s._externals[ev.x.map(fx.cut, output_type=bool).node_id][0]
    assert isinstance(checked, CheckedExternal) and checked.fn is fx.cut


def test_a_plan_raises_a_stage_error_at_the_declaring_line() -> None:
    _s, ev = _plan_events()
    mask, line = ev.x.map(fx.score, output_type="bool"), fx.here()
    with pytest.raises(StageError) as info:
        SequentialRunner().run(_plan(ev.x[mask]))
    assert info.value.user_frame.lineno == line
    assert info.value.cause_type == "OutputTypeError"
    assert "declares output_type 'bool'; its value is 'float64'" in info.value.cause_message


def test_the_stage_error_crosses_a_process_pool() -> None:
    _s, ev = _plan_events()
    mask, line = ev.x.map(fx.score, output_type="bool"), fx.here()
    plan = _plan(ev.x[mask])
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        future = pool.submit(fx.run_task, plan.process, plan.tasks[0].partition)
        with pytest.raises(StageError) as info:
            future.result(timeout=120)
    assert info.value.user_frame.lineno == line
    assert info.value.cause_type == "OutputTypeError"


def test_the_debug_runner_raises_a_stage_error_at_the_declaring_line() -> None:
    s, ev = _events()
    mask, line = ev.x.map(fx.score, output_type="bool"), fx.here()
    with pytest.raises(StageError) as info:
        debug_run(s, ev.x[mask], opt_level=0)
    assert info.value.user_frame.lineno == line
    assert info.value.cause_type == "OutputTypeError"


def test_a_write_plan_raises_at_the_declaring_line(tmp_path: Any) -> None:
    pytest.importorskip("pyarrow")
    import graphed.awkward as ga  # noqa: PLC0415

    _s, ev = _events()
    mask, line = ev.x.map(fx.score, output_type="bool"), fx.here()
    plan = ga.to_parquet(ev.x[mask], str(tmp_path / "out"), compute=False)
    assert isinstance(plan, Plan)
    _refused(lambda: SequentialRunner().run(plan), line)


def _value(v: Any) -> Any:
    return lambda x: v


@pytest.mark.parametrize(
    ("declared", "value", "fits"),
    [
        ("var * float64", ak.Array([[], [], [], []]), True),  # unknown: no values to disagree
        ("float64", ak.Array([1.0, None, 2.0, 3.0]), False),  # option-ness is part of the type
        ("?float64", ak.Array([1.0, None, 2.0, 3.0]), True),
        ("var * float64", np.zeros((4, 2)), False),  # regular is not var
        ("2 * float64", np.zeros((4, 2)), True),
        ("{a: float64}", ak.zip({"a": fx.X}, with_name="Pt"), False),  # a record name is a parameter
        ("Pt[a: float64]", ak.zip({"a": fx.X}, with_name="Pt"), True),
        ("var * {a: float64, b: int64}", ak.Array([[], [], [], []]), True),
        ("{a: var * float64, b: int64}", ak.zip({"a": [[]] * 4, "b": [1] * 4}, depth_limit=1), True),
        ("{a: var * float64, b: int32}", ak.zip({"a": [[]] * 4, "b": [1] * 4}, depth_limit=1), False),
        ("{c: var * float64, b: int64}", ak.zip({"a": [[]] * 4, "b": [1] * 4}, depth_limit=1), False),
        ("float64", ak.Array([[], [], [], []]), False),  # unknown fits in place of a type, not a list
        ("2 * var * float64", ak.to_regular(ak.Array([[[]] * 3] * 4), axis=1), False),
        ("3 * var * float64", ak.to_regular(ak.Array([[[]] * 3] * 4), axis=1), True),
    ],
)
def test_what_matches_at_the_edges(declared: str, value: Any, fits: bool) -> None:
    s, ev = _events()
    out, line = ev.x.map(_value(value), name=f"edge {declared}", output_type=declared), fx.here()
    if fits:
        assert s.materialize(out) is value
    else:
        _refused(lambda: s.materialize(out), line)


def test_unknown_fits_only_inside_an_array() -> None:
    s, ev = _events()
    out, line = gak.sum(ev.x).map(_value(None), name="none", output_type="float64"), fx.here()
    assert "its value is 'unknown'" in _refused(lambda: s.materialize(out), line).detail


def test_a_value_awkward_cannot_type_is_refused() -> None:
    s, ev = _events()
    out, line = ev.x.map(_value(object()), name="opaque", output_type="float64"), fx.here()
    assert "its value is 'object'" in _refused(lambda: s.materialize(out), line).detail


def test_the_correction_template_checks_its_float64_leaves(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("correctionlib")
    from graphed.preserve.externals import _base  # noqa: PLC0415
    from graphed.preserve.externals.correctionlib_external import CORRECTIONLIB_PLUGIN  # noqa: PLC0415

    def float32(resource: Any, params: Any, inputs: list[Any]) -> Any:
        return ak.values_astype(inputs[0], "float32")

    monkeypatch.setitem(_base._REGISTRY, "correctionlib", replace(CORRECTIONLIB_PLUGIN, evaluate=float32))
    s, ev = _events()
    cset = CORRECTIONLIB_PLUGIN.samples()[0]
    sf, line = gak.apply_correction(cset, "sf", [ev.x], fx.score, args=["nominal", "$0"]), fx.here()
    err = _refused(lambda: s.materialize(sf), line)
    assert "declares output_dtype 'float64'; its value is 'float32'" in err.detail


@pytest.mark.parametrize(
    ("value", "fits"),
    [
        (ak.zip({"a": [[1.0], [], [2.0], []], "b": [0.5] * 4}, depth_limit=1), True),
        (ak.Array([[], [], [], []]), True),  # an unknown leaf holds no values
        (ak.zip({"a": [[1.0], [], [2.0], []], "b": [1] * 4}, depth_limit=1), False),
    ],
)
def test_an_output_dtype_is_checked_on_every_leaf(value: Any, fits: bool) -> None:
    s, ev = _events()
    fn, leaf = _value(value), {"output_dtype": "float64"}
    out, line = s.record_external("map", fn, [ev.x], {"fn": f"leaves {fits}"}, form_params=leaf), fx.here()
    if fits:
        assert s.materialize(out) is value
    else:
        assert "declares output_dtype 'float64'" in _refused(lambda: s.materialize(out), line).detail


def test_an_unrepresentable_output_dtype_beside_a_form_is_refused_at_its_line() -> None:
    s, ev = _events()
    desc, form = AwkwardBackend().external_payload("map", {"fn": "f"}), s.form(ev.x)
    line = fx.here() + 2  # the call's first line
    with pytest.raises(GraphedTypeError) as info:
        s.record_external(
            "map", fx.cut, [ev.x], {"fn": "f"}, descriptor=desc, form=form, form_params={"output_dtype": "x"}
        )
    assert info.value.provenance.lineno == line
