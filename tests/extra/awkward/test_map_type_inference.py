"""#143: an undeclared map records the type its function gives a typetracer."""

from __future__ import annotations

import warnings
from typing import Any, cast

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import GraphedTypeError, GraphedWarning
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.execute import compile_ir
from graphed.numpy import NumpyBackend, NumpyForm, from_array
from graphed.session import Session


def events() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    data = ak.Array({"m": [[1.0, 2.0], [], [3.0]], "x": [1.0, 2.0, 3.0]})
    return session, from_awkward("ev", data, session=session)


def typestr(array: Any) -> str:
    return str(array.session.form(array).describe())


def count(values: Any) -> Any:
    return ak.num(values, axis=1)


def double(values: Any) -> Any:
    return values * 2


def raises_on_metadata(values: Any) -> Any:
    return ak.Array(values.to_list())


def test_the_recorded_type_is_the_functions_not_the_inputs() -> None:
    _, ev = events()
    n = ev.m.map(count)
    assert typestr(n) == "## * int64"
    assert ak.to_list(graphed.materialize(n)) == [2, 0, 1]


def test_an_axis_the_inferred_type_lacks_refuses_at_the_line() -> None:
    _, ev = events()
    with pytest.raises(GraphedTypeError, match="axis=1 exceeds"):
        gak.sum(ev.m.map(count), axis=1)


def test_a_multi_input_apply_is_traced_too() -> None:
    _, ev = events()
    assert typestr(graphed.apply(lambda m, x: ak.num(m, axis=1) * x, ev.m, ev.x)) == "## * float64"


def test_a_scalar_result_is_a_scalar_form() -> None:
    _, ev = events()
    assert typestr(ev.x.map(lambda values: ak.sum(values))) == "float64"


@pytest.mark.parametrize(
    "fn",
    [raises_on_metadata, lambda values: 1.0, lambda values: None, lambda values: {"a": values}],
    ids=["raises", "python", "none", "dict"],
)
def test_a_function_that_cannot_be_traced_warns_and_records_the_input_type(fn: Any) -> None:
    _, ev = events()
    with pytest.warns(
        GraphedWarning, match="could not infer the type of .*pass output_type="
    ) as w:
        out = ev.m.map(fn)
    assert typestr(out) == typestr(ev.m)
    assert w[0].filename == __file__


def test_a_declared_type_is_not_traced_and_stays_identity() -> None:
    _, ev = events()
    called: list[object] = []

    def spy(values: Any) -> Any:
        called.append(values)
        return ak.num(values, axis=1)

    declared = ev.m.map(spy, name="spy", output_type="int64")
    assert called == []
    assert typestr(declared) == "## * int64"
    assert declared.node_id != ev.m.map(spy, name="spy").node_id


def test_inference_changes_neither_identity_nor_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    def recorded() -> tuple[int, dict[str, Any], bytes]:
        session, ev = events()
        out = ev.m.map(count, name="f")
        return out.node_id, session._nodes()[out.node_id]["params"], compile_ir(session, out).ir

    traced = recorded()
    monkeypatch.setattr(AwkwardBackend, "traced_form", lambda _self, _fn, forms: forms[0])
    assert recorded() == traced
    assert traced[1] == {"fn": "f"}


def test_a_backend_that_cannot_trace_stays_silent_and_keeps_its_default_form() -> None:
    session = Session(NumpyBackend())
    x = from_array("x", np.array([1.0]), session=session)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GraphedWarning)
        out = x.map(double)
    assert cast("NumpyForm", session.form(out)).dtype == np.dtype(object)


def test_one_function_is_traced_once_per_input_type() -> None:
    _, ev = events()
    seen: list[str] = []

    def spy(values: Any) -> Any:
        seen.append(str(values.type))
        return values

    ev.x.map(spy, name="spy")
    ev.x.map(spy, name="spy")
    (ev.x * 2).map(spy, name="spy")
    assert seen == ["## * float64"]
    ev.m.map(spy, name="spy")
    assert seen == ["## * float64", "## * var * float64"]


def test_an_untraceable_function_warns_once_per_input_type() -> None:
    _, ev = events()
    with pytest.warns(GraphedWarning) as caught:
        ev.m.map(raises_on_metadata, name="f")
        (ev.m * 2).map(raises_on_metadata, name="f")
    assert len(caught) == 1


def test_a_leaf_dtype_is_a_declaration_and_is_not_traced() -> None:
    session, ev = events()
    called: list[object] = []

    def spy(values: Any) -> Any:
        called.append(values)
        return values

    params = {"fn": "leaf"}
    leaf = session.record_external(
        "map", spy, [ev.x], params, form_params={"output_dtype": "float32"}
    )
    assert called == []
    assert typestr(leaf) == "## * float32"


def test_a_map_without_a_name_param_is_refused_by_either_backend() -> None:
    session, ev = events()
    with pytest.raises(KeyError, match="fn"):
        session.record_external("map", double, [ev.x])
    numpy = Session(NumpyBackend())
    with pytest.raises(KeyError, match="fn"):
        numpy.record_external("map", double, [from_array("x", np.array([1.0]), session=numpy)])
