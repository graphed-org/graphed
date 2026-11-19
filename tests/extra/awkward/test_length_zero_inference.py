"""#245: an untraceable map's type comes from one call on length-zero arrays, and is checked."""

from __future__ import annotations

import sys
import warnings
from functools import partial
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import GraphedWarning
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.debug import StageError
from graphed.session import CheckedExternal, Session


def events() -> Any:
    session = Session(AwkwardBackend())
    return from_awkward("ev", ak.Array({"x": [1.0, 4.0, 9.0]}), session=session)


def typestr(array: Any) -> str:
    return str(array.session.form(array).describe())


def test_a_numpy_bound_function_is_typed_without_a_warning() -> None:
    ev = events()
    with warnings.catch_warnings():
        warnings.simplefilter("error", GraphedWarning)
        out = ev.x.map(lambda x: ak.Array(np.sqrt(ak.to_numpy(x))))
    assert typestr(out) == "## * float64"
    assert ak.to_list(graphed.materialize(out)) == [1.0, 2.0, 3.0]


def test_a_function_that_fails_on_empty_input_warns_naming_both_failures() -> None:
    ev = events()
    with pytest.warns(GraphedWarning, match="pass output_type=") as caught:
        out = ev.x.map(lambda x: ak.Array([ak.to_numpy(x)[0]] * len(x)))
    message = str(caught[0].message)
    assert "IndexError: index 0 is out of bounds" in message
    assert message.index("TypeError") < message.index("IndexError")
    assert typestr(out) == typestr(ev.x)


def test_an_unknown_typed_empty_result_warns() -> None:
    def guarded(x: Any) -> Any:
        values = ak.to_numpy(x)
        return ak.Array([]) if len(values) == 0 else ak.Array(values * 2)

    with pytest.warns(GraphedWarning, match=r"its type is '0 \* unknown'.*pass output_type="):
        events().x.map(guarded)


def test_a_traceable_function_is_called_once_on_its_typetracer() -> None:
    backends: list[str] = []

    def double(x: Any) -> Any:
        backends.append(ak.backend(x))
        return x * 2

    assert typestr(events().x.map(double)) == "## * float64"
    assert backends == ["typetracer"]


def _widens(x: Any) -> Any:
    values = ak.to_numpy(x)
    return ak.Array(values.astype(np.float32) if len(values) else values)


def _lies(x: Any) -> Any:
    return x if ak.backend(x) == "typetracer" else x > 2


def _sum_lies(x: Any) -> Any:
    return ak.sum(x) if ak.backend(x) == "typetracer" else ak.sum(ak.values_astype(x, np.int64))


def _pair(total: Any, *, kind: str) -> Any:
    if ak.typetracer.is_unknown_scalar(total):
        return ak.Array(ak.Array([1.0, 2.0]).layout.to_typetracer(forget_length=True))
    return ak.Array([1.0, 2.0] if kind == "float" else [1, 2])


@pytest.mark.parametrize(
    ("fn", "on_total", "message"),
    [
        (_widens, False, "declares array output_type 'float64'; its value is 'float32'"),
        (_lies, False, "declares array output_type 'float64'; its value is 'bool'"),
        (_sum_lies, False, "declares scalar output_type 'float64'; its value is 'int64'"),
        (
            partial(_pair, kind="int"),
            True,
            "declares array output_type 'float64'; its value is 'int64'",
        ),
    ],
    ids=["length-zero", "typetracer", "scalar-from-arrays", "array-from-scalars"],
)
def test_an_inferred_type_the_value_disagrees_with_raises_at_the_line(
    fn: Any, on_total: bool, message: str
) -> None:
    ev = events()
    operand = gak.sum(ev.x) if on_total else ev.x
    out, line = graphed.apply(fn, operand), sys._getframe().f_lineno
    with pytest.raises(StageError) as info:
        graphed.materialize(out)
    assert info.value.cause_type == "OutputTypeError"
    assert message in info.value.cause_message
    assert info.value.user_frame.lineno == line


def test_an_inferred_row_ness_other_than_the_inputs_is_checked_and_passes() -> None:
    ev = events()
    total = ev.x.map(lambda x: ak.sum(x))
    pair = graphed.apply(partial(_pair, kind="float"), gak.sum(ev.x))
    assert (typestr(total), typestr(pair)) == ("float64", "## * float64")
    keys = [ev.session._externals[out.node_id][0].key for out in (total, pair)]
    assert keys == ["scalar output_type", "array output_type"]
    assert graphed.materialize(total) == 14.0
    assert ak.to_list(graphed.materialize(pair)) == [1.0, 2.0]


def test_a_declared_scalar_from_arrays_still_refuses() -> None:
    out = events().x.map(lambda x: ak.sum(x), output_type="float64")
    with pytest.raises(StageError) as info:
        graphed.materialize(out)
    assert (
        "declares output_type 'float64'; its value is 'scalar float64'" in info.value.cause_message
    )


def test_a_field_named_unknown_is_a_known_type() -> None:
    ev = events()
    with warnings.catch_warnings():
        warnings.simplefilter("error", GraphedWarning)
        out = ev.x.map(lambda x: ak.zip({"unknown": ak.Array(np.sqrt(ak.to_numpy(x)))}))
    assert typestr(out) == "## * {unknown: float64}"


def test_a_backend_that_cannot_declare_still_records_the_traced_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(AwkwardBackend, "canonical_output_type")
    ev = events()
    out = ev.x.map(lambda x: x > 2)
    assert typestr(out) == "## * bool"
    assert not isinstance(ev.session._externals[out.node_id][0], CheckedExternal)


@pytest.mark.parametrize(
    ("fn", "reason"),
    [
        (lambda total: 1.0, r"<lambda>; pass"),
        (lambda total: _raise(KeyError("k")), r"\(KeyError: 'k'\)"),
    ],
    ids=["not-traced", "raises"],
)
def test_a_scalar_input_has_no_length_zero_call(fn: Any, reason: str) -> None:
    with pytest.warns(GraphedWarning, match=reason):
        graphed.apply(fn, gak.sum(events().x))


def _raise(exc: Exception) -> Any:
    raise exc
