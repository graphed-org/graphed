"""#245: an untraceable map's type comes from one call on length-zero arrays, and is checked."""

from __future__ import annotations

import sys
import warnings
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import GraphedWarning
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.debug import StageError
from graphed.session import Session


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


@pytest.mark.parametrize(
    ("fn", "inferred", "executed"),
    [(_widens, "float64", "float32"), (_lies, "float64", "bool")],
    ids=["length-zero", "typetracer"],
)
def test_an_inferred_type_the_value_disagrees_with_raises_at_the_line(
    fn: Any, inferred: str, executed: str
) -> None:
    out, line = events().x.map(fn), sys._getframe().f_lineno
    with pytest.raises(StageError) as info:
        graphed.materialize(out)
    assert info.value.cause_type == "OutputTypeError"
    assert (
        f"declares output_type {inferred!r}; its value is {executed!r}" in info.value.cause_message
    )
    assert info.value.user_frame.lineno == line


def test_a_traced_reduction_to_a_scalar_keeps_its_form_unchecked() -> None:
    from graphed.session import CheckedExternal  # noqa: PLC0415

    ev = events()
    out = ev.x.map(lambda x: ak.sum(x))
    assert typestr(out) == "float64"
    assert not isinstance(ev.session._externals[out.node_id][0], CheckedExternal)
    assert graphed.materialize(out) == 14.0


def test_a_field_named_unknown_is_a_known_type() -> None:
    ev = events()
    with warnings.catch_warnings():
        warnings.simplefilter("error", GraphedWarning)
        out = ev.x.map(lambda x: ak.zip({"unknown": ak.Array(np.sqrt(ak.to_numpy(x)))}))
    assert typestr(out) == "## * {unknown: float64}"


def test_a_backend_that_cannot_declare_still_records_the_traced_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from graphed.session import CheckedExternal  # noqa: PLC0415

    monkeypatch.delattr(AwkwardBackend, "canonical_output_type")
    ev = events()
    out = ev.x.map(lambda x: x > 2)
    assert typestr(out) == "## * bool"
    assert not isinstance(ev.session._externals[out.node_id][0], CheckedExternal)
