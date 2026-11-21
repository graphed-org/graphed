"""Text describes a deferred value; a format spec, which asks for its value, is refused (#313)."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.errors import DeferredTypeError, GraphedTypeError
from graphed.numpy import NumpyBackend, from_array


def _numpy() -> Any:
    return from_array("x", np.ones((4, 3)), session=Session(NumpyBackend()))


def _awkward() -> Any:
    data = ak.Array({"w": [1.0, 2.0], "j": [[1.0], []]})
    return from_awkward("ev", data, session=Session(AwkwardBackend()))


SCALARS: dict[str, Callable[[], Any]] = {
    "numpy sum": lambda: np.sum(_numpy()),
    "awkward sum": lambda: gak.sum(_awkward().w),
    "awkward option max": lambda: gak.max(_awkward().w),
    "awkward count": lambda: gak.num(_awkward().w, axis=0),
    "varied sum": lambda: graphed.vary(s := gak.sum(_awkward().w), "sf", up=s * 2),
    "awkward record": lambda: _awkward()[0],
}
ARRAYS: dict[str, Callable[[], Any]] = {
    "numpy array": _numpy,
    "numpy reduced axis 0": lambda: np.sum(_numpy(), axis=0),
    "awkward array": lambda: _awkward().w,
    "awkward reduced axis 1": lambda: gak.sum(_awkward().j, axis=1),
    "varied array": lambda: graphed.vary(w := _awkward().w, "sf", up=w * 2),
}
VALUES = {**SCALARS, **ARRAYS}
TEXT: dict[str, Callable[[Any], str]] = {
    "str": str,
    "f-string": lambda v: f"{v}",
    "format": format,
    "percent-s": lambda v: "%s" % v,  # noqa: UP031
}
SPECS: dict[str, Callable[[Any], str]] = {
    "f-string .3f": lambda v: f"{v:.3f}",
    "format d": lambda v: format(v, "d"),
    "f-string >5": lambda v: f"{v:>5}",
}
NUMBERS: dict[str, Callable[[Any], object]] = {
    "float": float,
    "int": int,
    "bool": bool,
    "percent-d": lambda v: "%d" % v,  # noqa: UP031
    "percent-f": lambda v: "%f" % v,  # noqa: UP031
}


@pytest.mark.parametrize("text", TEXT.values(), ids=TEXT.keys())
@pytest.mark.parametrize("make", VALUES.values(), ids=VALUES.keys())
def test_text_without_a_spec_describes_the_deferred_value(
    make: Callable[[], Any], text: Callable[[Any], str]
) -> None:
    value = make()
    assert text(value) == repr(value)


@pytest.mark.parametrize("make", VALUES.values(), ids=VALUES.keys())
def test_print_describes_the_deferred_value(
    make: Callable[[], Any], capsys: pytest.CaptureFixture[str]
) -> None:
    value = make()
    print(value)
    assert capsys.readouterr().out == repr(value) + "\n"


@pytest.mark.parametrize("spec", SPECS.values(), ids=SPECS.keys())
@pytest.mark.parametrize("make", VALUES.values(), ids=VALUES.keys())
def test_a_format_spec_asks_for_the_value_and_is_refused(
    make: Callable[[], Any], spec: Callable[[Any], str]
) -> None:
    value = make()
    with pytest.raises(DeferredTypeError, match=r"needs values") as info:
        spec(value)
    assert "format spec" in str(info.value) and "graphed.materialize(x)" in str(info.value)


def test_the_refusal_points_at_the_users_line() -> None:
    s = gak.sum(_awkward().w)
    with pytest.raises(DeferredTypeError) as info:
        line = sys._getframe().f_lineno + 1
        print(f"{s:.3f}")
    assert info.value.provenance.lineno == line


@pytest.mark.parametrize("number", NUMBERS.values(), ids=NUMBERS.keys())
@pytest.mark.parametrize("make", SCALARS.values(), ids=SCALARS.keys())
def test_a_deferred_scalar_still_has_no_number(
    make: Callable[[], Any], number: Callable[[Any], object]
) -> None:
    with pytest.raises(TypeError if number is NUMBERS["percent-d"] else GraphedTypeError):
        number(make())


@pytest.mark.parametrize("make", SCALARS.values(), ids=SCALARS.keys())
def test_a_deferred_scalar_keeps_its_repr(make: Callable[[], Any]) -> None:
    assert repr(make()).startswith(("graphed.Array<", "Varied(labels="))


def test_a_deferred_slice_bound_is_named_in_the_axis0_refusal() -> None:
    x = _numpy()
    with pytest.raises(TypeError, match=r"chain it: a\[1:graphed\.Array<"):
        x[1 : np.sum(x), 0]
