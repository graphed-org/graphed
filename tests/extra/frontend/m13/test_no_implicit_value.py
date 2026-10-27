"""A deferred value has no truth value and no eager numpy value.

Asking for either raises at the user's line instead of answering `True` or a 0-d object array.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import GraphedTypeError, Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.numpy import NumpyBackend, from_array


def _numpy() -> Any:
    return from_array(Session(NumpyBackend()), "x", np.array([1.0, 2.0]))


def _awkward() -> Any:
    return from_awkward(Session(AwkwardBackend()), "x", ak.Array([[1.0], [2.0, 3.0]]))


def _varied(make: Callable[[], Any]) -> Callable[[], Any]:
    def build() -> Any:
        x = make()
        return graphed.vary(x, "sf", up=x * 2, down=x / 2)

    return build


MAKERS = {
    "numpy": _numpy,
    "awkward": _awkward,
    "numpy-varied": _varied(_numpy),
    "awkward-varied": _varied(_awkward),
}


def _session(value: Any) -> Session:
    array: graphed.Array = graphed.nominal(value) if isinstance(value, graphed.Varied) else value
    return array.session


@pytest.mark.parametrize("make", MAKERS.values(), ids=MAKERS.keys())
def test_if_on_a_deferred_value_raises_at_the_users_line(make: Callable[[], Any]) -> None:
    value = make()
    count = _session(value).node_count()
    with pytest.raises(GraphedTypeError, match=r"'bool\(\.\.\.\)'.*no truth value") as info:
        line = sys._getframe().f_lineno + 1
        if value:
            pass
    assert info.value.provenance.lineno == line
    assert "session.materialize" in str(info.value)
    assert _session(value).node_count() == count


@pytest.mark.parametrize("convert", [np.asarray, np.array], ids=["asarray", "array"])
@pytest.mark.parametrize("make", MAKERS.values(), ids=MAKERS.keys())
def test_numpy_conversion_of_a_deferred_value_raises(
    make: Callable[[], Any], convert: Callable[[Any], Any]
) -> None:
    value = make()
    with pytest.raises(GraphedTypeError, match=r"'np\.asarray\(\.\.\.\)'.*no numpy value") as info:
        line = sys._getframe().f_lineno + 1
        convert(value)
    assert info.value.provenance.lineno == line


def test_np_where_on_an_awkward_deferred_condition_selects_elementwise() -> None:
    x = _awkward()
    picked = np.where(x > 1.5, 1, 0)
    assert isinstance(picked, graphed.Array)
    assert x.session.materialize(picked).tolist() == [[0], [1, 1]]


def test_numpy_dispatch_and_deferred_logic_still_record() -> None:
    x = _numpy()
    cond = False & (x > 1.5)
    assert isinstance(cond, graphed.Array)
    picked = np.where(cond, x, 0.0)
    assert isinstance(picked, graphed.Array)
    assert isinstance(np.float64(2.0) * x, graphed.Array)
    assert x.session.materialize(picked).tolist() == [0.0, 0.0]
