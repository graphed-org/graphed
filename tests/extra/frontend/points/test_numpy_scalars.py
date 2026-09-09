"""A numpy scalar coordinate names the universe its Python equivalent does.

`points=` has two channels and they read a number through the same normalization: a DECLARING key
(`{np.int64(2): member}`) and a PLACEMENT coordinate (`{"jes": np.int64(2)}`). Before this fix only
the declaring channel went through `numbers`, so a placement refused `np.float32`/`np.int64` and
`np.float64` — a `float` subclass whose own `repr` is `'np.float64(2.0)'` — leaked a raw
`ValueError` out of `Fraction`. The equality legs are the point: each numpy case is asserted against
the *pure-Python* run of the identical program, so a change that moved both channels together would
still pass while one that moved only one fails.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import graphed
from graphed import Session
from graphed.context import EventContext
from graphed.errors import GraphedError
from graphed.numpy import NumpyBackend, from_record

VECTOR = np.arange(1.0, 13.0)

#: each numpy scalar beside the Python spelling it must reproduce
EQUIVALENT = [(np.float64(2.0), 2.0), (np.float32(2.0), 2.0), (np.int64(2), 2)]


def _joint(declare: Any, place: Any) -> tuple[str, dict[str, str]]:
    """A jes family declared with `declare`, and a jes-dependent weight placed at that coordinate.

    Returns the joint label and its registered point, so a caller compares the NAME a coordinate
    minted and the POINT it resolved to, not merely that the call survived.
    """
    session = Session(NumpyBackend())
    pt = from_record(session, "ev", pt=VECTOR)["pt"]
    ctx = EventContext(session, pt, collections={"pt": pt})
    shifted = graphed.vary(ctx, "jes", collections={"pt": {declare: pt * 1.2, "0p5": pt * 0.8}})
    spt = shifted["pt"]
    registered = graphed.vary(
        shifted,
        "muF",
        spt * 1.0,
        is_weight=True,
        points=[("2", spt * 1.1), {"muF": "2", "jes": place}],
    )
    weight = graphed.weight(registered)
    label = next(name for name in graphed.labels(weight) if "__" in name)
    return label, dict(graphed.varied.point_registry(weight)[label])


@pytest.mark.parametrize(("scalar", "python"), EQUIVALENT, ids=repr)
def test_a_numpy_placement_coordinate_matches_its_python_value(scalar: Any, python: Any) -> None:
    assert _joint("2", scalar) == _joint("2", python)


@pytest.mark.parametrize(("scalar", "python"), EQUIVALENT, ids=repr)
def test_a_numpy_declaring_key_matches_its_python_value(scalar: Any, python: Any) -> None:
    assert _joint(scalar, "2") == _joint(python, "2")


@pytest.mark.parametrize(("scalar", "python"), EQUIVALENT, ids=repr)
def test_the_two_channels_name_one_universe(scalar: Any, python: Any) -> None:
    """The declaring and placement roads meet: declaring with the numpy scalar and placing with the
    Python number resolves, which it could not if the two normalizations disagreed."""
    assert _joint(scalar, python) == ("muF_2__jes_2", {"jes": "2", "muF": "2"})


def test_a_numpy_bool_is_refused_like_a_python_bool() -> None:
    """`np.bool_` is registered with no numeric ABC, so it must refuse where `True` refuses —
    a coordinate is a tag or a number, and `True` is neither."""
    for channel in ({"jes": np.bool_(True)}, {"jes": True}):
        with pytest.raises(GraphedError, match="a coordinate is a variation tag or a number"):
            graphed._points.Point(channel)


def test_a_numpy_float_keeps_its_own_precision() -> None:
    """`np.float32(0.1)` is a different value from `0.1` and must not be silently read as one: the
    coordinate is that float32's widened decimal, which is what the declaring channel already did."""
    assert graphed._points.coordinate(np.float32(0.1)) == graphed._tags.canonical_tag(np.float32(0.1))
    assert graphed._points.coordinate(np.float32(0.1)) != graphed._points.coordinate(0.1)
