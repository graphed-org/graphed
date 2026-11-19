"""A scalar among a gak function's array operands broadcasts as eager awkward does (#262)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed.awkward import AwkwardBackend, from_awkward, gak, gnano
from graphed.errors import GraphedTypeError
from graphed.systematics import universe

EAGER = ak.Array(
    {"x": [[1.0, 2.0], [], [3.0]], "y": [[0.5, 1.5], [], [2.5]], "f": [1.0, 2.0, 3.0]}
)

ACCEPTED: dict[str, Callable[[Any], Any]] = {
    "zip mapping, scalar last": lambda s: ak.zip({"x": s.x, "m": 0}),
    "zip mapping, scalar first": lambda s: ak.zip({"m": 0.5, "x": s.x}),
    "zip sequence": lambda s: ak.zip([s.x, 0.5]),
    "zip numpy scalar keeps its dtype": lambda s: ak.zip({"x": s.x, "m": np.int8(1)}),
    "zip two scalars": lambda s: ak.zip({"a": 1, "x": s.x, "b": np.float32(2)}),
    "concatenate axis=1, scalar last": lambda s: ak.concatenate([s.x, 0], axis=1),
    "concatenate axis=1, scalar first": lambda s: ak.concatenate([0, s.x], axis=1),
    "cartesian sequence": lambda s: ak.cartesian([s.x, 0]),
    "cartesian mapping": lambda s: ak.cartesian({"a": s.x, "b": 0}),
    "broadcast_arrays scalar out": lambda s: ak.broadcast_arrays(s.x, 0)[1],
    "broadcast_arrays array out": lambda s: ak.broadcast_arrays(0, s.x)[1],
    "where": lambda s: ak.where(s.f > 1, s.f, 0),
    "with_field": lambda s: ak.with_field(s, 0, "m").m,
}


def _value(out: Any) -> tuple[str, list[Any]]:
    value = ak.Array(out)
    return str(value.type), value.tolist()


@pytest.mark.parametrize("call", ACCEPTED.values(), ids=ACCEPTED.keys())
def test_a_scalar_operand_broadcasts_as_eager(call: Callable[[Any], Any]) -> None:
    src = from_awkward("e", EAGER, session=graphed.new_session())
    assert _value(graphed.materialize(call(src))) == _value(call(EAGER))


REFUSED: dict[str, tuple[Callable[[Any], Any], str]] = {
    "corr": (lambda s: ak.corr(s.x, 1.0, axis=1), "y must be a deferred array, not float"),
    "covar": (lambda s: ak.covar(1.0, s.x, axis=1), "x must be a deferred array, not float"),
    "linear_fit": (lambda s: ak.linear_fit(s.x, 2, axis=1), "y must be a deferred array, not int"),
    "mask": (lambda s: ak.mask(s.f, True), "condition must be a deferred array, not bool"),
    "concatenate axis=0": (lambda s: ak.concatenate([s.f, 0]), "a scalar adds a row"),
    "zip an eager object": (lambda s: ak.zip({"x": s.x, "o": object()}), "operand 1 must be"),
}


@pytest.mark.parametrize(("call", "why"), REFUSED.values(), ids=REFUSED.keys())
def test_an_operand_eager_refuses_or_chunks_cannot_hold_is_refused_by_name(
    call: Callable[[Any], Any], why: str
) -> None:
    src = from_awkward("e", EAGER, session=graphed.new_session())
    with pytest.raises(GraphedTypeError, match=why):
        call(src)


def test_an_eager_array_operand_is_still_refused_as_eager() -> None:
    src = from_awkward("e", EAGER, session=graphed.new_session())
    with pytest.raises(
        graphed.GraphedError, match="cannot combine a deferred array with an eager"
    ):
        ak.zip({"x": src.f, "w": np.ones(3)})


def test_no_deferred_operand_is_refused() -> None:
    with pytest.raises(GraphedTypeError, match="no operand is a deferred array"):
        gak.zip({"a": 0, "b": 1})


def test_a_varied_field_zips_with_a_constant_mass_per_universe() -> None:
    session = graphed.Session(AwkwardBackend())
    events = gnano.events(from_awkward("ev", EAGER, session=session), is_data=False)
    pt = graphed.vary(events.f, "s", points={"up": events.f * 2})
    out = ak.zip({"pt": pt, "mass": 0})
    for label, scale in (("nominal", 1.0), ("s_up", 2.0)):
        got = ak.Array(session.materialize(universe(out, label)))
        assert got.tolist() == ak.zip({"pt": EAGER.f * scale, "mass": 0}).tolist()
