"""Awkward idioms HiggsDNA uses record what eager awkward computes: numpy dtype classes (GW22),
dict-form cartesian (GW24), non-ufunc numpy functions (GW26), ``hasattr`` on a missing field (GW28),
a Varied's form answers (GW20) and numpy-typed scalar operands (GW4)."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import GraphedError, GraphedTypeError, Session
from graphed.awkward import AwkwardBackend, from_awkward, gak

JAGGED = ak.Array([[1.0, np.nan, 2.5], [], [3.5, 0.25]])
FLAT = ak.Array([0.5, np.inf, 3.0, -1.0])
F32 = ak.values_astype(ak.Array([[1.0, 2.0], [], [3.5]]), np.float32)
I32 = ak.values_astype(ak.Array([[1, 2], [], [3]]), np.int32)

Op = Callable[[Any, Any], Any]  # (module, array) -> result, run once with gak and once with ak


def _check(op: Op, *eager: ak.Array) -> None:
    session = Session(AwkwardBackend())
    deferred = [from_awkward(session, f"a{i}", arr) for i, arr in enumerate(eager)]
    got = ak.Array(session.materialize(op(gak, *deferred)))
    want = op(ak, *eager)
    assert str(got.type) == str(want.type)
    assert ak.array_equal(got, want, equal_nan=True)


@pytest.mark.parametrize("dtype", [np.float32, int, np.dtype("int16")], ids=["class", "builtin", "dtype"])
@pytest.mark.parametrize(
    "op",
    [
        lambda m, x, dtype: m.values_astype(x, dtype),
        lambda m, x, dtype: m.zeros_like(x, dtype=dtype),
        lambda m, x, dtype: m.ones_like(x, dtype=dtype),
        lambda m, x, dtype: m.full_like(x, 2, dtype=dtype),
    ],
    ids=["values_astype", "zeros_like", "ones_like", "full_like"],
)
def test_a_dtype_given_as_a_type_is_read_as_numpy_reads_it(op: Callable[..., Any], dtype: Any) -> None:
    _check(lambda m, x: op(m, x, dtype), F32)


def test_ak_values_astype_on_a_deferred_array_takes_a_dtype_class() -> None:
    _check(lambda m, x: (ak if m is gak else m).values_astype(x, np.float64), F32)


@pytest.mark.parametrize(
    "op",
    [
        lambda m, x, y: m.cartesian({"a": x, "b": y}),
        lambda m, x, y: m.cartesian({"a": x, "b": y}, nested=["a"]),
        lambda m, x, y: m.cartesian({"a": x, "b": y}, nested=True),
        lambda m, x, y: m.argcartesian({"a": x, "b": y}),
        lambda m, x, y: m.cartesian([x, y], nested=[0]),
        lambda m, x, y: ak.cartesian({"a": x, "b": y}),  # ak's own dispatch onto gak
    ],
    ids=["dict", "nested_key", "nested_all", "argcartesian", "list_nested", "ak_dispatch"],
)
def test_cartesian_names_its_slots_from_a_dict(op: Callable[[Any, Any, Any], Any]) -> None:
    _check(lambda m, x: op(m, x, x * 2), JAGGED)


@pytest.mark.parametrize(
    ("op", "arr"),
    [
        (lambda m, x: np.clip(x, 0.6, 2.0), FLAT),
        (lambda m, x: np.clip(x, 0.6, None), FLAT),
        (lambda m, x: np.clip(x, a_min=None, a_max=2.0), FLAT),
        (lambda m, x: np.where(x > 1.5, x, -1.0), JAGGED),
        (lambda m, x: np.nan_to_num(x, nan=-5.0), JAGGED),
    ],
    ids=["clip", "clip_low", "clip_high_kw", "where", "nan_to_num"],
)
def test_a_numpy_function_on_a_deferred_array_records_as_awkward_answers_it(op: Op, arr: ak.Array) -> None:
    _check(op, arr)


def test_a_numpy_function_awkward_lacks_in_gak_is_a_located_refusal() -> None:
    session = Session(AwkwardBackend())
    with pytest.raises(GraphedError, match=r"no deferred ak\.round"):
        np.round(from_awkward(session, "a", JAGGED))


def _varied(up: Callable[[Any], Any]) -> tuple[Any, Any]:
    session = Session(AwkwardBackend())
    jets = from_awkward(session, "jets", ak.zip({"pt": JAGGED, "eta": JAGGED * 2}))
    return jets, graphed.vary(jets, "jes", up=up(jets), down=jets)


def test_a_numpy_function_maps_over_the_universes_of_a_varied() -> None:
    jets, varied = _varied(lambda j: gak.with_field(j, j.pt * 2, "pt"))
    clipped = np.clip(varied.pt, 1.0, 4.0)
    session = jets.session
    for label in graphed.labels(varied):
        want = np.clip(graphed.universe(varied, label).pt, 1.0, 4.0)
        got = session.materialize(graphed.universe(clipped, label))
        assert ak.array_equal(got, session.materialize(want), equal_nan=True)
    assert graphed.labels(clipped) == graphed.labels(varied)


def test_a_missing_field_reads_as_an_absent_attribute() -> None:
    session = Session(AwkwardBackend())
    events = from_awkward(session, "ev", ak.zip({"pt": JAGGED}))
    assert hasattr(events, "pt")
    assert not hasattr(events, "genWeight")
    assert getattr(events, "genWeight", None) is None
    with pytest.raises(GraphedTypeError, match="no field named 'genWeight'") as info:
        _ = events.genWeight
    assert isinstance(info.value, AttributeError)
    again = pickle.loads(pickle.dumps(info.value))
    assert isinstance(again, AttributeError) and str(again) == str(info.value)


def test_a_varied_answers_its_form_attributes_once() -> None:
    _jets, varied = _varied(lambda j: gak.with_field(j, j.pt * 2, "pt"))
    assert varied.fields == ["pt", "eta"]
    assert varied.ndim == 2
    assert not hasattr(varied, "genWeight")


def test_a_varied_whose_universes_disagree_on_a_form_attribute_refuses() -> None:
    jets, _ = _varied(lambda j: j)
    varied = graphed.vary(jets.pt, "prec", up=gak.values_astype(jets.pt, np.float32), down=jets.pt)
    assert varied.ndim == 2
    with pytest.raises(GraphedError, match="universes disagree on 'typestr'"):
        _ = varied.typestr


SCALARS = [np.float32(0.5), np.float64(0.5), 0.5, 1, np.int8(1), np.int64(1), True, np.bool_(True)]


@pytest.mark.parametrize("scalar", SCALARS, ids=lambda s: f"{type(s).__name__}")
@pytest.mark.parametrize("arr", [F32, I32], ids=["f32", "i32"])
@pytest.mark.parametrize(
    "op",
    [
        lambda m, x, s: m.where(x > 1, x, s),
        lambda m, x, s: m.where(x > 1, s, x),
        lambda m, x, s: m.fill_none(m.mask(x, x > 1), s),
        lambda m, x, s: m.full_like(x, s),
    ],
    ids=["where_right", "where_left", "fill_none", "full_like"],
)
def test_a_scalar_operand_keeps_its_numpy_dtype(op: Callable[..., Any], arr: ak.Array, scalar: Any) -> None:
    _check(lambda m, x: op(m, x, scalar), arr)


@pytest.mark.parametrize("initial", [5, np.int8(5), np.int64(5)], ids=lambda s: f"{type(s).__name__}")
@pytest.mark.parametrize("reducer", ["max", "min"])
def test_an_integer_initial_reduces_an_integer_array(reducer: str, initial: Any) -> None:
    _check(lambda m, x: getattr(m, reducer)(x, axis=1, initial=initial), I32)


def test_a_wide_integer_fill_value_is_not_rounded_through_float() -> None:
    _check(lambda m, x: m.full_like(x, np.int64(2**62 + 1)), I32)


def test_scalars_of_different_dtypes_record_different_nodes() -> None:
    session = Session(AwkwardBackend())
    x = from_awkward(session, "x", F32)
    masked = gak.mask(x, x > 1)
    assert gak.fill_none(masked, np.float32(0)).node_id != gak.fill_none(masked, 0.0).node_id
