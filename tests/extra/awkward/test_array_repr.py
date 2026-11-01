"""A deferred array's repr and ``dir``, and a Session's repr."""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed.numpy as gnp
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak

vector = pytest.importorskip("vector")
vector.register_awkward()


def _events(name: str = "Events") -> Any:
    session = Session(AwkwardBackend(behavior=ak.behavior))
    data = ak.Array({"Jet_pt": [[1.0, 2.0], []], "Muon_pt": [[3.0], [4.0]], "MET_pt": [1.0, 2.0]})
    return from_awkward(name, data, session=session)


def test_repr_names_the_type_and_the_source_of_a_read_and_a_derived_array() -> None:
    ev = _events()
    assert repr(ev.MET_pt) == f"graphed.Array<{ev.MET_pt.typestr}, from 'Events'>"
    derived = gak.num(ev.Jet_pt, axis=1) * 2
    assert repr(derived) == f"graphed.Array<{derived.typestr}, from 'Events'>"


def test_repr_names_every_source_read_once() -> None:
    ev = _events()
    other = from_awkward("Other", ak.Array({"w": [1.0, 2.0]}), session=ev.session)
    assert repr(ev.MET_pt * other.w).endswith("from 'Events', 'Other'>")


def test_repr_cuts_a_long_type_and_never_the_source() -> None:
    session = Session(AwkwardBackend())
    data = ak.Array({f"branch_{i}": [1.0] for i in range(40)})
    text = repr(from_awkward("Events", data, session=session))
    assert "..., from 'Events'>" in text
    assert len(text) < 160


def test_repr_and_dir_read_nothing_and_record_nothing() -> None:
    ev = _events()
    x = ev.Jet_pt
    count = ev.session.node_count()
    repr(x)
    dir(ev)
    assert ev.session.node_count() == count


@pytest.mark.parametrize(
    ("make", "text"),
    [
        (lambda ev: gak.sum(ev.MET_pt), "float64"),
        (lambda ev: gak.max(ev.MET_pt, axis=0), "?float64"),
        (lambda ev: gak.argmax(ev.MET_pt), "?int64"),
    ],
)
def test_a_scalar_has_a_repr_and_a_dir(make: Any, text: str) -> None:
    scalar = make(_events())
    assert repr(scalar) == f"graphed.Array<{text}, from 'Events'>"
    assert "map" in dir(scalar)
    assert "ndim" not in dir(scalar)


def test_a_behavior_property_returning_a_plain_value_keeps_a_repr_and_a_dir() -> None:
    class Constant(ak.Array):  # type: ignore[misc]
        @property
        def constant(self) -> float:
            return 1.5

    behavior: Any = {("*", "C"): Constant}
    session = Session(AwkwardBackend(behavior=behavior))
    ev = from_awkward("x", ak.Array({"x": [1.0]}), session=session)
    value = gak.zip({"x": ev.x}, with_name="C").constant
    assert repr(value) == "graphed.Array<float, from 'x'>"
    assert "map" in dir(value)


def test_a_bound_methods_repr_names_the_receivers_node() -> None:
    pt = _events().Jet_pt
    mu = gak.zip({"pt": pt, "eta": pt, "phi": pt, "mass": pt}, with_name="Momentum4D")
    assert repr(mu.deltaR) == f"BoundMethod('deltaR' of Array(node_id={mu.node_id}))"


def test_dir_lists_only_names_attribute_access_reaches() -> None:
    session = Session(AwkwardBackend())
    data = ak.Array({"_private": [1.0], "not an identifier": [1.0], "kept": [1.0]})
    names = dir(from_awkward("x", data, session=session))
    assert "kept" in names
    assert not {"_private", "not an identifier"} & set(names)


def test_dir_lists_fields_and_keeps_the_frontend_names() -> None:
    names = dir(_events())
    assert {"Jet_pt", "MET_pt", "filter", "map", "session", "typestr"} <= set(names)
    assert names == sorted(names)


def test_dir_leaves_out_what_a_deferred_array_cannot_answer() -> None:
    assert not {"to_list", "show", "layout"} & set(dir(_events()))


def test_dir_lists_a_behaviors_properties_and_methods() -> None:
    ev = _events()
    pt = ev.Jet_pt
    mu = gak.zip({"pt": pt, "eta": pt, "phi": pt, "mass": pt}, with_name="Momentum4D")
    assert {"pt", "mass", "rho", "deltaR"} <= set(dir(mu))
    assert "rho" not in dir(pt)


def test_a_field_named_like_an_eager_method_is_listed() -> None:
    session = Session(AwkwardBackend())
    ev = from_awkward("x", ak.Array({"to_list": [1.0]}), session=session)
    assert "to_list" in dir(ev)


def test_a_numpy_array_gets_a_repr_and_a_records_fields() -> None:
    session = Session(gnp.NumpyBackend())
    x = gnp.from_array("y", np.arange(3.0), session=session)
    assert repr(x * 2) == "graphed.Array<vector[float64], from 'y'>"
    assert "shape" in dir(x)
    record = gnp.from_record("r", session=session, a=np.arange(3.0))
    assert "a" in dir(record)


def test_a_backend_naming_no_attributes_keeps_the_frontend_names() -> None:
    class Plain(gnp.NumpyBackend):
        attribute_names = None  # type: ignore[assignment]

    x = gnp.from_array("y", np.arange(3.0), session=Session(Plain()))
    assert "map" in dir(x)


def test_session_repr_counts_nodes_and_sources() -> None:
    ev = _events()
    ev.session.form(ev.MET_pt * 2)
    assert repr(ev.session) == "<graphed.Session: 3 nodes, 1 sources, AwkwardBackend>"
