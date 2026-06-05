"""Necessary-buffer (column) projection for the numpy backend (plan M5)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from graphed import ProjectionError, Session

import graphed_numpy as gn


def _events(s: Session) -> object:
    return gn.from_record(
        s,
        "events",
        pt=np.array([10.0, 40.0, 25.0, 5.0]),
        eta=np.array([0.1, -2.0, 1.5, 0.3]),
        phi=np.array([0.0, 1.0, 2.0, 3.0]),
        charge=np.array([1, -1, 1, -1]),
        mask=np.array([True, True, False, True]),
    )


def test_record_source_projects_to_touched_fields_only() -> None:
    s = Session(gn.NumpyBackend())
    events = _events(s)
    out = (events["pt"] * 2.0 + events["eta"]).reduce("sum")
    proj = gn.project(out)
    assert proj.columns_for("events") == frozenset({"pt", "eta"})
    assert proj.total_columns() == 2  # not the whole 5-column record


def test_projection_through_filter_and_field() -> None:
    s = Session(gn.NumpyBackend())
    events = _events(s)
    selected = events["pt"].filter(events["mask"])
    proj = gn.project(selected)
    assert proj.columns_for("events") == frozenset({"pt", "mask"})


def test_projection_correctness_against_materialize() -> None:
    s = Session(gn.NumpyBackend())
    events = _events(s)
    out = (events["pt"] * 2.0 + events["eta"]).reduce("sum")
    assert np.isclose(float(s.materialize(out)), 2 * (10 + 40 + 25 + 5) + (0.1 - 2 + 1.5 + 0.3))


def test_flat_source_is_whole_buffer() -> None:
    s = Session(gn.NumpyBackend())
    a = gn.from_array(s, "a", np.array([1.0, 2.0, 3.0]))
    proj = gn.project(a.reduce("sum"))
    assert proj.columns_for("a") == frozenset({"a"})


def test_unread_source_is_not_projected() -> None:
    s = Session(gn.NumpyBackend())
    events = _events(s)
    proj = gn.project(events["pt"])
    assert proj.columns_for("events") == frozenset({"pt"})  # eta/phi/charge/mask not read


def test_field_access_on_non_record_is_ill_typed() -> None:
    s = Session(gn.NumpyBackend())
    flat = gn.from_array(s, "a", np.array([1.0, 2.0]))
    with pytest.raises(Exception):  # noqa: B017 - GraphedTypeError wraps the backend error
        flat["pt"]


@pytest.mark.parametrize("policy", ["pass", "warn", "raise"])
def test_on_fail_policy_on_opaque_op(policy: str) -> None:
    s = Session(gn.NumpyBackend())
    events = _events(s)
    opaque = events["pt"].map(lambda a: a, name="blackbox")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if policy == "raise":
            with pytest.raises(ProjectionError):
                gn.project(opaque, on_fail="raise")
            return
        proj = gn.project(opaque, on_fail=policy)
    if policy == "warn":
        assert len(w) == 1
        assert proj.columns_for("events") == frozenset({"pt", "eta", "phi", "charge", "mask"})  # conservative
    else:  # pass
        assert len(w) == 0
        assert proj.columns_for("events") == frozenset({"pt"})  # best-effort: only what was passed
