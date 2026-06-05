"""Field projection through complex topologies (plan M5) — diamond / star / nested shapes where a
record field feeds re-converging branches. Each field must be reported EXACTLY once and no sibling
field may be over-touched, however the branches fan out and re-join (a dask failure mode)."""

from __future__ import annotations

import numpy as np
from graphed import Session

import graphed_numpy as gn


def _events(s: Session) -> object:
    return gn.from_record(
        s,
        "events",
        pt=np.arange(1.0, 5.0),
        eta=np.linspace(0, 1, 4),
        phi=np.linspace(1, 2, 4),
        charge=np.array([1.0, -1.0, 1.0, -1.0]),
        mass=np.full(4, 0.1),
    )


def test_diamond_reads_each_field_exactly_once() -> None:
    s = Session(gn.NumpyBackend())
    ev = _events(s)
    a = ev["pt"] * 2.0  # the apex, reused by two branches
    out = (a + ev["eta"]) + (a - ev["phi"])
    read = gn.project(out).columns_for("events")
    assert read == frozenset({"pt", "eta", "phi"})  # not charge / mass


def test_star_one_field_feeds_many_branches() -> None:
    s = Session(gn.NumpyBackend())
    ev = _events(s)
    hub = ev["pt"]
    out = hub
    for field in ("eta", "phi", "charge", "mass"):
        out = out + (hub + ev[field])
    read = gn.project(out).columns_for("events")
    assert read == frozenset({"pt", "eta", "phi", "charge", "mass"})


def test_nested_diamonds_do_not_overtouch() -> None:
    s = Session(gn.NumpyBackend())
    ev = _events(s)
    v = ev["pt"]
    for _ in range(4):
        v = (v + ev["eta"]) - (v - ev["phi"])
    read = gn.project(v).columns_for("events")
    assert read == frozenset({"pt", "eta", "phi"})


def test_diamond_through_a_reduction_boundary() -> None:
    s = Session(gn.NumpyBackend())
    ev = _events(s)
    left = (ev["pt"] + ev["eta"]).reduce("sum")  # boundary in one branch
    right = ev["phi"].reduce("sum")
    out = left + right
    read = gn.project(out).columns_for("events")
    assert read == frozenset({"pt", "eta", "phi"})
