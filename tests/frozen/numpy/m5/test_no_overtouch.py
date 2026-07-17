"""Over-touch protection for the numpy backend (plan M5 / A.3): the projection must read EXACTLY the
necessary columns and NEVER more — the dask-awkward over-touching bug this milestone avoids."""

from __future__ import annotations

import numpy as np
import pytest

import graphed.numpy as gn
from graphed import Session

WIDE = {f"c{i}": np.arange(4, dtype=float) for i in range(20)}  # a 20-column record


def _events(s: Session) -> object:
    return gn.from_record(
        s,
        "events",
        pt=np.array([10.0, 40.0, 25.0, 5.0]),
        eta=np.array([0.1, -2.0, 1.5, 0.3]),
        phi=np.array([0.0, 1.0, 2.0, 3.0]),
        charge=np.array([1, -1, 1, -1]),
        mass=np.array([0.1, 0.1, 0.1, 0.1]),
        mask=np.array([True, True, False, True]),
    )


def test_single_field_reads_exactly_that_field() -> None:
    s = Session(gn.NumpyBackend())
    read = gn.project(_events(s)["pt"]).columns_for("events")
    assert read == frozenset({"pt"})
    assert not (read & {"eta", "phi", "charge", "mass", "mask"})  # no over-touch of siblings


def test_two_fields_read_exactly_two() -> None:
    s = Session(gn.NumpyBackend())
    ev = _events(s)
    read = gn.project(ev["pt"] + ev["eta"]).columns_for("events")
    assert read == frozenset({"pt", "eta"})
    assert "phi" not in read and "mass" not in read


def test_filter_does_not_overtouch_siblings() -> None:
    s = Session(gn.NumpyBackend())
    ev = _events(s)
    read = gn.project(ev["pt"].filter(ev["mask"])).columns_for("events")
    assert read == frozenset({"pt", "mask"})
    assert not (read & {"eta", "phi", "charge", "mass"})


def test_one_of_a_wide_record_reads_exactly_one() -> None:
    s = Session(gn.NumpyBackend())
    rec = gn.from_record(s, "wide", **WIDE)
    read = gn.project(rec["c7"].reduce("sum")).columns_for("wide")
    assert read == frozenset({"c7"})  # not the other 19 columns


def test_unused_source_is_never_read() -> None:
    s = Session(gn.NumpyBackend())
    a = gn.from_record(s, "a", x=np.array([1.0]), y=np.array([2.0]))
    gn.from_record(s, "b", p=np.array([3.0]), q=np.array([4.0]))  # entirely unused
    proj = gn.project(a["x"])
    assert proj.columns_for("a") == frozenset({"x"})
    assert proj.columns_for("b") == frozenset()  # source b is not read at all


def test_reduction_does_not_overtouch() -> None:
    s = Session(gn.NumpyBackend())
    ev = _events(s)
    read = gn.project((ev["pt"] * 2.0).reduce("sum")).columns_for("events")
    assert read == frozenset({"pt"})


@pytest.mark.parametrize(
    ("fields", "expected"),
    [(("pt",), {"pt"}), (("pt", "charge"), {"pt", "charge"}), (("mass",), {"mass"})],
)
def test_read_set_is_minimal(fields: tuple[str, ...], expected: set[str]) -> None:
    s = Session(gn.NumpyBackend())
    ev = _events(s)
    out = ev[fields[0]]
    for f in fields[1:]:
        out = out + ev[f]
    read = gn.project(out).columns_for("events")
    assert read == frozenset(expected)
    assert not (read - frozenset(expected)), f"over-touched: {sorted(read - frozenset(expected))}"
