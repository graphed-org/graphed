"""A deferred awkward array answers ``.layout`` with its form's typetracer layout (#267)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, from_parquet, gnano
from graphed.systematics import universe

vector = pytest.importorskip("vector")
vector.register_awkward()

COMPONENTS = ("pt", "eta", "phi", "mass")
JETS = ak.zip(
    {
        "pt": [[30.0, 12.0], []],
        "eta": [[0.1, -1.2], []],
        "phi": [[0.5, 2.0], []],
        "mass": [[5.0, 1.0], []],
    }
)
EVENTS = ak.zip({"Jet": JETS, "run": [1, 2]}, depth_limit=1)


def _events() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    return session, gnano.events(from_awkward("ev", EVENTS, session=session), is_data=False)


def _zip(jets: Any) -> Any:
    return vector.zip({f: jets[f] for f in COMPONENTS})


def _value(out: Any) -> tuple[str, list[Any]]:
    value = ak.Array(out)
    return str(value.type), value.tolist()


def test_vector_awk_of_a_deferred_zip_equals_eager() -> None:
    session = Session(AwkwardBackend())
    src = from_awkward("e", JETS, session=session)
    got = vector.awk(_zip(src))
    want = vector.awk(_zip(JETS))
    assert _value(session.materialize(got)) == _value(want)
    assert _value(session.materialize(got.unit())) == _value(want.unit())
    assert _value(session.materialize(got.deltaphi(got))) == _value(want.deltaphi(want))


def _sources(tmp_path: Path) -> dict[str, Any]:
    session, events = _events()
    path = str(tmp_path / "j.parquet")
    ak.to_parquet(JETS, path)
    return {
        "from_awkward": from_awkward("e", JETS, session=session),
        "parquet": from_parquet("p", path, session=session),
        "an op": from_awkward("e", JETS, session=session).pt * 2,
        "event context": events,
        "context collection": events.Jet,
        "behavior record": _zip(events.Jet),
    }


@pytest.mark.parametrize(
    "name",
    ["from_awkward", "parquet", "an op", "event context", "context collection", "behavior record"],
)
def test_the_layout_has_the_eager_form_and_an_unknown_length(tmp_path: Path, name: str) -> None:
    x = _sources(tmp_path)[name]
    eager = {
        "from_awkward": JETS,
        "parquet": JETS,
        "an op": JETS.pt * 2,
        "event context": EVENTS,
        "context collection": EVENTS.Jet,
        "behavior record": _zip(EVENTS.Jet),
    }[name]
    assert x.layout.form == eager.layout.form
    assert x.layout.purelist_depth == eager.layout.purelist_depth
    assert str(x.layout.length) == "##"


def _leaf(layout: Any) -> Any:
    while not layout.is_numpy:
        layout = layout.content(layout.fields[0]) if layout.is_record else layout.content
    return layout


VALUE_READS: dict[str, Callable[[Any], Any]] = {
    "len": len,
    "to_list": lambda layout: layout.to_list(),
    "np.asarray": np.asarray,
    "ak.to_numpy": lambda layout: ak.to_numpy(_leaf(layout)),
    "iteration": list,
    "a leaf element as a number": lambda layout: float(_leaf(layout)[0]),
    "a leaf element as a bool": lambda layout: bool(_leaf(layout)[0]),
    "ak.Array(layout).tolist": lambda layout: ak.Array(layout).tolist(),
}


@pytest.mark.parametrize("read", VALUE_READS.values(), ids=VALUE_READS.keys())
@pytest.mark.parametrize("name", ["from_awkward", "parquet", "context collection"])
def test_no_value_reads_through_the_layout(tmp_path: Path, name: str, read: Any) -> None:
    layout = _sources(tmp_path)[name].layout
    with pytest.raises((TypeError, RuntimeError)):
        read(layout)


def test_a_leaf_element_reads_as_the_unknown_marker() -> None:
    session = Session(AwkwardBackend())
    assert str(_leaf(from_awkward("e", JETS, session=session).layout)[0]) == "##"


def test_a_varied_answers_the_layout_its_universes_share() -> None:
    session, events = _events()
    pt = graphed.vary(events.Jet.pt, "s", points={"up": events.Jet.pt * 2})
    assert pt.layout.form == EVENTS.Jet.pt.layout.form
    got = vector.awk(vector.zip({"pt": pt, **{f: events.Jet[f] for f in COMPONENTS[1:]}}))
    for label, scale in (("nominal", 1.0), ("s_up", 2.0)):
        jets = ak.with_field(EVENTS.Jet, EVENTS.Jet.pt * scale, "pt")
        assert _value(session.materialize(universe(got, label))) == _value(vector.awk(_zip(jets)))


def test_a_varied_whose_universes_differ_in_form_is_refused_by_name() -> None:
    _session, events = _events()
    pt = graphed.vary(
        events.Jet.pt, "s", points={"up": ak.values_astype(events.Jet.pt, np.float32)}
    )
    with pytest.raises(graphed.GraphedError, match="layouts differ in form"):
        _ = pt.layout


def test_a_varieds_field_named_layout_wins() -> None:
    session = Session(AwkwardBackend())
    arr = from_awkward("f", ak.Array({"layout": [1.0, 2.0]}), session=session)
    v = graphed.vary(arr, "s", points={"up": ak.with_field(arr, arr.layout * 2, "layout")})
    assert ak.Array(session.materialize(universe(v.layout, "s_up"))).tolist() == [2.0, 4.0]
