"""A value cut by the mask that derives a context is in that context's rows (graphed#280)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak, gnano
from graphed.systematics import labels, universe
from graphed.systematics.lineage import Followed

DATA = {
    "pho": [[{"pt": 40.0}], [], [{"pt": 55.0}, {"pt": 30.0}], []],
    "Jet": [[{"pt": 40.0}, {"pt": 20.0}], [{"pt": 55.0}], [{"pt": 35.0}], []],
}
EAGER = ak.Array(DATA)


def _events() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    return session, gnano.events(from_awkward("events", EAGER, session=session))


def _value(session: Session, x: Any) -> list[Any]:
    return list(ak.to_list(session.materialize(x)))


def _cut(events: Any) -> Any:
    return ak.num(events.pho) >= 1


def _cut_first(events: Any) -> Any:
    dipho = ak.firsts(events.pho)
    dipho, cut_events = dipho[_cut(events)], events[_cut(events)]
    cut_events["x"] = dipho.pt * 2
    return cut_events.x


def _context_first(events: Any) -> Any:
    cut_events = events[_cut(events)]
    cut_events["x"] = ak.firsts(events.pho)[_cut(events)].pt * 2
    return cut_events.x


def _same_mask_object(events: Any) -> Any:
    cut = _cut(events)
    cut_events = events[cut]
    cut_events["x"] = ak.firsts(events.pho)[cut].pt * 2
    return cut_events.x


SHAPES: dict[str, Callable[[Any], Any]] = {
    "the value is cut before the context": _cut_first,
    "the context is cut before the value": _context_first,
    "one mask object": _same_mask_object,
}


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_a_value_cut_by_the_contexts_mask_is_assigned_as_is(shape: Callable[[Any], Any]) -> None:
    session, events = _events()
    assert _value(session, shape(events)) == shape(ak.Array(DATA)).tolist() == [80.0, 110.0]


def test_a_value_cut_after_a_parent_assignment_combines_with_the_earlier_cut() -> None:
    session, events = _events()
    cut_events = events[_cut(events)]
    events["n"] = ak.num(events.Jet)
    dipho = ak.firsts(events.pho)[_cut(events)]
    cut_events["x"] = dipho.pt * ak.num(cut_events.pho)
    assert _value(session, cut_events.x) == [40.0, 110.0]


def test_a_second_cut_on_top_is_assigned_as_is() -> None:
    session, events = _events()
    cut = _cut(events)
    cut_events, dipho = events[cut], ak.firsts(events.pho)[cut]
    cut2 = ak.num(cut_events.pho) >= 2
    twice, dipho = cut_events[cut2], dipho[cut2]
    twice["x"] = dipho.pt
    assert _value(session, twice.x) == [55.0]


def test_a_mask_read_through_a_rederived_sibling_cuts_the_value_once() -> None:
    session, events = _events()
    cut = _cut(events)
    first = events[cut]
    events["n"] = ak.num(events.Jet)
    second = events[cut]
    cut2 = ak.num(first.pho) >= 2
    twice, dipho = second[cut2], ak.firsts(second.pho)[cut2]
    twice["x"] = dipho.pt
    assert _value(session, twice.x) == [55.0]


def test_a_flat_mask_cut_combines_with_a_cut_context() -> None:
    session, events = _events()
    flat = ak.flatten(events.Jet.pt)
    total = ak.sum(flat[flat > 30]) + ak.sum(ak.firsts(events[_cut(events)].Jet.pt))
    assert session.materialize(total) == 130.0 + 75.0


def test_a_jagged_mask_leaves_the_value_in_its_rows() -> None:
    session, events = _events()
    events["hard"] = events.Jet[events.Jet.pt > 30]
    assert _value(session, ak.num(events.hard)) == [1, 1, 1, 0]


def test_a_varied_value_and_a_varied_mask_are_assigned_per_universe() -> None:
    session, events = _events()
    jets = events.Jet
    events = graphed.vary(
        events, "jes", collections={"Jet": {"up": gak.with_field(jets, jets.pt * 2, "pt")}}
    )
    cut = ak.num(events.Jet[events.Jet.pt > 50]) >= 1
    lead = ak.firsts(events.Jet)[cut]
    cut_events = events[cut]
    cut_events["x"] = lead.pt
    got = {label: _value(session, universe(cut_events.x, label)) for label in labels(cut_events)}
    assert got == {"nominal": [55.0], "jes_up": [80.0, 110.0, 70.0]}


def test_a_derived_context_records_only_the_collections_it_reads() -> None:
    session, events = _events()
    events["a"], events["b"] = events.Jet, events.pho
    cut = _cut(events)
    before = session.node_count()
    cut_events = events[cut]
    record = session.node_count() - before
    assert _value(session, ak.num(cut_events.a)) == [2, 1]
    assert session.node_count() - before == record + 2  # `a`'s cut and its count; `b` stays unread
    assert list(cut_events.b.fields) == ["pt"]


def test_followed_collections_carry_once_and_keep_a_carry_error() -> None:
    def carry(value: Any) -> Any:
        if value == "bad":
            raise KeyError("inside the carry")
        return value + 1

    held = Followed({"a": 1, "b": "bad"}, carry)
    assert (held.get("a"), held.get("a"), held.get("zz", "none")) == (2, 2, "none")
    with pytest.raises(KeyError, match="inside the carry"):
        held.get("b")
    del held["a"]
    assert list(held) == ["b"]
