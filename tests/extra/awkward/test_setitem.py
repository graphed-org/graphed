"""ak's field assignment on a deferred awkward array, ``events["Photon"] = photons`` (GW7)."""

from __future__ import annotations

import copy
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.context import members_of
from graphed.errors import GraphedTypeError
from graphed.systematics import labels

DATA = {"m": [1.0, 2.0], "r": [{"y": 1, "z": 5}, {"y": 2, "z": 6}]}
DATA_ROWS = ak.to_list(ak.Array(DATA))


def _source() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    return session, from_awkward(session, "ev", ak.Array(DATA))


@pytest.mark.parametrize("key", ["w", "m", ("r", "q"), ("r", "y"), ("r",)])
def test_assignment_records_what_eager_awkward_computes(key: Any) -> None:
    session, ev = _source()
    ev[key] = ev.m * 2
    eager = ak.Array(DATA)
    eager[key] = eager.m * 2
    got: Any = session.materialize(ev)
    assert (str(got.type), ak.to_list(got)) == (str(eager.type), ak.to_list(eager))


def test_other_proxies_of_the_node_are_untouched() -> None:
    session, ev = _source()
    twin = type(ev)(session, ev.node_id)
    ev["w"] = ev.m
    assert ev.node_id != twin.node_id
    assert ak.to_list(session.materialize(twin)) == ak.to_list(ak.Array(DATA))


def test_assignment_on_a_varied_container_applies_to_every_universe() -> None:
    session, ev = _source()
    varied = gak.with_field(ev, graphed.vary(ev.m, "s", points={"up": ev.m * 3}), "m")
    varied["w"] = varied.m + 1
    for label, m in (("nominal", [1.0, 2.0]), ("s_up", [3.0, 6.0])):
        got: Any = session.materialize(graphed.member_of(varied, label))
        assert got.w.tolist() == [x + 1 for x in m], label


def _slots(x: Any) -> dict[str, Any]:
    def plain(v: Any) -> Any:
        if isinstance(v, graphed.Array):
            return v.node_id
        return {k: plain(m) for k, m in v.items()} if isinstance(v, dict) else v

    slots = [n for c in type(x).__mro__ for n in getattr(c, "__slots__", ())]
    names = [n for n in slots if n not in ("__weakref__", "_site") and hasattr(x, n)]
    return {n: plain(getattr(x, n)) for n in names}


@pytest.mark.parametrize("member", [False, True], ids=["varied", "member_of"])
def test_assignment_rebinds_every_slot_of_the_recorded_result(member: bool) -> None:
    _, ev = _source()
    x: Any = graphed.vary(ev, "s", points={"up": gak.with_field(ev, ev.m * 2, "m")})
    value: Any = graphed.vary(ev.m, "t", points={"dn": ev.m * 3})
    if member:
        x, value = graphed.member_of(x, "s_up"), graphed.member_of(value, "t_dn")
    before, expected = _slots(x), _slots(gak.with_field(x, value, "w"))
    x["w"] = value
    assert _slots(x) == expected
    assert {n for n in expected if expected[n] != before[n]} - {"_node_id", "_members"}


@pytest.mark.parametrize("key", [0, ["m"], (), ("r", 1)])
def test_a_non_field_key_is_refused_as_ak_refuses_it(key: Any) -> None:
    _, ev = _source()
    with pytest.raises(GraphedTypeError, match="only fields may be assigned"):
        ev[key] = ev.m


@pytest.mark.parametrize("value", [None, 1j, np.datetime64("2020-01-01")], ids=repr)
def test_a_non_scalar_value_is_refused_naming_with_field(value: Any) -> None:
    _, ev = _source()
    with pytest.raises(GraphedTypeError, match=r"with_field.*bool/int/float/str scalar"):
        ev["w"] = value


@pytest.mark.parametrize("value", [np.array([1.0]), [1.0]], ids=repr)
def test_an_eager_array_value_is_refused_naming_the_input_fix(value: Any) -> None:
    _, ev = _source()
    with pytest.raises(GraphedTypeError, match=r"from_awkward.*graphed\.materialize"):
        ev["w"] = value


def _universes(session: Session, x: Any) -> dict[str, Any]:
    return {label: ak.to_list(session.materialize(m)) for label, m in members_of(x).items()}


@pytest.mark.parametrize("key", ["w", ("r", "q")], ids=["field", "path"])
def test_a_varied_value_promotes_a_plain_array_as_rebinding_does(key: Any) -> None:
    session, ev = _source()
    value = graphed.vary(ev.m, "s", up=ev.m * 3, down=ev.m)
    rebound = (
        gak.with_field(ev, value, key)
        if isinstance(key, str)
        else gak.with_field(ev, gak.with_field(ev.r, value, "q"), "r")
    )
    ev[key] = value
    assert isinstance(ev, graphed.Varied)
    assert labels(ev) == ("nominal", "s_up", "s_down")
    assert _universes(session, ev) == _universes(session, rebound)


def test_a_promoted_array_takes_a_later_plain_assignment() -> None:
    session, ev = _source()
    ev["w"] = graphed.vary(ev.m, "s", up=ev.m * 3, down=ev.m)
    ev["v"] = ev.m + 1
    assert _universes(session, ev)["s_up"] == [
        {**row, "w": row["m"] * 3, "v": row["m"] + 1} for row in DATA_ROWS
    ]


def test_promotion_rebinds_this_object_only() -> None:
    session, ev = _source()
    alias, twin = ev, type(ev)(session, ev.node_id)
    ev["w"] = graphed.vary(ev.m, "s", up=ev.m * 3, down=ev.m)
    assert alias is ev
    assert isinstance(twin, graphed.Array)
    assert ak.to_list(session.materialize(twin)) == DATA_ROWS


def test_a_promoted_array_carries_no_array_state_and_copies_as_a_container() -> None:
    session, ev = _source()
    ev["w"] = graphed.vary(ev.m, "s", up=ev.m * 3, down=ev.m)
    assert [n for n in ("_node_id", "_session", "_site", "_labels") if hasattr(ev, n)] == []
    for copied in (copy.copy(ev), copy.deepcopy(ev)):
        assert type(copied) is type(ev)
        assert _universes(session, copied) == _universes(session, ev)


@pytest.mark.parametrize("value", [0.5, np.float32(0.5), 3, True, "tag"], ids=repr)
@pytest.mark.parametrize("assign", ["with_field", "setitem"])
def test_a_scalar_field_is_broadcast_in_its_own_dtype(value: Any, assign: str) -> None:
    session, ev = _source()
    if assign == "with_field":
        ev = gak.with_field(ev, value, "kl")
    else:
        ev["kl"] = value
    got: Any = session.materialize(ev)
    eager = ak.with_field(ak.Array(DATA), value, "kl")
    assert (str(got.type), ak.to_list(got)) == (str(eager.type), ak.to_list(eager))


@pytest.mark.parametrize("varied_value", [False, True], ids=["plain", "varied"])
def test_a_container_holding_the_proxy_keeps_its_member(varied_value: bool) -> None:
    session, ev = _source()
    held = graphed.vary(ev, "s", up=gak.with_field(ev, ev.m * 3, "m"))
    before = _universes(session, held)
    ev["w"] = graphed.vary(ev.m, "t", up=ev.m * 5) if varied_value else ev.m
    assert _universes(session, held) == before
