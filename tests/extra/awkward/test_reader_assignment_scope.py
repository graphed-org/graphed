"""What an assignment into a value a systematics reader returns reaches (#299)."""

from __future__ import annotations

from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.context import EventContext
from graphed.systematics import nominal, universe, weight


def _context() -> tuple[Any, Any]:
    session = Session(AwkwardBackend())
    data = ak.zip({"pt": [1.0, 2.0], "w": [1.0, 1.0], "j": ak.zip({"x": [[1.0], [2.0]]})})
    rec = from_awkward("e", data, session=session)
    return rec, EventContext(session, rec, collections={"j": rec.j, "pt": rec.pt, "w": rec.w})


def _collections(ctx: Any) -> dict[str, Any]:
    return {name: graphed.varied.member_nodes(ctx[name]) for name in ctx._collections}


@pytest.mark.parametrize("label", ["jes_up", "nominal"])
def test_a_collection_assigned_into_a_context_universe_stays_in_that_universe(label: str) -> None:
    rec, ctx = _context()
    j = {"up": gak.zip({"x": rec.j.x * 1.1}), "down": gak.zip({"x": rec.j.x * 0.9})}
    v = graphed.vary(ctx, "jes", collections={"j": j})
    others = [x for x in ("nominal", "jes_up", "jes_down") if x != label]
    before = {x: _collections(universe(v, x)) for x in others}
    parent = _collections(v)

    u = universe(v, label)
    u["own"] = u["pt"] * 2
    u["parents"] = v["pt"] * 3

    assert universe(v, label) is u and {"own", "parents"} <= set(u._collections)
    assert _collections(v) == parent
    assert {x: _collections(universe(v, x)) for x in others} == before
    assert graphed.context_of(u["parents"]) is u


@pytest.mark.parametrize("pick", [lambda w: w, nominal], ids=["container", "member"])
def test_a_field_assigned_into_the_ambient_weight_leaves_the_registry_as_it_was(pick: Any) -> None:
    rec, ctx = _context()
    up = gak.zip({"a": rec.w * 2, "b": rec.w})
    v = graphed.vary(ctx, "sf", gak.zip({"a": rec.w, "b": rec.w}), is_weight=True, up=up)
    w = pick(weight(v))
    w["c"] = rec.w
    assert gak.fields(nominal(weight(v))) == ["a", "b"]
    assert gak.fields(graphed.varied.member_of(w, "nominal")) == ["a", "b", "c"]
