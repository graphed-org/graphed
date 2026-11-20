"""Assigning into a universe read from a container leaves the container as it was (#231)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import awkward as ak
import pytest

import graphed
from graphed.awkward import from_awkward, gak

READERS: dict[str, Callable[[Any], Any]] = {
    "nominal": graphed.nominal,
    "universe": lambda v: graphed.universe(v, "s_up"),
    "member_of": lambda v: graphed.member_of(v, "s_up"),
    "member_of an uncarried label": lambda v: graphed.member_of(v, "other_up"),
}


def _container() -> tuple[Any, Any]:
    src = from_awkward(
        "e",
        ak.zip({"a": [1.0, 2.0], "b": [3.0, 4.0]}, depth_limit=1),
        session=graphed.new_session(),
    )
    r = gak.zip({"a": src.a}, depth_limit=1)
    up = gak.zip({"a": src.a * 2}, depth_limit=1)
    return src, graphed.vary(r, "s", up=up, down=gak.zip({"a": src.a / 2}, depth_limit=1))


def _node_ids(v: Any) -> dict[str, int]:
    return {label: graphed.universe(v, label).node_id for label in graphed.labels(v)}


@pytest.mark.parametrize("read", READERS.values(), ids=READERS.keys())
@pytest.mark.parametrize("value", ["plain", "varied"])
def test_an_assignment_into_a_read_universe_stays_out_of_the_container(
    read: Callable[[Any], Any], value: str
) -> None:
    src, v = _container()
    before = _node_ids(v)
    universe = read(v)
    universe["f"] = src.b if value == "plain" else graphed.vary(src.b, "t", up=src.b * 2)
    assert _node_ids(v) == before
    assert gak.fields(graphed.nominal(v)) == ["a"]
    assert "f" in gak.fields(universe)


def test_an_assignment_inside_varied_apply_stays_out_of_the_container() -> None:
    src, v = _container()
    before = _node_ids(v)

    def add_f(member: Any) -> Any:
        member["f"] = src.b
        return member

    applied = v.apply(add_f)
    assert _node_ids(v) == before
    assert gak.fields(graphed.nominal(applied)) == ["a", "f"]
