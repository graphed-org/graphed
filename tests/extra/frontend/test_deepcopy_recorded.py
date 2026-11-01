"""A deep copy of what holds recorded nodes stays on their one graph."""

from __future__ import annotations

import copy
import sys
from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.awkward.gnano import events as nano_events


@pytest.fixture
def root() -> Any:
    data = ak.Array({"a": [1.0, 2.0, 3.0], "Jet": [[{"pt": 1.0}], [], [{"pt": 2.0}]]})
    return from_awkward("e", data, session=Session(AwkwardBackend()))


def test_an_array_keeps_its_session_and_node(root: Any) -> None:
    a = root.a
    copied = copy.deepcopy(a)
    assert copied is not a
    assert copied.session is a.session
    assert copied.node_id == a.node_id
    assert graphed.materialize(copied + a).tolist() == [2.0, 4.0, 6.0]


def test_a_container_of_arrays_copies(root: Any) -> None:
    held = {"x": [root.a]}
    copied = copy.deepcopy(held)
    assert copied["x"] is not held["x"]
    assert graphed.materialize(copied)["x"][0].tolist() == [1.0, 2.0, 3.0]


def test_a_varied_keeps_its_labels_and_members(root: Any) -> None:
    varied = graphed.vary(root.a * 2, "s", up=root.a * 3, down=root.a)
    copied = copy.deepcopy(varied)
    assert graphed.labels(copied) == graphed.labels(varied)
    assert graphed.materialize(graphed.universe(copied, "s_up")).tolist() == [3.0, 6.0, 9.0]


def test_a_copied_read_stays_in_its_context(root: Any) -> None:
    ev = nano_events(root)
    pt = ev.Jet.pt
    context = graphed.context_of(pt)
    assert context is not None
    assert graphed.context_of(copy.deepcopy(pt)) is context


def test_a_long_history_copies_and_keeps_its_recording(root: Any) -> None:
    a = root.a
    for i in range(sys.getrecursionlimit()):
        a = a + i
    copied = copy.deepcopy(a)
    assert copied._site is a._site


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
def test_a_session_and_a_context_are_themselves(root: Any, copier: Any) -> None:
    ev = nano_events(root)
    assert copier(root.session) is root.session
    assert copier(graphed.context_of(ev.Jet.pt)) is graphed.context_of(ev.Jet.pt)
