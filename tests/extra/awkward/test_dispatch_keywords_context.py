"""ak's defaulted keywords on an event context and on a ``Varied`` (#256 on the context stack)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gnano
from graphed.errors import GraphedTypeError
from graphed.systematics import universe

EAGER = ak.Array({"run": [1, 2, 3], "MET": [{"pt": 5.0}, {"pt": 15.0}, {"pt": 25.0}]})


def _events() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    return session, gnano.events(from_awkward("ev", EAGER, session=session), is_data=False)


def _value(session: Session, out: Any) -> tuple[str, list[Any]]:
    value = ak.Array(session.materialize(out))
    return str(value.type), value.tolist()


CONTEXT_CALLS: dict[str, Callable[[Any, dict[str, Any]], Any]] = {
    "zip of context fields": lambda ev, kw: ak.zip({"a": ev.MET.pt, "b": ev.run}, **kw),
    "with_field on the context": lambda ev, kw: ak.with_field(ev, ev.MET.pt * 2, "x", **kw).x,
}


@pytest.mark.parametrize("call", CONTEXT_CALLS.values(), ids=CONTEXT_CALLS.keys())
@pytest.mark.parametrize("keyword", [{"highlevel": True}, {"behavior": None}, {"attrs": None}])
def test_a_defaulted_keyword_on_a_context_equals_the_call_without_it(
    call: Callable[[Any, dict[str, Any]], Any], keyword: dict[str, Any]
) -> None:
    session, ev = _events()
    assert _value(session, call(ev, keyword)) == _value(session, call(ev, {}))


@pytest.mark.parametrize("call", CONTEXT_CALLS.values(), ids=CONTEXT_CALLS.keys())
def test_a_non_default_keyword_on_a_context_is_refused_by_name(
    call: Callable[[Any, dict[str, Any]], Any],
) -> None:
    _session, ev = _events()
    with pytest.raises(GraphedTypeError, match="highlevel=False"):
        call(ev, {"highlevel": False})


def _varied(ev: Any) -> Any:
    met = ev.MET.pt
    return graphed.vary(met, "s", points={"up": met * 2})


@pytest.mark.parametrize("keyword", [{"highlevel": True}, {"behavior": None}, {"attrs": None}])
def test_a_defaulted_keyword_on_a_varied_equals_the_call_without_it(
    keyword: dict[str, Any],
) -> None:
    session, ev = _events()
    v = _varied(ev)
    got, want = ak.zip({"a": v, "b": ev.run}, **keyword), ak.zip({"a": v, "b": ev.run})
    for label in ("nominal", "s_up"):
        assert _value(session, universe(got, label)) == _value(session, universe(want, label))


def test_a_non_default_keyword_on_a_varied_is_refused_by_name() -> None:
    _session, ev = _events()
    with pytest.raises(GraphedTypeError, match="attrs="):
        ak.zip({"a": _varied(ev), "b": ev.run}, attrs={"k": 1})
