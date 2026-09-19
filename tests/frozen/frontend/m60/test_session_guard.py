"""m60 integ-m60-X — a `Session` refuses an `Array` it did not record.

A node id only means something in its own store, so a foreign `Array` whose id happens to exist
here is spliced in as whatever node wears that id: a wrong answer, not an error. The method set is
an INSTRUMENT, not a list — a `Session` method that grows an `Array` parameter without a leg here
reds the first test below.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import pytest
from m60_seams import DATUM, ToyForm, colliding_sessions, toy_session, walk_handlers

import graphed
from graphed import GraphedTypeError, Session, compile_ir

#: the `record_*` family: refuses with `GraphedTypeError`, carrying `record_op`'s message
RECORDING = ("record_exchange", "record_external", "record_join", "record_op")
#: the reading entries: refuse with `TypeError` (`GraphedTypeError` is not one, so this separates)
READING = ("form", "materialize", "provenance", "serialized_ir", "walk")


def _identity(value: object) -> object:
    return value


def _session_methods_taking_an_array() -> set[str]:
    """Public `Session` methods with an `Array`-annotated parameter.

    By ANNOTATION and not by parameter name: every such parameter here annotates `Array`,
    `Sequence[Array]` or `*outputs: Array`, while the names are six different spellings.
    """
    found: set[str] = set()
    for name, member in inspect.getmembers(Session, inspect.isfunction):
        if name.startswith("_"):
            continue
        params = inspect.signature(member).parameters.values()
        if any("Array" in str(param.annotation) for param in params):
            found.add(name)
    return found


def _module_verbs_taking_a_session() -> set[str]:
    """`graphed.__all__` functions that take a session beside arrays.

    Keyed on the `session` PARAMETER and not on annotations: `compile_ir`'s array parameter is
    `*outputs: Any`, so an annotation walk finds nothing to guard.
    """
    found: set[str] = set()
    for name in graphed.__all__:
        member = getattr(graphed, name)
        if isinstance(member, type) or not callable(member):
            continue
        try:
            params = inspect.signature(member).parameters
        except (TypeError, ValueError):
            continue
        if "session" in params:
            found.add(name)
    return found


def _calls(own: Session, mine: Any, foreign: Any) -> dict[str, Callable[[], object]]:
    return {
        "record_exchange": lambda: own.record_exchange(foreign, {"scheme": "count", "parts": 2}),
        "record_external": lambda: own.record_external("map", _identity, [foreign], {"fn": "identity"}),
        "record_join": lambda: own.record_join(mine, foreign, {"on": "k", "how": "inner"}),
        "record_op": lambda: own.record_op("add", [mine, foreign]),
        "form": lambda: own.form(foreign),
        "materialize": lambda: own.materialize(foreign),
        "provenance": lambda: own.provenance(foreign),
        "serialized_ir": lambda: own.serialized_ir(foreign, optimize=False),
        "walk": lambda: own.walk(foreign, **walk_handlers()),  # type: ignore[arg-type]
    }


def test_the_instrument_finds_no_array_taking_method_without_a_leg() -> None:
    assert _session_methods_taking_an_array() == {*RECORDING, *READING}


def test_a_foreign_arrays_id_addresses_a_real_node_of_the_refusing_session() -> None:
    """The premise the refusal exists for: the ids COLLIDE, so today's answer is another
    session's node, silently."""
    own, alpha, other, beta = colliding_sessions()

    assert beta.node_id == alpha.node_id
    assert own.form_of(beta.node_id) == ToyForm("alpha")
    assert other.form(beta) == ToyForm("beta")


@pytest.mark.parametrize("method", [*RECORDING, *READING])
def test_every_entry_refuses_a_foreign_array_and_records_nothing(method: str) -> None:
    own, alpha, _other, beta = colliding_sessions()
    before = own.node_count()

    expected = GraphedTypeError if method in RECORDING else TypeError
    with pytest.raises(expected) as info:
        _calls(own, alpha, beta)[method]()

    assert own.node_count() == before
    if method in RECORDING:
        assert "recorded in a different Session" in str(info.value)


def test_the_instrument_finds_no_module_verb_taking_a_session_without_a_leg() -> None:
    assert _module_verbs_taking_a_session() == {"compile_ir"}


def test_a_module_verb_refuses_a_foreign_array_and_records_nothing() -> None:
    own, _alpha, _other, beta = colliding_sessions()
    before = own.node_count()

    with pytest.raises(TypeError):
        compile_ir(own, beta)

    assert own.node_count() == before


def test_own_session_calls_record_and_read_exactly_as_today() -> None:
    own, alpha, _other, _beta = colliding_sessions()

    assert own.form(alpha) == ToyForm("alpha")
    assert own.materialize(alpha) == ("alpha", [DATUM], {})
    assert own.provenance(alpha).function == "colliding_sessions"
    assert own.walk(alpha, source=lambda _n: DATUM, op=lambda *a: a[1], external=lambda *a: a) == "alpha"
    assert own.node_count() == 2

    exchanged = own.record_exchange(alpha, {"scheme": "count", "parts": 2})
    joined = own.record_join(alpha, exchanged, {"on": "k", "how": "inner"})
    external = own.record_external("map", _identity, [joined], {"fn": "identity"})
    assert own.node_count() == 5
    assert [own.form(node).tag for node in (exchanged, joined, external)] == ["exchange", "join", "map"]

    twice = [toy_session("x") for _ in range(2)]
    bytes_of = {
        session.serialized_ir(session.record_op("alpha", [mine]), optimize=False) for session, mine in twice
    }
    assert len(bytes_of) == 1
