"""§2.3e: a contexted operand's handle survives every gak function that can carry it.

This is a SEPARATE, SCOPED gate from §2.3c's classification test. That one is metadata-only — it
reads a classification and never calls anything — so it runs over the full public surface for free.
A propagation gate must CALL each function, and a blanket call is impossible over the measured
surface (payload-first verbs, eager verbs returning Python objects, the refusing `join`, verbs
needing typed extra operands). So it enumerates only the three classes that can carry a handle,
takes its AUXILIARY arguments from fixtures that live in `src` beside the classification — a new
function arrives with both, and this file stays untouched — and owns the CONTEXTED primary operand
itself, because a fixture-supplied context-free primary would degrade the check to `None == None`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vary_ctx_fixtures import OPERAND_KINDS, events_context, operands

import graphed
from graphed.awkward import functions as gak_module
from graphed.awkward import gak

#: the classes this gate calls; the other two are exempt BY CLASSIFICATION, not by omission
CARRYING = ("broadcast", "container-traversing", "tuple-returning")
EXEMPT = {"eager-metadata", "refusing"}
#: §2.3e(3)'s membership floor — containment plus a monotone count, never an exact set, because a
#: frozen equality reds the moment a future gak boundary verb arrives with its classification
REFUSING_FLOOR = 1
BROADCAST_FLOOR = 40


def _fill(value: object, owned: Mapping[str, Any], filled: list[Any]) -> Any:
    """Replace every declared SLOT with the test's own contexted array of the kind it names —
    including inside a Mapping or Sequence argument, since `gak.zip`'s mapping is its only
    array-bearing operand and there is otherwise no position to substitute into."""
    if isinstance(value, gak_module.GakSlot):
        assert value.kind in OPERAND_KINDS  # the KIND vocabulary is frozen at m48
        replacement = owned[value.kind]
        filled.append(replacement)
        return replacement
    if isinstance(value, Mapping):
        return {key: _fill(item, owned, filled) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return type(value)(_fill(item, owned, filled) for item in value)  # type: ignore[call-arg]
    return value


def _handles(result: object) -> list[object]:
    if isinstance(result, tuple):
        return [graphed.context_of(item) for item in result]
    return [graphed.context_of(result)]


def _carrying_names() -> list[str]:
    return sorted(
        name for name, kind in gak_module.GAK_DISPOSITIONS.items() if kind in CARRYING
    )


def test_every_carrying_function_propagates_the_context_handle_it_was_given() -> None:
    _session, ctx = events_context()
    owned = operands(ctx)  # one contexted array per kind, ALL read through the SAME context
    assert all(graphed.context_of(operand) is ctx for operand in owned.values())

    for name in _carrying_names():
        fixture = gak_module.GAK_ARG_FIXTURES[name]
        filled: list[Any] = []
        args = _fill(fixture.args, owned, filled)
        kwargs = _fill(fixture.kwargs, owned, filled)
        assert filled, f"{name}'s fixture declares no substitution slot, so nothing carried a handle"
        handles = _handles(getattr(gak, name)(*args, **kwargs))
        assert handles, f"gak.{name} returned nothing to read a handle from"
        for handle in handles:
            assert handle is not None, f"gak.{name} DROPPED the handle it was given"
            assert handle is ctx, f"gak.{name} answered with a handle that is not its input's"


def test_every_carrying_function_has_an_argument_fixture_in_src() -> None:
    """The self-repairing rule: the fixture is a property of the function and lives beside its
    classification, so adding a gak function never requires editing this frozen file."""
    missing = [name for name in _carrying_names() if name not in gak_module.GAK_ARG_FIXTURES]
    assert not missing, f"{missing} are classified as carrying but cannot be called by this gate"


def test_the_exempt_set_is_exactly_eager_metadata_and_refusing() -> None:
    classes = set(gak_module.GAK_DISPOSITIONS.values())
    assert classes - set(CARRYING) == EXEMPT


def test_the_membership_floor_on_the_exempt_classes() -> None:
    """Classification is implementer-editable `src`, so without a membership floor a hard member
    could be re-classified into an exempt class and stay green while the class NAMES stay right."""
    dispositions = gak_module.GAK_DISPOSITIONS
    refusing = {name for name, kind in dispositions.items() if kind == "refusing"}
    broadcast = {name for name, kind in dispositions.items() if kind == "broadcast"}
    assert "join" in refusing  # gak's only boundary verb at freeze
    assert len(refusing) >= REFUSING_FLOOR
    assert len(broadcast) >= BROADCAST_FLOOR
