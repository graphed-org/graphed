"""m60 integ-m60-E — `expand` is public.

A third-party verb that records its own `External` family has no other way to be `Varied`-aware:
it must map itself over the union's labels. `expand` is that mapping, and an unvaried call must
pass straight through, recording exactly what the bare call records.
"""

from __future__ import annotations

import inspect

from m60_seams import DATUM, VARIED_LABELS, third_party_verb, toy_session, varied_vector

import graphed
import graphed.systematics.varied
from graphed import Array, Varied


def test_expand_is_public_and_is_the_systematics_verb() -> None:
    assert graphed.expand is graphed.systematics.varied.expand
    assert "expand" in graphed.__all__


def test_a_wrapped_third_party_verb_answers_a_varied_over_a_varied_operand() -> None:
    session, varied, _plain = varied_vector()
    before = session.node_count()

    out = graphed.expand(third_party_verb, (varied,), {})

    assert isinstance(out, Varied)
    assert graphed.labels(out) == VARIED_LABELS
    members = [graphed.universe(out, label) for label in VARIED_LABELS]
    assert len({member.node_id for member in members}) == len(VARIED_LABELS)
    assert session.node_count() == before + len(VARIED_LABELS)


def test_an_unvaried_call_passes_straight_through_with_the_bare_calls_ir() -> None:
    built = []
    for wrapped in (False, True):
        session, x = toy_session()
        out = graphed.expand(third_party_verb, (x,), {}) if wrapped else third_party_verb(x)
        assert isinstance(out, Array) and not isinstance(out, Varied)
        assert session.materialize(out) == DATUM * 10.0
        built.append((session.node_count(), session.serialized_ir(out, optimize=False)))

    assert built[0] == built[1]


def test_expand_stays_outside_the_array_consuming_verb_surface() -> None:
    """m48 §2.3d walks `graphed.__all__` for functions whose annotations mention `Array` and
    demands a `VERB_DISPOSITIONS` entry for each. `expand` takes `(fn, args, kwargs)`, so joining
    `__all__` adds no disposition obligation and the older frozen enumeration stays green."""
    signature = inspect.signature(graphed.systematics.varied.expand)

    assert not any("Array" in str(p.annotation) for p in signature.parameters.values())
    assert "expand" not in graphed.VERB_DISPOSITIONS
