"""m54 §2.2 refusals and §2.7 typetracer errors — both raise `GraphedTypeError` and record nothing.

The two families are distinct mechanisms and are pinned apart: a non-JSON ARGUMENT is refused by the
encoder before the typetracer is ever asked, while a well-formed call that the METHOD itself rejects
is refused by the record-time typetracer run. Either way the session's node count is unchanged, so a
"record first, validate later" implementation fails here even though its exception type is right.

The accepted-constant test is the discrimination for the refusals: a blanket "collections are not
constants" rule would pass every refusal case and fail that one.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import currentframe
from typing import Any

import awkward as ak
import numpy as np
import pytest
from m54_behavior_fixtures import eager, op_count, recorded

from graphed import GraphedTypeError
from graphed.awkward import gak

#: (case id, method name, the call, the argument locator the message must name)
REFUSED: list[tuple[str, str, Callable[[Any], Any], str]] = [
    ("ndarray", "scaled", lambda jets: jets.scaled(np.arange(3.0)), "args[0]"),
    ("eager_array", "scaled", lambda jets: jets.scaled(eager("Jet").pt), "args[0]"),
    ("callable_kwarg", "scaled", lambda jets: jets.scaled(2.0, offset=len), "kwargs['offset']"),
    ("nan", "scaled", lambda jets: jets.scaled(float("nan")), "args[0]"),
    ("inf", "scaled", lambda jets: jets.scaled(2.0, offset=float("inf")), "kwargs['offset']"),
    ("set", "combo", lambda jets: jets.combo({1.0, 2.0}), "args[0]"),
    ("non_str_dict_keys", "weighted", lambda jets: jets.weighted({0: 1.0, 1: 2.0}), "args[0]"),
    ("marker_dict", "weighted", lambda jets: jets.weighted({"$": 0}), "args[0]"),
    ("marker_in_list", "combo", lambda jets: jets.combo([{"$": 0}, 1.0]), "args[0][0]"),
    ("marker_nested", "weighted", lambda jets: jets.weighted({"$": {"$": 0}}), "args[0]"),
    ("marker_kwarg", "scaled", lambda jets: jets.scaled(2.0, offset={"$": 0}), "kwargs['offset']"),
]


@pytest.mark.parametrize(
    ("method", "call", "locator"),
    [case[1:] for case in REFUSED],
    ids=[case[0] for case in REFUSED],
)
def test_a_non_json_argument_is_refused_before_any_node_is_recorded(
    method: str, call: Callable[[Any], Any], locator: str
) -> None:
    session, jets, _probe = recorded()
    before = op_count(session)

    with pytest.raises(GraphedTypeError) as excinfo:
        call(jets)

    message = str(excinfo.value)
    assert method in message
    assert locator in message
    assert op_count(session) == before


def test_the_json_representable_constants_are_accepted() -> None:
    session, jets, probe = recorded()
    lead, partner = gak.firsts(jets, axis=1), gak.firsts(probe, axis=1)
    weights = {"pt": 2.0, "eta": 3.0}

    blended = ak.Array(session.materialize(jets.weighted(weights)))
    assert ak.array_equal(blended, eager("Jet").weighted(weights))

    explicit_none = ak.Array(session.materialize(lead.near(partner, threshold=None)))
    reference = ak.firsts(eager("Jet"), axis=1).near(
        ak.firsts(eager("Probe"), axis=1), threshold=None
    )
    assert ak.array_equal(explicit_none, reference, equal_nan=True)


def test_a_dict_is_refused_only_when_the_marker_is_its_whole_shape() -> None:
    """`{"$": i}` is how an array reference is encoded, so only a dict that IS that shape is
    ambiguous. A blanket "no `$` key" rule would refuse this one and is what the leg discriminates."""
    session, jets, _probe = recorded()
    weights = {"pt": 2.0, "eta": 3.0, "$": 1.0}

    got = ak.Array(session.materialize(jets.weighted(weights)))
    assert ak.array_equal(got, eager("Jet").weighted(weights))


def test_an_array_from_another_session_is_refused_as_a_method_argument() -> None:
    session, jets, probe = recorded()
    _foreign_session, foreign, _foreign_probe = recorded()
    lead, partner = gak.firsts(jets, axis=1), gak.firsts(probe, axis=1)
    foreign_lead = gak.firsts(foreign, axis=1)
    before = op_count(session)

    with pytest.raises(GraphedTypeError) as excinfo:
        lead.near(foreign_lead)

    assert "different Session" in str(excinfo.value)
    assert op_count(session) == before
    assert lead.near(partner).node_id  # the same call on this session's own array records


def test_an_array_from_another_session_is_refused_by_a_binary_op() -> None:
    """The guard lives in `Session.record_op`, which every recorder routes through — so a method is
    not a special case and a guard installed only in the method encoder fails here."""
    session, jets, _probe = recorded()
    _foreign_session, foreign, _foreign_probe = recorded()
    mine, theirs = jets.pt, foreign.pt
    before = op_count(session)

    with pytest.raises(GraphedTypeError) as excinfo:
        _ = mine + theirs

    assert "different Session" in str(excinfo.value)
    assert op_count(session) == before


def test_a_zero_dimensional_numpy_array_is_coerced_not_refused() -> None:
    """The refusal is about ELEMENTS, not about numpy: a 0-d array is a scalar and passes `.item()`."""
    _session, jets, _probe = recorded()
    assert jets.scaled(np.array(2.0)).node_id == jets.scaled(2.0).node_id


def test_an_unknown_keyword_is_refused_at_the_call_site() -> None:
    session, jets, _probe = recorded()
    before = op_count(session)

    with pytest.raises(GraphedTypeError) as excinfo:
        line = currentframe().f_lineno + 1  # type: ignore[union-attr]
        jets.scaled(2.0, bogus=1.0)

    assert f"{__file__}:{line}" in str(excinfo.value)
    assert op_count(session) == before


def test_an_argument_the_method_rejects_is_refused_at_the_call_site() -> None:
    """`deltaR` demands a vector; a bare float array is a legal graph operand and an illegal one
    for this method — the failure belongs to the typetracer run, not to the encoder."""
    session, jets, _probe = recorded()
    lead = gak.firsts(jets, axis=1)
    not_a_vector = gak.firsts(jets.pt, axis=1)
    before = op_count(session)

    with pytest.raises(GraphedTypeError) as excinfo:
        line = currentframe().f_lineno + 1  # type: ignore[union-attr]
        lead.deltaR(not_a_vector)

    assert f"{__file__}:{line}" in str(excinfo.value)
    assert op_count(session) == before
