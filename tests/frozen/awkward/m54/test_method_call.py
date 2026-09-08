"""m54 §2.2 — calling a method records one op whose value is the eager call's.

Four argument shapes, because each reaches the encoder differently: a positional constant, a keyword
constant, an array passed positionally, and the same array passed by KEYWORD (an array operand that
is not in `args`, which a positional-only encoder would drop or mis-order).
"""

from __future__ import annotations

from inspect import currentframe

import awkward as ak
import numpy as np
from m54_behavior_fixtures import eager, recorded

from graphed.awkward import gak


def _leads() -> tuple[object, object, object, ak.Array, ak.Array]:
    session, jets, probe = recorded()
    lead = gak.firsts(jets, axis=1)
    partner = gak.firsts(probe, axis=1)
    return (
        session,
        lead,
        partner,
        ak.firsts(eager("Jet"), axis=1),
        ak.firsts(eager("Probe"), axis=1),
    )


def test_a_positional_constant_evaluates_as_the_eager_call() -> None:
    session, jets, _probe = recorded()
    got = ak.Array(session.materialize(jets.scaled(2.0)))
    assert ak.array_equal(got, eager("Jet").scaled(2.0))


def test_a_keyword_constant_evaluates_as_the_eager_call() -> None:
    session, jets, _probe = recorded()
    got = ak.Array(session.materialize(jets.scaled(2.0, offset=5.0)))
    assert ak.array_equal(got, eager("Jet").scaled(2.0, offset=5.0))


def test_an_array_argument_evaluates_the_same_positionally_and_by_keyword() -> None:
    session, lead, partner, e_lead, e_partner = _leads()
    reference = e_lead.near(e_partner)

    positional = ak.Array(session.materialize(lead.near(partner)))
    keyword = ak.Array(session.materialize(lead.near(other=partner)))
    assert ak.array_equal(positional, reference, equal_nan=True)
    assert ak.array_equal(keyword, reference, equal_nan=True)


def test_an_array_and_a_constant_mix_in_one_call() -> None:
    session, lead, partner, e_lead, e_partner = _leads()
    got = ak.Array(session.materialize(lead.near(partner, threshold=2.0)))
    assert ak.array_equal(got, e_lead.near(e_partner, threshold=2.0))


def test_a_numpy_scalar_constant_is_coerced_and_matches_the_python_scalar() -> None:
    session, jets, _probe = recorded()
    coerced = jets.scaled(np.float64(2.0))
    plain = jets.scaled(2.0)

    assert coerced.node_id == plain.node_id  # `.item()` at the encoder, not a second node
    assert ak.array_equal(
        ak.Array(session.materialize(coerced)), eager("Jet").scaled(2.0)
    )


def test_the_call_site_is_the_recorded_provenance() -> None:
    session, jets, _probe = recorded()
    line = currentframe().f_lineno + 1  # type: ignore[union-attr]
    result = jets.scaled(3.0)
    assert f"{__file__}:{line}" in str(session.provenance(result))
