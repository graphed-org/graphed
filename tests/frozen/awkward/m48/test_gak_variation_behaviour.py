"""Per-class gak representatives, §2 validation errors, and §2.4's label-aligned combination.

§2.3c's exhaustiveness gate asserts a classification EXISTS, not that it is RIGHT, and the corpus
matrices cannot reach these classes — the corpus fixture uses `ak.with_field`, never `ak.zip` — so
each otherwise-unanchored class gets one named representative here.
"""

from __future__ import annotations

import pytest
from vary_ctx_fixtures import EVENTS, awkward_session

import graphed
from graphed import GraphedError
from graphed.awkward import from_awkward, gak


def _varied_pt() -> tuple[graphed.Session, object, graphed.Varied]:
    session, root = awkward_session()
    jets = root.Jet
    return session, root, graphed.vary(jets.pt, "jes", up=jets.pt * 1.05, down=jets.pt * 0.95)


def test_zip_traverses_its_mapping_argument() -> None:
    """*container-traversing*: the `Varied` is INSIDE the Mapping, never a positional operand."""
    _s, root, varied = _varied_pt()
    zipped = gak.zip({"pt": varied, "eta": root.Jet.eta})
    assert isinstance(zipped, graphed.Varied)
    assert list(graphed.labels(zipped)) == list(graphed.labels(varied))
    for label in graphed.labels(varied):
        member = gak.zip({"pt": graphed.universe(varied, label), "eta": root.Jet.eta})
        assert graphed.universe(zipped, label).node_id == member.node_id


def test_unzip_returns_a_tuple_of_varied() -> None:
    """*tuple-returning*: the wrapper rebuilds its result, so it bypasses `record_op`'s merge."""
    _s, root, varied = _varied_pt()
    unzipped = gak.unzip(gak.zip({"pt": varied, "eta": root.Jet.eta}))
    assert isinstance(unzipped, tuple) and len(unzipped) == 2
    assert all(isinstance(item, graphed.Varied) for item in unzipped)
    assert all(list(graphed.labels(item)) == list(graphed.labels(varied)) for item in unzipped)


def test_the_eager_metadata_verbs_answer_on_the_nominal_member() -> None:
    """Sound because §2.1 requires form compatibility across members — the same argument §2.2 uses
    for the form-answered properties."""
    _s, root, varied = _varied_pt()
    zipped = gak.zip({"pt": varied, "eta": root.Jet.eta})
    assert gak.fields(zipped) == gak.fields(gak.zip({"pt": root.Jet.pt, "eta": root.Jet.eta}))
    assert gak.type_of(varied) == gak.type_of(root.Jet.pt)


def test_a_varied_meeting_one_derived_from_itself_stays_label_aligned() -> None:
    """`jets[jets.pt > 25]` is the canonical case; within a universe every use is coherent."""
    _s, _root, varied = _varied_pt()
    mask = varied > 25.0
    selected = varied[mask]
    assert list(graphed.labels(selected)) == list(graphed.labels(varied))
    for label in graphed.labels(varied):
        member = graphed.universe(varied, label)
        assert graphed.universe(selected, label).node_id == member[graphed.universe(mask, label)].node_id


def test_the_union_order_is_the_first_operands_then_the_seconds() -> None:
    """§3.2's determinism gate and §6.1c's positional layout both depend on this order, and an
    unbound one would let two conforming implementations disagree."""
    _s, root, jes = _varied_pt()
    jer = graphed.vary(root.Jet.pt, "jer", hi=root.Jet.pt * 1.02, lo=root.Jet.pt * 0.98)
    assert list(graphed.labels(jes * jer)) == ["nominal", "jes_up", "jes_down", "jer_hi", "jer_lo"]
    assert list(graphed.labels(jer * jes)) == ["nominal", "jer_hi", "jer_lo", "jes_up", "jes_down"]


def test_a_label_absent_from_one_operand_takes_that_operands_nominal_member() -> None:
    """One-at-a-time: cross products can never arise implicitly, since §2.1 makes each label belong
    to exactly one knob."""
    _s, root, jes = _varied_pt()
    jer = graphed.vary(root.Jet.pt, "jer", hi=root.Jet.pt * 1.02)
    combined = jes * jer
    expected = graphed.universe(jes, "jes_up") * graphed.nominal(jer)
    assert graphed.universe(combined, "jes_up").node_id == expected.node_id
    cross = graphed.universe(jes, "jes_up") * graphed.universe(jer, "jer_hi")
    assert graphed.universe(combined, "jes_up").node_id != cross.node_id


def test_an_unknown_label_raises_listing_the_valid_ones() -> None:
    _s, _root, varied = _varied_pt()
    with pytest.raises(KeyError, match="jes_up"):
        graphed.universe(varied, "jes_upp")


def test_a_form_incompatible_member_is_a_construction_time_error_naming_the_label() -> None:
    _s, root, _varied = _varied_pt()
    with pytest.raises(GraphedError, match="sig_up"):
        graphed.vary(root.MET.pt, "sig", up=root.Jet)


def test_members_from_another_session_or_another_source_are_refused() -> None:
    """`aggregate_plan`'s single-source check is the otherwise-deferred failure surface, which is
    why this is checked at construction instead."""
    _s1, root1, _varied = _varied_pt()
    _s2, root2 = awkward_session()
    with pytest.raises(GraphedError):
        graphed.vary(root1.MET.pt, "sig", up=root2.MET.pt)

    session, root = awkward_session()
    other = from_awkward(session, "other", EVENTS)  # a SECOND partitioned-source root
    with pytest.raises(GraphedError):
        graphed.vary(root.MET.pt, "sig", up=other.MET.pt)
