"""C1 iteration witness — the dependency discriminator's four cases (numpy idiom, awkward-free).

Pins the two hard-won distinctions the frozen suite spreads across trees, so a regression during the
C3/C4 work trips here first: a spectator foreign nuisance (carrier varied but not by it) and a
stacked weight (foreign nuisance in the ambient's `_tags`) both stay the union; a fresh carrier and a
carrier that shares the foreign nuisance both fan out.
"""

from __future__ import annotations

import numpy as np

import graphed
from graphed import Session
from graphed.context import EventContext
from graphed.numpy import NumpyBackend, from_record


def _record() -> tuple[Session, object]:
    session = Session(NumpyBackend())
    return session, from_record(session, "ev", pt=np.arange(1.0, 7.0), eta=np.arange(1.0, 7.0) / 6)


def test_a_fresh_carrier_fans_out_a_foreign_member() -> None:
    _s, record = _record()
    x = record["pt"]
    other = graphed.vary(x, "jer", up=x * 5.0)  # jer-varied member
    z = graphed.vary(x * 4.0, "jes", up=other)  # fresh (plain) target → fanout
    assert "jes_up__jer_up" in graphed.labels(z)
    assert graphed.member_of(z, "jes_up__jer_up").node_id == graphed.member_of(other, "jer_up").node_id


def test_a_spectator_foreign_nuisance_stays_the_union() -> None:
    _s, record = _record()
    x = record["pt"]
    base = graphed.vary(x, "jes", up=x * 1.1, down=x * 0.9)  # carrier varied by jes
    supplied = graphed.vary(x * 1.1, "inner", hi=x * 9.9)  # member varied by a DIFFERENT nuisance
    stacked = graphed.vary(base, "jer", up=supplied, down=x * 0.9)
    # inner is a spectator on a jes-varied carrier → no jer x inner joint; jer_up is supplied nominal
    assert not [label for label in graphed.labels(stacked) if "__" in label]
    assert graphed.universe(stacked, "jer_up").node_id == graphed.nominal(supplied).node_id


def test_a_carrier_sharing_the_foreign_nuisance_fans_out() -> None:
    _s, record = _record()
    x = record["pt"]
    jes = graphed.vary(x, "jes", up=x * 1.1, down=x * 0.9)
    dependent = jes * 3.0  # jes-varied, and the carrier (jes) shares that nuisance
    corr = graphed.vary(jes, "corr", variations={"a": dependent})
    assert "corr_a__jes_up" in graphed.labels(corr)
    assert "corr_a__jes_down" in graphed.labels(corr)


def test_a_stacked_weight_stays_the_union() -> None:
    _s, record = _record()
    pt = record["pt"]
    ctx = EventContext(_s, pt, collections={"pt": pt})
    weight = ctx["pt"] * 0.5
    first = graphed.vary(ctx, "btag", weight, is_weight=True, variations={"up": weight * 1.2})
    ambient = graphed.weight(first)  # its _tags carries btag → a stacked weight nuisance
    second = graphed.vary(first, "mu", ambient, is_weight=True, variations={"up": ambient * 1.05})
    labels = graphed.labels(graphed.weight(second))
    assert not [label for label in labels if "__" in label]  # mu does NOT fan out over stacked btag
    assert set(labels) == {"nominal", "btag_up", "mu_up"}
