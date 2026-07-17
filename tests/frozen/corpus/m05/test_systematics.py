"""The two systematic kinds behave distinctly (the property M7/M9 depend on).

* kinematic (JES) variations re-run the selection -> the selected-event SET changes;
* weight (b-tag / photon SF) variations keep the SAME selected events, only reweighting them.
"""

from __future__ import annotations

import awkward as ak

from graphed_corpus import fingerprint, ttbar_region, ttgamma_region
from graphed_corpus.analyses.systematics import _apply_jes


def _selected_count(events: ak.Array, *, region: str, variation: str) -> int:
    """Unweighted count of events passing the ttbar selection under a variation."""
    jets = _apply_jes(events.Jet, variation=variation)
    good = jets[jets.pt > 25]
    n_good = ak.num(good, axis=1)
    n_b = ak.sum(good.btag > 0.7, axis=1)
    base = n_good >= 4
    sel = base & (n_b == 1) if region == "4j1b" else base & (n_b >= 2)
    return int(ak.sum(sel))


def test_kinematic_variation_changes_selection(events: ak.Array) -> None:
    nominal = _selected_count(events, region="4j1b", variation="nominal")
    jes_up = _selected_count(events, region="4j1b", variation="jes_up")
    jes_dn = _selected_count(events, region="4j1b", variation="jes_down")
    assert jes_up != nominal and jes_dn != nominal
    assert jes_up > nominal > jes_dn  # scaling jet pt up admits more events


def test_weight_variation_preserves_selection(events: ak.Array) -> None:
    nominal = _selected_count(events, region="4j2b", variation="nominal")
    btag_up = _selected_count(events, region="4j2b", variation="btag_up")
    btag_dn = _selected_count(events, region="4j2b", variation="btag_down")
    assert btag_up == nominal == btag_dn  # weight SF does not change which events are selected


def test_variations_produce_distinct_histograms(events: ak.Array) -> None:
    fps = {
        v: fingerprint(ttbar_region(events, region="4j1b", variation=v))
        for v in ("nominal", "jes_up", "jes_down", "btag_up", "btag_down")
    }
    assert len(set(fps.values())) == len(fps), "every variation must yield a distinct histogram"


def test_ttgamma_variations_distinct(events: ak.Array) -> None:
    fps = {
        v: fingerprint(ttgamma_region(events, variation=v))
        for v in ("nominal", "jes_up", "pho_up", "pho_down")
    }
    assert len(set(fps.values())) == len(fps)
