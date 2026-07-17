"""AGC ttbar-style and TTGamma-style slices exercising the systematic patterns from the corpus.

These distill the two systematic *kinds* the plan repeatedly stresses (A.3.1, M7, M9):

* **weight systematics** — a per-event reweighting that does NOT change which events/objects are
  selected (e.g. b-tagging scale factors, pileup, lepton SF). Implemented as a multiplicative
  weight variation; the selection is identical across variations.
* **kinematic systematics** — a variation (e.g. JES/JER) that shifts object kinematics and so
  *changes the selection itself* and the observables. Implemented by scaling jet pt before the
  object/region selection is recomputed.

Both produce a process x variation set of histograms exactly as coffea/AGC does. The histograms are
weighted (Double storage) but we round contents for stable cross-platform comparison.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
from hist import Hist

from ..histograms import STABLE_DECIMALS


def _btag_weight(jets: ak.Array, *, variation: str) -> ak.Array:
    """A correctionlib-style b-tag scale factor as a per-event weight (weight systematic).

    Stands in for a real correctionlib JSON evaluation; M3/M9 replace this with an External node
    whose PayloadDescriptor content-hashes the correction set.
    """
    central = 0.95 + 0.10 * jets.btag  # per-jet SF in [0.95, 1.05]
    if variation == "btag_up":
        central = central * 1.03
    elif variation == "btag_down":
        central = central * 0.97
    return ak.prod(central, axis=1)


def _apply_jes(jets: ak.Array, *, variation: str) -> ak.Array:
    """Kinematic (JES) variation: scale jet pt, which changes downstream selection."""
    if variation == "jes_up":
        return ak.with_field(jets, jets.pt * 1.05, "pt")
    if variation == "jes_down":
        return ak.with_field(jets, jets.pt * 0.95, "pt")
    return jets


def _round_hist(h: Hist) -> Hist:
    view = h.view()
    view[...] = np.round(view, STABLE_DECIMALS)
    return h


def ttbar_region(events: ak.Array, *, region: str, variation: str) -> Hist:
    """AGC-style ttbar slice: >=4 jets pt>25; 4j1b (==1 b-tag) or 4j2b (>=2 b-tags).

    Observable: HT (scalar sum jet pt). `variation` is one of nominal / jes_up / jes_down
    (kinematic, re-runs selection) / btag_up / btag_down (weight only).
    """
    jets = _apply_jes(events.Jet, variation=variation)
    good = jets[jets.pt > 25]
    n_good = ak.num(good, axis=1)
    is_b = good.btag > 0.7
    n_b = ak.sum(is_b, axis=1)

    base = n_good >= 4
    if region == "4j1b":
        sel = base & (n_b == 1)
    elif region == "4j2b":
        sel = base & (n_b >= 2)
    else:  # pragma: no cover - guarded by the fixture catalog
        raise ValueError(region)

    sel_jets = good[sel]
    ht = ak.sum(sel_jets.pt, axis=1)
    weight = _btag_weight(sel_jets, variation=variation)

    h = Hist.new.Reg(40, 0, 800, name="ht").Double()
    h.fill(np.round(ak.to_numpy(ht), STABLE_DECIMALS), weight=ak.to_numpy(weight))
    return _round_hist(h)


def ttgamma_region(events: ak.Array, *, variation: str) -> Hist:
    """TTGamma-style slice: >=1 photon pt>20, >=1 muon pt>30, >=2 jets pt>25; photon-pt observable
    with a photon-ID scale factor as a weight systematic (pho_up/pho_down)."""
    photons = events.Photon[events.Photon.pt > 20]
    muons = events.Muon[events.Muon.pt > 30]
    jets = _apply_jes(events.Jet, variation=variation)
    good_jets = jets[jets.pt > 25]

    sel = (ak.num(photons, axis=1) >= 1) & (ak.num(muons, axis=1) >= 1) & (ak.num(good_jets, axis=1) >= 2)
    lead_pho_pt = ak.firsts(photons[sel].pt)

    sf = 0.98
    if variation == "pho_up":
        sf = 1.01
    elif variation == "pho_down":
        sf = 0.95
    weight = np.full(int(ak.sum(sel)), sf, dtype=np.float64)

    h = Hist.new.Reg(30, 0, 300, name="photon_pt").Double()
    h.fill(np.round(ak.to_numpy(ak.drop_none(lead_pho_pt)), STABLE_DECIMALS), weight=weight)
    return _round_hist(h)


# Process x variation set the AGC/coffea pattern produces.
TTBAR_FIXTURES = {
    f"ttbar_{region}_{var}": (region, var)
    for region in ("4j1b", "4j2b")
    for var in ("nominal", "jes_up", "jes_down", "btag_up", "btag_down")
}
TTGAMMA_FIXTURES = {f"ttgamma_{var}": var for var in ("nominal", "jes_up", "jes_down", "pho_up", "pho_down")}
