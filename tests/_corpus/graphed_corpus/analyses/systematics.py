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


def _region_mask(good: ak.Array, *, region: str) -> ak.Array:
    """Event mask: >=4 good jets, and the region's b-tag multiplicity."""
    base = ak.num(good, axis=1) >= 4
    n_b = ak.sum(good.btag > 0.7, axis=1)
    if region == "4j1b":
        return base & (n_b == 1)
    if region == "4j2b":
        return base & (n_b >= 2)
    raise ValueError(region)  # pragma: no cover - guarded by the fixture catalog


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
    sel_jets = good[_region_mask(good, region=region)]
    ht = ak.sum(sel_jets.pt, axis=1)
    weight = _btag_weight(sel_jets, variation=variation)

    h = Hist.new.Reg(40, 0, 800, name="ht").Double()
    h.fill(np.round(ak.to_numpy(ht), STABLE_DECIMALS), weight=ak.to_numpy(weight))
    return _round_hist(h)


def btag_sf_rel_uncertainty(pt: ak.Array) -> ak.Array:
    """Per-jet *fractional* b-tag SF uncertainty, as a function of the jet pT it is evaluated at.

    Rising from 1% to a 6% plateau at 100 GeV. Because it reads the pT of the universe being
    computed, a JES shift changes the size of the b-tag uncertainty as well as the selection —
    which is exactly why the two axes do not factorize.
    """
    return 0.01 + 0.05 * np.minimum(pt / 100.0, 1.0)


#: b-tag direction as a signed multiple of the fractional uncertainty; 0 reproduces the SF exactly.
_BTAG_DIRECTION = {"nominal": 0.0, "btag_up": 1.0, "btag_down": -1.0}

#: the pT-independent uncertainty `_btag_weight`'s flat +-3% rule uses.
_FLAT_REL_UNCERTAINTY = 0.03


def ttbar_joint_reference(
    events: ak.Array,
    *,
    region: str,
    jes: str,
    btag: str,
    pt_dependent: bool = True,
    freeze_selection: bool = False,
) -> Hist:
    """`ttbar_region` at an arbitrary *(jes, btag)* coordinate PAIR rather than one variation.

    `jes` is nominal / jes_up / jes_down, `btag` is nominal / btag_up / btag_down; selection,
    observable and binning are `ttbar_region`'s. Contents are returned unrounded — rounding is the
    comparison helpers' job (:func:`~graphed_corpus.histograms.bin_values`).

    The two knobs isolate where the two axes' cross term comes from: `pt_dependent=False` falls back
    to the flat +-3% rule, leaving only JES selection migration; `freeze_selection=True` additionally
    takes the jet and event selection from nominal kinematics while still evaluating the observable
    and the SF at the shifted pT, which removes the migration too and so leaves no cross term at all.
    """
    shifted = _apply_jes(events.Jet, variation=jes)
    selector = events.Jet if freeze_selection else shifted
    keep = selector.pt > 25
    sel = _region_mask(selector[keep], region=region)
    sel_jets = shifted[keep][sel]

    rel = btag_sf_rel_uncertainty(sel_jets.pt) if pt_dependent else _FLAT_REL_UNCERTAINTY
    sf = (0.95 + 0.10 * sel_jets.btag) * (1.0 + _BTAG_DIRECTION[btag] * rel)

    h = Hist.new.Reg(40, 0, 800, name="ht").Double()
    h.fill(ak.to_numpy(ak.sum(sel_jets.pt, axis=1)), weight=ak.to_numpy(ak.prod(sf, axis=1)))
    return h


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
