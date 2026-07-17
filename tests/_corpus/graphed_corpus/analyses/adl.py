"""ADL benchmark queries 1-8 as plain awkward analyses (the graded functional-test ladder).

Derived from the IRIS-HEP ADL benchmark task definitions
(github.com/iris-hep/adl-benchmarks-index) and the coffea reference implementations. Each function
takes the event array and returns a 1D reference histogram. These are the fixtures M3/M4/M5 use to
prove ``graphed`` reproduces plain awkward bit-for-bit.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
from hist import Hist

from ..histograms import hist1d

PI = np.pi


def _delta_phi(a: ak.Array, b: ak.Array) -> ak.Array:
    return (a - b + PI) % (2 * PI) - PI


def _delta_r(eta1: ak.Array, phi1: ak.Array, eta2: ak.Array, phi2: ak.Array) -> ak.Array:
    return np.hypot(eta1 - eta2, _delta_phi(phi1, phi2))


def _pair_mass(o1: ak.Array, o2: ak.Array) -> ak.Array:
    """Invariant mass of two objects from (pt, eta, phi, mass)."""
    px = o1.pt * np.cos(o1.phi) + o2.pt * np.cos(o2.phi)
    py = o1.pt * np.sin(o1.phi) + o2.pt * np.sin(o2.phi)
    pz = o1.pt * np.sinh(o1.eta) + o2.pt * np.sinh(o2.eta)
    e1 = np.sqrt(o1.pt**2 * np.cosh(o1.eta) ** 2 + o1.mass**2)
    e2 = np.sqrt(o2.pt**2 * np.cosh(o2.eta) ** 2 + o2.mass**2)
    e = e1 + e2
    m2 = e**2 - (px**2 + py**2 + pz**2)
    return np.sqrt(np.maximum(m2, 0.0))


def q1_met(events: ak.Array) -> Hist:
    """Q1: histogram of MET."""
    return hist1d(events.MET.pt, bins=50, start=0, stop=200, name="met")


def q2_jet_pt(events: ak.Array) -> Hist:
    """Q2: pt of all jets."""
    return hist1d(events.Jet.pt, bins=50, start=0, stop=200, name="jet_pt")


def q3_jet_pt_central(events: ak.Array) -> Hist:
    """Q3: pt of jets with abs(eta) < 1.0."""
    jets = events.Jet[abs(events.Jet.eta) < 1.0]
    return hist1d(jets.pt, bins=50, start=0, stop=200, name="jet_pt_central")


def q4_met_ge2jets(events: ak.Array) -> Hist:
    """Q4: MET for events with >= 2 jets of pt > 40."""
    njet = ak.num(events.Jet[events.Jet.pt > 40], axis=1)
    return hist1d(events.MET.pt[njet >= 2], bins=50, start=0, stop=200, name="met_2jets")


def q5_met_osmuon_zwindow(events: ak.Array) -> Hist:
    """Q5: MET for events with an opposite-sign muon pair of mass in [60, 120]."""
    mu = events.Muon
    pairs = ak.combinations(mu, 2, fields=["a", "b"])
    opp = pairs.a.charge != pairs.b.charge
    mass = _pair_mass(pairs.a, pairs.b)
    in_window = (mass > 60) & (mass < 120) & opp
    keep = ak.any(in_window, axis=1)
    return hist1d(events.MET.pt[keep], bins=50, start=0, stop=200, name="met_osmuon")


def q6_trijet_pt(events: ak.Array) -> Hist:
    """Q6: pt of the trijet system whose mass is closest to 172.5 (events with >= 3 jets)."""
    has3 = ak.num(events.Jet, axis=1) >= 3
    jets = events.Jet[has3]
    tri = ak.combinations(jets, 3, fields=["a", "b", "c"])
    px = tri.a.pt * np.cos(tri.a.phi) + tri.b.pt * np.cos(tri.b.phi) + tri.c.pt * np.cos(tri.c.phi)
    py = tri.a.pt * np.sin(tri.a.phi) + tri.b.pt * np.sin(tri.b.phi) + tri.c.pt * np.sin(tri.c.phi)
    pz = tri.a.pt * np.sinh(tri.a.eta) + tri.b.pt * np.sinh(tri.b.eta) + tri.c.pt * np.sinh(tri.c.eta)
    e = (
        np.sqrt(tri.a.pt**2 * np.cosh(tri.a.eta) ** 2 + tri.a.mass**2)
        + np.sqrt(tri.b.pt**2 * np.cosh(tri.b.eta) ** 2 + tri.b.mass**2)
        + np.sqrt(tri.c.pt**2 * np.cosh(tri.c.eta) ** 2 + tri.c.mass**2)
    )
    mass = np.sqrt(np.maximum(e**2 - (px**2 + py**2 + pz**2), 0.0))
    tri_pt = np.sqrt(px**2 + py**2)
    best = ak.argmin(abs(mass - 172.5), axis=1, keepdims=True)
    chosen = ak.flatten(tri_pt[best])
    return hist1d(chosen, bins=50, start=0, stop=300, name="trijet_pt")


def q7_sum_isolated_jet_pt(events: ak.Array) -> Hist:
    """Q7: scalar sum of pt of jets (pt > 30) not within dR 0.4 of a lepton (pt > 10)."""
    jets = events.Jet[events.Jet.pt > 30]
    leptons = ak.concatenate(
        [events.Muon[events.Muon.pt > 10], events.Electron[events.Electron.pt > 10]], axis=1
    )
    j, lp = ak.unzip(ak.cartesian([jets, leptons], nested=True))
    dr = _delta_r(j.eta, j.phi, lp.eta, lp.phi)
    isolated = ak.all(dr > 0.4, axis=2)
    # events with zero leptons: all jets isolated
    isolated = ak.fill_none(isolated, True)
    iso_jets = jets[isolated]
    ht = ak.sum(iso_jets.pt, axis=1)
    return hist1d(ht, bins=50, start=0, stop=400, name="ht_isolated")


def q8_mt_met_lepton(events: ak.Array) -> Hist:
    """Q8: events with >= 3 leptons and an OSSF pair; transverse mass of MET + the highest-pt
    lepton not in the OSSF pair whose mass is closest to 91.2."""
    muons = ak.with_field(events.Muon, ak.zeros_like(events.Muon.pt, dtype=np.int64), "flavor")
    eles = ak.with_field(events.Electron, ak.ones_like(events.Electron.pt, dtype=np.int64), "flavor")
    lep = ak.concatenate([muons, eles], axis=1)
    lep = lep[ak.argsort(lep.pt, axis=1, ascending=False)]

    n_lep = ak.num(lep, axis=1)
    mask3 = n_lep >= 3
    lep = lep[mask3]
    met = events.MET[mask3]

    idx = ak.local_index(lep, axis=1)
    pairs = ak.combinations(ak.zip({"lep": lep, "i": idx}), 2, fields=["a", "b"])
    ossf = (pairs.a.lep.charge != pairs.b.lep.charge) & (pairs.a.lep.flavor == pairs.b.lep.flavor)
    mass = _pair_mass(pairs.a.lep, pairs.b.lep)
    mass = ak.where(ossf, mass, np.inf)
    has_ossf = ak.any(ossf, axis=1)

    best = ak.argmin(abs(mass - 91.2), axis=1, keepdims=True)
    pair_i = ak.flatten(pairs.a.i[best])
    pair_j = ak.flatten(pairs.b.i[best])

    not_in_pair = (idx != pair_i) & (idx != pair_j)
    others = lep[not_in_pair]
    lead_other = ak.firsts(others)  # already pt-sorted

    keep = has_ossf & (ak.num(others, axis=1) >= 1)
    met_k = met[keep]
    lead_k = lead_other[keep]
    dphi = _delta_phi(lead_k.phi, met_k.phi)
    mt = np.sqrt(2 * lead_k.pt * met_k.pt * (1 - np.cos(dphi)))
    return hist1d(mt, bins=50, start=0, stop=200, name="mt")


ADL_QUERIES = {
    "q1": q1_met,
    "q2": q2_jet_pt,
    "q3": q3_jet_pt_central,
    "q4": q4_met_ge2jets,
    "q5": q5_met_osmuon_zwindow,
    "q6": q6_trijet_pt,
    "q7": q7_sum_isolated_jet_pt,
    "q8": q8_mt_met_lepton,
}
