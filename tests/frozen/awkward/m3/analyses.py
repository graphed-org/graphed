"""The corpus analyses (M0.5) re-expressed against the graphed frontend + gak namespace (M3).

These mirror graphed_corpus/analyses/adl.py op-for-op so that recording them exercises the Required
Operations Catalog through graphed. Each returns the final Array of the recorded graph.
"""

from __future__ import annotations

import numpy as np
from graphed import Array, Session

from graphed_awkward import gak


def _delta_phi(a: Array, b: Array) -> Array:
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def _delta_r(eta1: Array, phi1: Array, eta2: Array, phi2: Array) -> Array:
    return np.hypot(eta1 - eta2, _delta_phi(phi1, phi2))


def _pair_mass(o1: Array, o2: Array) -> Array:
    px = o1.pt * np.cos(o1.phi) + o2.pt * np.cos(o2.phi)
    py = o1.pt * np.sin(o1.phi) + o2.pt * np.sin(o2.phi)
    pz = o1.pt * np.sinh(o1.eta) + o2.pt * np.sinh(o2.eta)
    e = np.sqrt(o1.pt**2 * np.cosh(o1.eta) ** 2 + o1.mass**2) + np.sqrt(
        o2.pt**2 * np.cosh(o2.eta) ** 2 + o2.mass**2
    )
    return np.sqrt(np.maximum(e**2 - (px**2 + py**2 + pz**2), 0.0))


def q1(events: Array) -> Array:
    return events.MET.pt


def q2(events: Array) -> Array:
    return events.Jet.pt


def q3(events: Array) -> Array:
    return events.Jet[abs(events.Jet.eta) < 1.0].pt


def q4(events: Array) -> Array:
    njet = gak.num(events.Jet[events.Jet.pt > 40], axis=1)
    return events.MET.pt[njet >= 2]


def q5(events: Array) -> Array:
    mu = events.Muon
    pairs = gak.combinations(mu, 2, fields=["a", "b"])
    opp = pairs.a.charge != pairs.b.charge
    mass = _pair_mass(pairs.a, pairs.b)
    keep = gak.any((mass > 60) & (mass < 120) & opp, axis=1)
    return events.MET.pt[keep]


def q6(events: Array) -> Array:
    jets = events.Jet[gak.num(events.Jet, axis=1) >= 3]
    tri = gak.combinations(jets, 3, fields=["a", "b", "c"])
    a, b, c = tri.a, tri.b, tri.c
    px = a.pt * np.cos(a.phi) + b.pt * np.cos(b.phi) + c.pt * np.cos(c.phi)
    py = a.pt * np.sin(a.phi) + b.pt * np.sin(b.phi) + c.pt * np.sin(c.phi)
    tri_pt = np.sqrt(px**2 + py**2)
    pz = a.pt * np.sinh(a.eta) + b.pt * np.sinh(b.eta) + c.pt * np.sinh(c.eta)
    e = (
        np.sqrt(a.pt**2 * np.cosh(a.eta) ** 2 + a.mass**2)
        + np.sqrt(b.pt**2 * np.cosh(b.eta) ** 2 + b.mass**2)
        + np.sqrt(c.pt**2 * np.cosh(c.eta) ** 2 + c.mass**2)
    )
    mass = np.sqrt(np.maximum(e**2 - (px**2 + py**2 + pz**2), 0.0))
    best = gak.argmin(abs(mass - 172.5), axis=1, keepdims=True)
    return gak.flatten(tri_pt[best])


def q7(events: Array) -> Array:
    jets = events.Jet[events.Jet.pt > 30]
    leptons = gak.concatenate(
        [events.Muon[events.Muon.pt > 10], events.Electron[events.Electron.pt > 10]], axis=1
    )
    pair = gak.cartesian([jets, leptons], nested=True)
    j, lp = pair["0"], pair["1"]
    dr = _delta_r(j.eta, j.phi, lp.eta, lp.phi)
    isolated = gak.fill_none(gak.all(dr > 0.4, axis=2), True)
    return gak.sum(jets[isolated].pt, axis=1)


def q8(events: Array) -> Array:
    muons = gak.with_field(events.Muon, gak.zeros_like(events.Muon.pt, dtype="int64"), "flavor")
    eles = gak.with_field(events.Electron, gak.ones_like(events.Electron.pt, dtype="int64"), "flavor")
    lep = gak.concatenate([muons, eles], axis=1)
    lep = lep[gak.argsort(lep.pt, axis=1, ascending=False)]
    lep = lep[gak.num(lep, axis=1) >= 3]
    met = events.MET[gak.num(lep, axis=1) >= 3]
    idx = gak.local_index(lep, axis=1)
    pairs = gak.combinations(gak.zip({"lep": lep, "i": idx}), 2, fields=["a", "b"])
    ossf = (pairs.a.lep.charge != pairs.b.lep.charge) & (pairs.a.lep.flavor == pairs.b.lep.flavor)
    mass = gak.where(ossf, _pair_mass(pairs.a.lep, pairs.b.lep), np.inf)
    best = gak.argmin(abs(mass - 91.2), axis=1, keepdims=True)
    not_in_pair = (idx != gak.flatten(pairs.a.i[best])) & (idx != gak.flatten(pairs.b.i[best]))
    lead = gak.firsts(lep[not_in_pair])
    dphi = _delta_phi(lead.phi, met.phi)
    return np.sqrt(2 * lead.pt * met.pt * (1 - np.cos(dphi)))


def agc_object_selection(events: Array) -> Array:
    """AGC ttbar object-selection slice: >=4 jets pt>25, exactly 1 b-tag (4j1b region) -> HT."""
    jets = events.Jet[events.Jet.pt > 25]
    n_good = gak.num(jets, axis=1)
    n_b = gak.sum(jets.btag > 0.7, axis=1)
    sel = (n_good >= 4) & (n_b == 1)
    return gak.sum(jets[sel].pt, axis=1)


ADL = {"q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5, "q6": q6, "q7": q7, "q8": q8}


def record(fn: object) -> tuple[Session, Array]:
    from graphed_corpus import make_events

    import graphed_awkward as ga

    s = Session(ga.AwkwardBackend())
    events = ga.from_awkward(s, "events", make_events(n_events=400))
    return s, fn(events)  # type: ignore[operator]
