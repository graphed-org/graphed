"""Over-touch protection (plan M5 / A.3): projection must read EXACTLY the necessary columns and
NEVER more — the dask-awkward bug this milestone exists to avoid. Every assertion below pins the
*exact* read set and explicitly checks that unused sibling columns / collections are absent.
"""

from __future__ import annotations

import numpy as np
import pytest
from graphed import Session
from graphed_corpus import make_events

from graphed_awkward import AwkwardBackend, from_awkward, gak
from graphed_awkward.projection import project

# the synthetic record's 20 leaf columns, grouped by collection
ALL_COLUMNS = {
    f"{c}.{f}"
    for c, fs in {
        "Muon": ["pt", "eta", "phi", "charge", "mass"],
        "Electron": ["pt", "eta", "phi", "charge", "mass"],
        "Jet": ["pt", "eta", "phi", "mass", "btag"],
        "Photon": ["pt", "eta", "phi"],
        "MET": ["pt", "phi"],
    }.items()
    for f in fs
}


def _read(build) -> frozenset[str]:
    s = Session(AwkwardBackend())
    events = from_awkward(s, "events", make_events(n_events=200))
    return project(build(events)).columns_for("events")


def test_single_column_reads_exactly_that_column() -> None:
    read = _read(lambda e: e.Jet.pt)
    assert read == frozenset({"Jet.pt"})
    # nothing else — no sibling jet columns, no other collections
    assert not (read & (ALL_COLUMNS - {"Jet.pt"}))


def test_filter_does_not_overtouch_sibling_jet_columns() -> None:
    # jets[jets.pt > 30].eta needs ONLY Jet.pt (the mask) and Jet.eta (the result) — NOT phi/mass/btag
    read = _read(lambda e: e.Jet[e.Jet.pt > 30].eta)
    assert read == frozenset({"Jet.pt", "Jet.eta"})
    assert "Jet.phi" not in read and "Jet.mass" not in read and "Jet.btag" not in read


def test_one_collection_does_not_pull_other_collections() -> None:
    read = _read(lambda e: gak.sum(e.Jet.pt, axis=1))
    assert read == frozenset({"Jet.pt"})
    assert not any(c.startswith(("Muon.", "Electron.", "Photon.", "MET.")) for c in read)


def test_met_pt_does_not_touch_met_phi() -> None:
    read = _read(lambda e: e.MET.pt)
    assert read == frozenset({"MET.pt"})
    assert "MET.phi" not in read


def test_dimuon_does_not_touch_unused_muon_mass_or_other_collections() -> None:
    def dimuon(e: object) -> object:
        pairs = gak.combinations(e.Muon, 2, fields=["a", "b"])
        a, b = pairs.a, pairs.b
        mass = np.sqrt(2 * a.pt * b.pt * (np.cosh(a.eta - b.eta) - np.cos(a.phi - b.phi)))
        keep = gak.any((mass > 60) & (mass < 120) & (a.charge != b.charge), axis=1)
        return e.MET.pt[keep]

    read = _read(dimuon)
    assert read == frozenset({"Muon.pt", "Muon.eta", "Muon.phi", "Muon.charge", "MET.pt"})
    assert "Muon.mass" not in read  # the mass formula here never uses Muon.mass
    assert not any(c.startswith(("Electron.", "Jet.", "Photon.")) for c in read)


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda e: e.Muon.pt + e.Muon.eta, {"Muon.pt", "Muon.eta"}),
        (lambda e: e.Jet.btag, {"Jet.btag"}),
        (lambda e: gak.num(e.Electron, axis=1), set()),  # multiplicity touches structure, not leaf data
    ],
)
def test_read_set_is_minimal(build, expected: set[str]) -> None:
    read = _read(build)
    assert read == frozenset(expected)
    assert not (read - frozenset(expected)), f"over-touched: {sorted(read - frozenset(expected))}"
