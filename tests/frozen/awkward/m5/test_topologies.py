"""Column projection through complex topologies (plan M5) — diamond / star / nested shapes where a
column feeds re-converging branches. Touch-tracking must report each column EXACTLY once and never
over-touch a sibling, regardless of how the branches fan out and re-join (these re-convergence cases
were dask-awkward failure points)."""

from __future__ import annotations

from graphed import Session
from graphed_corpus import make_events

from graphed_awkward import AwkwardBackend, from_awkward, gak
from graphed_awkward.projection import project


def _events(s: Session) -> object:
    return from_awkward(s, "events", make_events(n_events=200))


def test_diamond_reads_each_column_exactly_once() -> None:
    # Muon.pt fans out (via `a`) into two branches that re-converge at `out`
    s = Session(AwkwardBackend())
    ev = _events(s)
    a = ev.Muon.pt * 2.0
    left = a + ev.Muon.eta
    right = a - ev.Muon.phi
    out = left + right
    read = project(out).columns_for("events")
    assert read == frozenset({"Muon.pt", "Muon.eta", "Muon.phi"})  # no charge/mass, no other collection


def test_star_one_column_feeds_many_branches() -> None:
    # Muon.pt is the hub of a star: combined with every other muon column, then all re-converge
    s = Session(AwkwardBackend())
    ev = _events(s)
    hub = ev.Muon.pt
    branches = [hub + ev.Muon.eta, hub - ev.Muon.phi, hub * ev.Muon.charge, hub + ev.Muon.mass]
    out = branches[0]
    for b in branches[1:]:
        out = out + b
    read = project(out).columns_for("events")
    assert read == frozenset({"Muon.pt", "Muon.eta", "Muon.phi", "Muon.charge", "Muon.mass"})
    # the hub feeding many branches must not pull in any OTHER collection
    assert not any(c.startswith(("Electron.", "Jet.", "Photon.", "MET.")) for c in read)


def test_nested_diamonds_do_not_overtouch() -> None:
    # stacked diamonds, each apex re-used by two branches
    s = Session(AwkwardBackend())
    ev = _events(s)
    v = ev.Muon.pt
    for _ in range(4):
        v = (v + ev.Muon.eta) - (v - ev.Muon.phi)  # v fans out to two branches that re-converge
    read = project(v).columns_for("events")
    assert read == frozenset({"Muon.pt", "Muon.eta", "Muon.phi"})


def test_diamond_through_a_reduction_boundary() -> None:
    # a branch passes through a reduction (a stage boundary) before re-joining the other
    s = Session(AwkwardBackend())
    ev = _events(s)
    a = ev.Muon.pt + ev.Muon.eta
    reduced = gak.sum(a, axis=1)  # boundary in one branch
    other = gak.sum(ev.Muon.phi, axis=1)
    out = reduced + other
    read = project(out).columns_for("events")
    assert read == frozenset({"Muon.pt", "Muon.eta", "Muon.phi"})
