"""M10 — buffer-level projection (finding A.3): keep the shape/data distinction awkward reports.

The sharpest pin: a count-only analysis projects to `{collection: OFFSETS}` — non-empty, truthful,
servable from a counter branch (TTree) or an index column (RNTuple) — where the column-level M5
`project` necessarily reports the empty set. And the collapse `to_projection()` reproduces the
frozen M5 column view exactly, so the two granularities can never drift apart.
"""

from __future__ import annotations

import numpy as np
import pytest
from graphed import BufferNeed, BufferProjection, ProjectionError, Session
from graphed_corpus import make_events

from graphed_awkward import AwkwardBackend, from_awkward, gak, project, project_buffers


def _session():
    s = Session(AwkwardBackend())
    return from_awkward(s, "events", make_events(n_events=200))


def _buffers(build) -> dict[str, BufferNeed]:
    return project_buffers(build(_session())).buffers_for("events")


def test_count_only_reports_offsets_not_the_empty_set() -> None:
    # THE case column-level projection cannot express: a multiplicity needs the Electron list
    # structure (offsets / counter branch) and no leaf data whatsoever.
    needs = _buffers(lambda e: gak.num(e.Electron, axis=1))
    assert needs == {"Electron": BufferNeed.OFFSETS}
    # the column-level view of the same graph is empty (the frozen M5 pin) — under-specified
    arr = _session()
    assert project(gak.num(arr.Electron, axis=1)).columns_for("events") == frozenset()


def test_leaf_read_is_data() -> None:
    needs = _buffers(lambda e: e.Jet.pt)
    assert needs["Jet.pt"] is BufferNeed.DATA
    assert not [c for c, n in needs.items() if n is BufferNeed.DATA and c != "Jet.pt"]


def test_data_implies_structure_no_redundant_offsets_entry() -> None:
    # reading Jet.pt brings the jet offsets along; no separate {Jet: OFFSETS} entry
    needs = _buffers(lambda e: gak.sum(e.Jet.pt, axis=1))
    assert needs == {"Jet.pt": BufferNeed.DATA}


def test_mixed_count_and_data() -> None:
    # cut on jet multiplicity, then read muon pt: Jet contributes structure only
    def build(e):
        return e.Muon.pt[gak.num(e.Jet, axis=1) >= 2]

    needs = _buffers(build)
    assert needs["Muon.pt"] is BufferNeed.DATA
    assert needs["Jet"] is BufferNeed.OFFSETS
    assert not any(c.startswith("Jet.") for c in needs)  # no Jet leaf data


def test_no_overtouch_at_buffer_level() -> None:
    # the M5 no-overtouch guarantee holds at the finer granularity too
    needs = _buffers(lambda e: e.Jet[e.Jet.pt > 30].eta)
    data = {c for c, n in needs.items() if n is BufferNeed.DATA}
    assert data == {"Jet.pt", "Jet.eta"}
    assert not any(c.startswith(("Muon", "Electron", "Photon", "MET")) for c in needs)


def test_to_projection_collapses_exactly_to_the_m5_column_view() -> None:
    builds = [
        lambda e: e.Jet.pt,
        lambda e: e.Jet[e.Jet.pt > 30].eta,
        lambda e: gak.sum(e.Jet.pt, axis=1),
        lambda e: gak.num(e.Electron, axis=1),
        lambda e: e.Muon.pt + e.Muon.eta,
        lambda e: e.MET.pt,
    ]
    for build in builds:
        a1 = _session()
        a2 = _session()
        buf = project_buffers(build(a1))
        col = project(build(a2))
        assert buf.to_projection().read_columns == col.read_columns


def test_dimuon_buffer_needs() -> None:
    def dimuon(e):
        pairs = gak.combinations(e.Muon, 2, fields=["a", "b"])
        a, b = pairs.a, pairs.b
        mass = np.sqrt(2 * a.pt * b.pt * (np.cosh(a.eta - b.eta) - np.cos(a.phi - b.phi)))
        keep = gak.any((mass > 60) & (mass < 120) & (a.charge != b.charge), axis=1)
        return e.MET.pt[keep]

    needs = _buffers(dimuon)
    data = {c for c, n in needs.items() if n is BufferNeed.DATA}
    assert data == {"Muon.pt", "Muon.eta", "Muon.phi", "Muon.charge", "MET.pt"}
    assert "Muon.mass" not in needs


def test_opaque_map_is_conservative_under_warn_and_raises_under_raise() -> None:
    arr = _session()
    mapped = arr.Jet.pt.map(lambda x: x)
    with pytest.raises(ProjectionError):
        project_buffers(mapped)
    with pytest.warns(UserWarning):
        buf = project_buffers(mapped, on_fail="warn")
    # conservative: every leaf column is read as DATA
    needs = buf.buffers_for("events")
    assert needs and all(n is BufferNeed.DATA for n in needs.values())


def test_result_type_round_trips() -> None:
    arr = _session()
    buf = project_buffers(gak.num(arr.Electron, axis=1))
    assert isinstance(buf, BufferProjection)
    assert buf.offsets_only_for("events") == frozenset({"Electron"})
    assert buf.columns_for("events") == frozenset()
