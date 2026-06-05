"""Necessary-buffer (column) projection via the reporting typetracer (plan M5)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from graphed import ProjectionError, Session
from graphed_corpus import make_events

from graphed_awkward import AwkwardBackend, from_awkward, gak
from graphed_awkward.projection import project


def _dimuon(s: Session) -> object:
    events = from_awkward(s, "events", make_events(n_events=300))
    mu = events.Muon
    pairs = gak.combinations(mu, 2, fields=["a", "b"])
    a, b = pairs.a, pairs.b
    mass = np.sqrt(2 * a.pt * b.pt * (np.cosh(a.eta - b.eta) - np.cos(a.phi - b.phi)))
    keep = gak.any((mass > 60) & (mass < 120) & (a.charge != b.charge), axis=1)
    return events.MET.pt[keep]


def test_projects_to_only_the_muon_branches_used() -> None:
    s = Session(AwkwardBackend())
    out = _dimuon(s)
    proj = project(out)
    assert proj.columns_for("events") == frozenset(
        {"Muon.pt", "Muon.eta", "Muon.phi", "Muon.charge", "MET.pt"}
    )


def test_projected_reads_dramatically_fewer_columns() -> None:
    # the synthetic record has 20 leaf columns; the dimuon analysis reads only 5. (Byte-level
    # measurement against a real NanoAOD file is a tracked follow-up; columns are the proxy here.)
    s = Session(AwkwardBackend())
    out = _dimuon(s)
    read = project(out).total_columns()
    assert read == 5
    assert read * 3 < 20  # a >3x reduction in columns transferred


def test_projection_is_metadata_only_and_unchanged_after_reduction() -> None:
    # projection replays the recorded ops (which a fused stage contains unchanged), so reducing the
    # graphed-core store must not change the projected columns.
    s = Session(AwkwardBackend())
    out = _dimuon(s)
    before = project(out).read_columns
    s._store.reduce()  # M4 stage fusion on the underlying store
    after = project(out).read_columns
    assert before == after


@pytest.mark.parametrize("policy", ["pass", "warn", "raise"])
def test_on_fail_policy_on_opaque_op(policy: str) -> None:
    s = Session(AwkwardBackend())
    events = from_awkward(s, "events", make_events(n_events=100))
    opaque = events.Muon.pt.map(lambda a: a, name="blackbox")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if policy == "raise":
            with pytest.raises(ProjectionError):
                project(opaque, on_fail="raise")
            return
        proj = project(opaque, on_fail=policy)
    if policy == "warn":
        assert len(w) == 1
        assert len(proj.columns_for("events")) == 20  # conservative: all columns
    else:  # pass
        assert len(w) == 0
        assert "Muon.pt" in proj.columns_for("events")
