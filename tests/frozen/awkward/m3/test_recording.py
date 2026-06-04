"""All corpus analyses record metadata-only with correct typetracer forms (plan M3).

Any catalog op the frontend cannot record is a failure, not a silent gap — hence the full ADL
1-8 + AGC object-selection ladder is parametrized here.
"""

from __future__ import annotations

import numpy as np
import pytest
from analyses import ADL, agc_object_selection, record
from graphed import Session
from graphed_corpus import make_events

from graphed_awkward import AwkwardBackend, from_awkward, gak

ALL = {**ADL, "agc": agc_object_selection}


@pytest.mark.parametrize("name", sorted(ALL))
def test_analysis_records_metadata_only(name: str) -> None:
    s, out = record(ALL[name])
    assert s.node_count() > 0
    form = s.form(out)
    assert form.is_typetracer, f"{name} form must be a metadata-only typetracer"
    assert "*" in form.describe()  # a real awkward type was inferred


def test_dimuon_smoke_records_correct_forms() -> None:
    s = Session(AwkwardBackend())
    events = from_awkward(s, "events", make_events(n_events=300))
    pairs = gak.combinations(events.Muon, 2, fields=["a", "b"])
    a, b = pairs.a, pairs.b
    mass = np.sqrt(2 * a.pt * b.pt * (np.cosh(a.eta - b.eta) - np.cos(a.phi - b.phi)))
    inwin = (mass > 60) & (mass < 120)
    # forms correct at each step, all metadata-only
    assert "var * float64" in s.form(mass).describe()
    assert "var * bool" in s.form(inwin).describe()
    assert s.form(mass).is_typetracer


def test_repeated_subexpression_interns() -> None:
    s = Session(AwkwardBackend())
    events = from_awkward(s, "events", make_events(n_events=200))
    first = events.Muon.pt
    n = s.node_count()
    second = events.Muon.pt  # identical field chain -> interns, adds zero nodes
    assert second.node_id == first.node_id
    assert s.node_count() == n
