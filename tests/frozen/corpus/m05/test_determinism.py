"""Determinism: the dataset and every analysis are reproducible across runs."""

from __future__ import annotations

import awkward as ak
from conftest import all_fixtures

from graphed_corpus import fingerprint, make_events


def test_dataset_is_deterministic() -> None:
    a = make_events(n_events=2000, seed=7)
    b = make_events(n_events=2000, seed=7)
    assert ak.to_list(a.Muon.pt) == ak.to_list(b.Muon.pt)
    assert ak.to_list(a.Jet.btag) == ak.to_list(b.Jet.btag)


def test_seed_changes_data() -> None:
    a = make_events(n_events=2000, seed=1)
    b = make_events(n_events=2000, seed=2)
    assert ak.to_list(a.MET.pt) != ak.to_list(b.MET.pt)


def test_analyses_are_deterministic(events: ak.Array) -> None:
    reg = all_fixtures()
    for name, fn in reg.items():
        assert fingerprint(fn(events)) == fingerprint(fn(events)), f"{name} not deterministic"
