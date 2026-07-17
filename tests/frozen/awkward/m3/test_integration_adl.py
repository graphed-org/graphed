"""ADL benchmark queries 1-8 — end-to-end integration tests (plan M3 / A.8).

Each query is recorded through the graphed frontend + AwkwardBackend, **executed** end-to-end via
`materialize`, histogrammed, and asserted to reproduce the corpus's plain-awkward reference
histogram bit-for-bit on the same events. This is the graded functional ladder (column histogram →
MET cuts → object selection → jet/lepton combinatorics → dilepton+MET → 3-lepton mT): if any ADL
query cannot run and reproduce, M3 is not done.
"""

from __future__ import annotations

import awkward as ak
import pytest
from analyses import ADL
from graphed import Session
from graphed_corpus import make_events
from graphed_corpus.analyses import adl as corpus_adl
from graphed_corpus.histograms import bin_values, hist1d

from graphed_awkward import AwkwardBackend, from_awkward

ADL_NAMES = [f"q{i}" for i in range(1, 9)]


@pytest.fixture(scope="module")
def adl_events() -> ak.Array:
    # the corpus default sample (20k events) so every query — including the rare 3-lepton q8 —
    # produces a non-trivial histogram to compare.
    return make_events(n_events=20000, seed=1234)


@pytest.mark.parametrize("name", ADL_NAMES)
def test_adl_query_runs_end_to_end_and_reproduces_corpus(name: str, adl_events: ak.Array) -> None:
    # 1. record + execute the query through graphed
    s = Session(AwkwardBackend())
    events = from_awkward(s, "events", adl_events)
    values = s.materialize(ADL[name](events))

    # 2. the corpus plain-awkward reference on the same events
    reference = corpus_adl.ADL_QUERIES[name](adl_events)
    axis = reference.axes[0]

    # 3. histogram the graphed output with identical binning and compare bit-for-bit
    produced = hist1d(
        values, bins=axis.size, start=float(axis.edges[0]), stop=float(axis.edges[-1]), name=name
    )
    assert int(sum(bin_values(reference))) > 0, f"{name}: reference histogram is empty (not a real test)"
    assert bin_values(produced) == bin_values(reference), f"{name}: graphed histogram differs from corpus"


def test_all_eight_adl_queries_are_covered() -> None:
    assert set(ADL_NAMES) == {f"q{i}" for i in range(1, 9)}
    assert all(name in ADL for name in ADL_NAMES)
