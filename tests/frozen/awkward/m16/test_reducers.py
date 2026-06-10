"""M16.3: the 13 missing dask-awkward reducers, evaluated exactly as awkward (parity plan P0).

Every case materializes the deferred reduction and compares against the same ak.* call on the
raw array — no hand-written expectations to drift. Typetracer form inference must succeed at
record time for every kind (metadata only)."""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from m16_helpers import session_events

from graphed_awkward import gak, project

ONE_INPUT = ["sum", "any", "all", "count", "count_nonzero", "min", "max", "prod", "mean", "std", "var", "ptp"]


@pytest.mark.parametrize("kind", ONE_INPUT)
@pytest.mark.parametrize("axis", [None, 1])
def test_reducer_matches_awkward(kind: str, axis: int | None) -> None:
    s, g, raw = session_events()
    deferred = getattr(gak, kind)(g.Jet.pt, axis=axis)
    assert s.form(deferred).is_typetracer
    got = s.materialize(deferred)
    ref = getattr(ak, kind)(raw.Jet.pt, axis=axis)
    if axis is None:
        assert float(np.asarray(got)) == pytest.approx(float(ref), nan_ok=True)
    else:
        # same awkward kernel on the same data: exactly equal (None/NaN included)
        assert ak.array_equal(
            ak.Array(got), ak.Array([ref] if ref is None else ref), equal_nan=True
        ) or ak.to_list(got) == ak.to_list(ref)


def test_std_var_ddof() -> None:
    s, g, raw = session_events()
    got = s.materialize(gak.std(g.met, axis=None, ddof=1))
    assert float(np.asarray(got)) == pytest.approx(float(ak.std(raw.met, axis=None, ddof=1)))
    got_v = s.materialize(gak.var(g.met, axis=None, ddof=1))
    assert float(np.asarray(got_v)) == pytest.approx(float(ak.var(raw.met, axis=None, ddof=1)))


def test_moment_matches_awkward() -> None:
    s, g, raw = session_events()
    got = s.materialize(gak.moment(g.Jet.pt, 2, axis=1))
    assert ak.array_equal(ak.Array(got), ak.moment(raw.Jet.pt, 2, axis=1), equal_nan=True)


def test_softmax_matches_awkward() -> None:
    s, g, raw = session_events()
    got = s.materialize(gak.softmax(g.Jet.pt, axis=1))
    assert ak.array_equal(ak.Array(got), ak.softmax(raw.Jet.pt, axis=1), equal_nan=True)


def test_two_array_reducers_match_awkward() -> None:
    s, g, raw = session_events()
    got = s.materialize(gak.corr(g.Jet.pt, g.Jet.eta, axis=1))
    assert ak.array_equal(ak.Array(got), ak.corr(raw.Jet.pt, raw.Jet.eta, axis=1), equal_nan=True)
    got = s.materialize(gak.covar(g.Jet.pt, g.Jet.eta, axis=1))
    assert ak.array_equal(ak.Array(got), ak.covar(raw.Jet.pt, raw.Jet.eta, axis=1), equal_nan=True)
    got = ak.Array(s.materialize(gak.linear_fit(g.Jet.pt, g.Jet.eta, axis=1)))
    ref = ak.linear_fit(raw.Jet.pt, raw.Jet.eta, axis=1)
    assert ak.array_equal(got.intercept, ref.intercept, equal_nan=True)
    assert ak.array_equal(got.slope, ref.slope, equal_nan=True)


def test_projection_tracks_through_new_reducers() -> None:
    _s, g, _ = session_events()
    out = gak.mean(g.Jet.pt, axis=1)
    assert project(out).columns_for("events") == frozenset({"Jet.pt"})
