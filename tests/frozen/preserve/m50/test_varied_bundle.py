"""m50 §9.2 — one bundle preserving N variation universes over the value/weight/spec triple.

A varied ``build_bundle`` (a ``Varied`` weight over the triple ``build_bundle`` already takes) yields
a bundle whose ``reproduce`` answers ``{label: array}`` — each universe's histogram bit-for-bit vs its
build-time counts (m9's ``np.array_equal`` comparison). An UNVARIED bundle keeps today's singular shape:
``reproduce`` returns a BARE array. The manifest's ``format_version`` is the bumped value for the
varied bundle and still ``1`` for the unvaried control; without both, the bump is invisible.
``inspect`` lists the labels without executing.

Freeze-time spellings this suite pins (the implementer inherits them):
* the varied path is triggered by a ``Varied`` ``value``/``weight`` — no new keyword;
* the bumped ``format_version`` value is ``2``;
* the per-label manifest key is ``manifest["analysis"]["variations"]``, keyed by label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest
from graphed_corpus import make_events

from graphed import Session, labels, universe, vary
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.preserve import UnresolvedPayload, build_bundle, inspect, reproduce

HIST = {"name": "w", "bins": 10, "lo": 0.0, "hi": 200.0}
_BUMPED_FORMAT_VERSION = 2
_MANIFEST_LABEL_KEY = "variations"  # under manifest["analysis"]
_STABLE_DECIMALS = 6  # mirrors preserve._histogram's cross-platform rounding


def _counts(value: Any, weight: Any) -> np.ndarray:
    v = np.asarray(ak.to_numpy(ak.Array(value)), dtype="float64")
    w = np.asarray(ak.to_numpy(ak.Array(weight)), dtype="float64")
    c, _ = np.histogram(v, bins=HIST["bins"], range=(HIST["lo"], HIST["hi"]), weights=w)
    return np.round(c, _STABLE_DECIMALS)


def _record() -> tuple[Session, Any, Any, Any]:
    """A recorded per-event value and a per-event weight (both flat, so ``_histogram`` can run)."""
    events = make_events(n_events=200, seed=50)
    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", events)
    value = gak.sum(ev.Jet.pt, axis=1)
    weight = ev.MET.pt
    return s, value, weight, events


def _build_varied(root: Path) -> tuple[Any, dict[str, np.ndarray], tuple[str, ...]]:
    s, value, weight_nom, events = _record()
    weight = vary(weight_nom, "sf", up=weight_nom * 1.1, down=weight_nom * 0.9)
    lbls = labels(weight)
    build_time = {
        label: _counts(s.materialize(value), s.materialize(universe(weight, label))) for label in lbls
    }
    bundle = build_bundle(
        root / "varied", session=s, value=value, weight=weight, datasets={"events": events}, histogram=HIST
    )
    return bundle, build_time, lbls


def _build_unvaried(root: Path) -> tuple[Any, np.ndarray]:
    s, value, weight, events = _record()
    reference = _counts(s.materialize(value), s.materialize(weight))
    bundle = build_bundle(
        root / "unvaried", session=s, value=value, weight=weight, datasets={"events": events}, histogram=HIST
    )
    return bundle, reference


def test_reproduce_returns_a_histogram_per_label(tmp_path: Path) -> None:
    bundle, build_time, lbls = _build_varied(tmp_path)
    out = reproduce(bundle)
    assert isinstance(out, dict) and set(out) == set(lbls)
    # non-vacuous: universes actually fill AND differ, so a mapping that reuses one array is red
    assert build_time["nominal"].sum() > 0
    assert not np.array_equal(build_time["sf_up"], build_time["sf_down"])
    for label in lbls:
        assert np.array_equal(out[label], build_time[label]), f"universe {label!r} not bit-for-bit"


def test_unvaried_bundle_reproduces_a_bare_array(tmp_path: Path) -> None:
    bundle, reference = _build_unvaried(tmp_path)
    out = reproduce(bundle)
    assert isinstance(out, np.ndarray)  # backward-compat: no dict for an unvaried bundle
    assert reference.sum() > 0
    assert np.array_equal(out, reference)


def test_format_version_bumps_only_for_the_varied_bundle(tmp_path: Path) -> None:
    varied, _, _ = _build_varied(tmp_path)
    unvaried, _ = _build_unvaried(tmp_path)
    assert varied.manifest["format_version"] == _BUMPED_FORMAT_VERSION
    assert unvaried.manifest["format_version"] == 1


def test_manifest_carries_a_per_label_output_map(tmp_path: Path) -> None:
    bundle, _, lbls = _build_varied(tmp_path)
    assert set(bundle.manifest["analysis"][_MANIFEST_LABEL_KEY]) == set(lbls)


def test_inspect_lists_labels_without_executing(tmp_path: Path) -> None:
    bundle, _, lbls = _build_varied(tmp_path)
    text = inspect(bundle)
    for label in lbls:
        assert label in text
    # strip the data blob execution needs; inspect still renders, reproduce fails loudly
    (bundle.root / "store" / "objects" / bundle.manifest["sources"]["events"]).unlink()
    assert all(label in inspect(bundle) for label in lbls)
    with pytest.raises(UnresolvedPayload):
        reproduce(bundle)
