"""m50 fix-cycle-1 (INT-1) EXTRA witness — the axis-mode/variation member of the
preserve↔histogram content-address contract.

No FROZEN test preserves a §6.2 axis-mode fill, so this manual witness proves the third member of
the INT-1 class (unweighted / multi-weight / **variation**) is closed: the preserve plugin
re-derives H1's DISCRIMINATED ``_fill_chash`` (``spec + "\\x00" + "variation=<json>"``) from the
node's params, so ``build_bundle``'s integrity check accepts the axis-mode fill and ``reproduce``
replays its evaluator-side variation loop bit-for-bit.

Not collected by ``scripts/run-tests.sh`` (``preserve`` maps to ``tests/frozen/preserve`` only) —
run directly::

    .venv/bin/python -m pytest tests/extra/preserve/m50 -q

The construction is guard-free by design: it records ONE external node through the base
``record_external`` over plain array inputs — exactly the technique the frozen m27 multi-weight
test uses. A real ``Histogram.fill(..., variation_axis=True)`` additionally records
``histogram.weight_guard`` externals for which no preserve plugin exists yet (a pre-existing gap,
orthogonal to the content_hash contract), which would block the WHOLE-graph reproduce; that node
family is out of INT-1's scope.
"""

from __future__ import annotations

import json

import awkward as ak
import boost_histogram as bh
import numpy as np
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.preserve import HISTOGRAM_PLUGIN, build_bundle, record_external, reproduce

gh = pytest.importorskip("graphed_histogram")  # optional package; installed in the shared venv

#: fold order, "nominal" first — a weight-only axis-mode loop node carries exactly this set
NODE_LABELS = ("nominal", "wgt_down", "wgt_up")

EVENTS = ak.Array(
    {
        "x": [1.0, 4.0, 7.0, 2.5] * 20,
        "wn": [1.0, 1.0, 1.0, 1.0] * 20,
        "wu": [1.2, 1.2, 1.2, 1.2] * 20,
        "wd": [0.8, 0.8, 0.8, 0.8] * 20,
    }
)


def _axis_spec() -> str:
    """The frontend-declared axis-mode spec: the value axes plus a non-growth ``"variation"``
    StrCategory (declared over the SORTED label set, §6.2(ii))."""
    var_ax = bh.axis.StrCategory(sorted(NODE_LABELS))
    var_ax.__dict__["name"] = "variation"  # the kwarg form is a TypeError; the codec round-trips this
    return gh.spec_of(bh.Histogram(bh.axis.Regular(4, 0.0, 8.0), var_ax, storage=bh.storage.Weight()))


def _eager_axis_reference(spec: str) -> bh.Histogram:
    """The oracle: ``FillEvaluator``'s axis loop is ``h.fill(value, label, weight=block)`` per
    label — the value fixed, the weight block varying across the variation categories."""
    h = gh.zero_of(spec)
    xv = np.asarray(EVENTS.x)
    for label, wcol in zip(NODE_LABELS, (EVENTS.wn, EVENTS.wd, EVENTS.wu), strict=True):
        h.fill(xv, [label] * len(xv), weight=np.asarray(wcol))
    return h


def test_axis_mode_variation_fill_preserves_and_reproduces(tmp_path) -> None:  # type: ignore[no-untyped-def]
    spec = _axis_spec()
    # H1's discriminated node id: the variation set is folded into the content hash
    disc_payload = spec + "\x00" + "variation=" + json.dumps(list(NODE_LABELS))
    disc_hash = gh.content_hash(disc_payload)

    s = Session(AwkwardBackend())
    g = from_awkward(s, "events", EVENTS)
    params = {
        "spec": spec,
        "n_axes": 1,
        "weighted": True,
        "sampled": False,
        "variation": json.dumps(list(NODE_LABELS)),
    }
    # inputs = [value, one weight block per label in NODE_LABELS order] (the axis-loop input layout)
    fill = record_external(s, HISTOGRAM_PLUGIN, disc_payload.encode(), [g.x, g.wn, g.wd, g.wu], params=params)
    stored = next(n for n in s._store.nodes() if n["id"] == fill.node_id)
    assert stored["descriptor"]["content_hash"] == disc_hash  # recorded the DISCRIMINATED id

    reference = s.materialize(fill)  # record-time eval through the real FillEvaluator

    # build_bundle would raise "hashes to X not recorded Y" if the plugin re-derived only sha256(spec)
    bundle = build_bundle(tmp_path / "b", session=s, value=fill, datasets={"events": EVENTS}, payloads={})
    entry = next(e for e in bundle.manifest["externals"] if e["kind"] == "histogram")
    payload = bundle.store.get(entry["store"])
    assert entry["content_hash"] == disc_hash
    assert payload is not None
    # the synthesized payload is the DISCRIMINATED canonical (carries the variation), not the bare spec
    assert payload.decode() != spec and "variation=" in payload.decode()
    assert HISTOGRAM_PLUGIN.content_hash(payload) == entry["content_hash"]  # integrity, as build checks

    got = reproduce(bundle)  # the eval-side half: reconstruct the axis loop from params
    assert isinstance(got, bh.Histogram)
    assert any(a.__dict__.get("name") == "variation" for a in got.axes)  # the variation axis survived
    eager = _eager_axis_reference(spec)
    assert np.array_equal(got.view(flow=True)["value"], reference.view(flow=True)["value"])
    assert np.array_equal(got.view(flow=True)["value"], eager.view(flow=True)["value"])
    assert np.array_equal(got.view(flow=True)["variance"], eager.view(flow=True)["variance"])


def test_same_spec_pair_discriminated_and_bare_both_preserve(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Adversarial: the ONE same-spec collision the fix must reconcile. Two multi-weight fills with
    IDENTICAL params but DIFFERENT recorded ids — H1's ``Histogram.fill`` mints the DISCRIMINATED
    ``spec+"\\x00"+"n_weights=2"`` hash, the legacy ``record_external`` over bare spec bytes mints
    ``sha256(spec)``. A params-only synthesize could serve only one; threading the recorded hash lets
    the plugin emit whichever form each node actually recorded, so BOTH pass integrity."""
    spec = gh.spec_of(bh.Histogram(bh.axis.Regular(4, 0.0, 8.0), storage=bh.storage.Weight()))
    params = {"spec": spec, "n_axes": 1, "weighted": True, "n_weights": 2, "sampled": False}
    events = ak.Array(
        {"x": [1.0, 4.0, 7.0, 2.5] * 20, "a": [0.5, 1.0, 2.0, 1.5] * 20, "b": [1.0, 0.5, 1.0, 2.0] * 20}
    )

    disc_payload = spec + "\x00" + "n_weights=2"
    for label, payload_bytes, expect_hash in (
        ("discriminated", disc_payload.encode(), gh.content_hash(disc_payload)),
        ("bare", spec.encode(), gh.content_hash(spec)),
    ):
        s = Session(AwkwardBackend())
        g = from_awkward(s, "events", events)
        fill = record_external(s, HISTOGRAM_PLUGIN, payload_bytes, [g.x, g.a, g.b], params=params)
        stored = next(n for n in s._store.nodes() if n["id"] == fill.node_id)
        assert stored["descriptor"]["content_hash"] == expect_hash, label

        bundle = build_bundle(
            tmp_path / label, session=s, value=fill, datasets={"events": events}, payloads={}
        )
        entry = next(e for e in bundle.manifest["externals"] if e["kind"] == "histogram")
        assert entry["content_hash"] == expect_hash, label
        assert HISTOGRAM_PLUGIN.content_hash(bundle.store.get(entry["store"])) == expect_hash, label

        got = reproduce(bundle)
        eager = bh.Histogram(bh.axis.Regular(4, 0.0, 8.0), storage=bh.storage.Weight())
        eager.fill(np.asarray(events.x), weight=np.asarray(events.a) * np.asarray(events.b))
        assert np.array_equal(got.view(flow=True)["value"], eager.view(flow=True)["value"]), label
