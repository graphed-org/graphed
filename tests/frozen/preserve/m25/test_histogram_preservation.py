"""M25: histogram fills are PRESERVABLE Externals (closing the M9 <- M23 integration gap).

Since M23, a histogram fill IS an External node whose content hash is the SHA-256 of its
canonical axes/storage spec — exactly the node family the bundle's `payloads` serves. The
`histogram` plugin's payload is that spec itself (tiny, declarative, SYNTHESIZED at build time
from the node's own params — callers supply nothing), its evaluator reconstructs the fill, and
`build_bundle` accepts HISTOGRAM-TERMINAL analyses directly: no (value, weight, spec) triple —
the preserved IR ends at the fill, and `reproduce()` returns the histogram itself, bit for bit.
"""

from __future__ import annotations

import awkward as ak
import boost_histogram as bh
import graphed_histogram as gh
import numpy as np
import pytest
from graphed import Session
from graphed_awkward import AwkwardBackend, from_awkward, gak

from graphed_preserve import (
    HISTOGRAM_PLUGIN,
    PreserveError,
    build_bundle,
    get_plugin,
    inspect,
    reproduce,
    validate_plugin,
)

EVENTS = ak.Array({"x": [[10.0, 40.0], [], [70.0, 25.0, 90.0]] * 30})


def _record() -> tuple[Session, object, bh.Histogram]:
    s = Session(AwkwardBackend())
    g = from_awkward(s, "events", EVENTS)
    h = gh.boost.Histogram(bh.axis.Regular(12, 0.0, 100.0), storage=bh.storage.Int64())
    h.fill(gak.flatten(g.x * 1.5, axis=1))
    eager = bh.Histogram(bh.axis.Regular(12, 0.0, 100.0), storage=bh.storage.Int64())
    eager.fill(ak.flatten(EVENTS.x * 1.5, axis=None))
    return s, h.fill_nodes()[0], eager


def test_the_histogram_plugin_is_registered_and_its_hash_validates() -> None:
    assert get_plugin("histogram") is HISTOGRAM_PLUGIN
    validate_plugin(HISTOGRAM_PLUGIN)  # deterministic across processes, non-vacuous


def test_payload_hash_coheres_with_the_node_identity() -> None:
    h = gh.boost.Histogram(bh.axis.Regular(7, 0.0, 1.0))
    spec = gh.spec_of(h)
    # the SAME hash on both sides of the seam: the node's descriptor and the plugin's payload
    assert HISTOGRAM_PLUGIN.content_hash(spec.encode()) == gh.content_hash(spec)


def test_histogram_terminal_bundle_reproduces_bit_for_bit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    s, fill, eager = _record()
    bundle = build_bundle(
        tmp_path / "bundle", session=s, value=fill, datasets={"events": EVENTS}, payloads={}
    )  # no weight, no histogram spec, no payload bytes: the fill IS the analysis terminal
    entry = next(e for e in bundle.manifest["externals"] if e["kind"] == "histogram")
    assert entry["content_hash"].startswith("sha256:")  # synthesized + integrity-checked at build

    got = reproduce(bundle)
    assert isinstance(got, bh.Histogram)
    assert np.array_equal(got.values(flow=True), eager.values(flow=True))

    rendered = inspect(bundle)  # no execution
    assert "histogram" in rendered and "histogram: None" in rendered
    assert "PRESERVATION RISK" not in rendered  # the fill is durable, not opaque

    again = build_bundle(
        tmp_path / "bundle2",
        session=_record()[0],
        value=_record()[1],
        datasets={"events": EVENTS},
        payloads={},
    )
    assert again.fingerprint() == bundle.fingerprint()  # content-addressed determinism


def test_the_triple_path_must_stay_coherent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    s, fill, _ = _record()
    with pytest.raises(PreserveError, match="together"):
        build_bundle(
            tmp_path / "b",
            session=s,
            value=fill,
            weight=fill,
            datasets={"events": EVENTS},
            payloads={},
        )  # a weight without a histogram spec (or vice versa) is incoherent
