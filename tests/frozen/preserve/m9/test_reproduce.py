"""M9 acceptance — a bundle reproduces its histogram bit-for-bit from references alone.

Plan M9: "A Preservation Bundle built for the AGC ttbar slice ... reproduces its histograms
bit-for-bit." Here the bundle is built for the reduced AGC ttbar slice (ONNX model + correctionlib
scale factors + JES/weight systematics) and ``reproduce`` (which runs purely off the content-addressed
store) must equal the in-process build-time result exactly.
"""

from __future__ import annotations

from typing import Any

import agc
import numpy as np

from graphed_preserve import reproduce


def test_reproduce_matches_build_bit_for_bit(agc_bundle: tuple[Any, np.ndarray]) -> None:
    bundle, reference = agc_bundle
    out = reproduce(bundle)
    assert int(reference.sum()) > 0, "non-vacuous: the AGC slice must actually fill the histogram"
    assert np.array_equal(out, reference), "reproduce (from references) != build-time materialize"


def test_bundle_carries_real_hep_standard_payloads(agc_bundle: tuple[Any, np.ndarray]) -> None:
    bundle, _ = agc_bundle
    kinds = {e["kind"] for e in bundle.manifest["externals"]}
    assert kinds == {"correctionlib", "onnx_model"}  # corrections + inference, content-addressed
    assert bundle.manifest["opaque_nodes"] == []  # nothing cloudpickled -> no preservation risk


def test_bundle_is_a_self_contained_directory(agc_bundle: tuple[Any, np.ndarray]) -> None:
    bundle, _ = agc_bundle
    assert (bundle.root / "manifest.json").is_file()
    # every referenced blob (IR, datasets, payloads, sourcemap) lives in the bundle's own store
    objects = {p.name for p in (bundle.root / "store" / "objects").iterdir()}
    m = bundle.manifest
    referenced = {m["analysis"]["ir"], m["provenance"], *m["sources"].values()}
    referenced |= {e["store"] for e in m["externals"]}
    assert referenced <= objects


def test_reproduce_is_deterministic(agc_bundle: tuple[Any, np.ndarray]) -> None:
    bundle, _ = agc_bundle
    assert np.array_equal(reproduce(bundle), reproduce(bundle))


def test_systematic_variations_change_the_result(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # weight systematic (correctionlib up/down) and kinematic systematic (JES) both move the histogram
    nominal, _ = agc.build_agc(
        tmp_path / "nom", config={"systematic": "nominal", "jes_factor": 1.0, "min_btag": 1}
    )
    up, _ = agc.build_agc(tmp_path / "up", config={"systematic": "up", "jes_factor": 1.0, "min_btag": 1})
    jes, _ = agc.build_agc(
        tmp_path / "jes", config={"systematic": "nominal", "jes_factor": 1.05, "min_btag": 1}
    )
    rn, ru, rj = reproduce(nominal), reproduce(up), reproduce(jes)
    assert not np.array_equal(rn, ru), "correctionlib up variation must change the weighted histogram"
    assert not np.array_equal(rn, rj), "JES variation must change selection/observable"
