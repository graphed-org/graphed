"""M9 acceptance — the bundle is self-fingerprinting (no over- or under-sensitivity).

Plan M9: "changing any auxiliary input — a correctionlib JSON, the ONNX model weights, a dataset
file, a config value, or a seed — changes the bundle's top-level content hash; changing nothing
reproduces an identical hash." Review focus: the fingerprint changes on every result-determining
input and NOT on irrelevant ones.
"""

from __future__ import annotations

import agc


def _fp(root, **opts):  # type: ignore[no-untyped-def]
    bundle, _ = agc.build_agc(root, **opts)
    return bundle.fingerprint()


def test_identical_inputs_give_an_identical_fingerprint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # no over-sensitivity: rebuilding the same analysis + inputs yields the same hash
    # (no timestamps / absolute paths / host leak into the manifest)
    assert _fp(tmp_path / "a") == _fp(tmp_path / "b")


def test_changing_the_correctionlib_json_changes_the_fingerprint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert _fp(tmp_path / "a") != _fp(tmp_path / "b", sf_scale=1.10)


def test_changing_the_onnx_weights_changes_the_fingerprint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert _fp(tmp_path / "a") != _fp(tmp_path / "b", onnx_weight=0.09)


def test_changing_the_dataset_changes_the_fingerprint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert _fp(tmp_path / "a") != _fp(tmp_path / "b", seed=agc.SEED + 1)


def test_changing_a_config_value_changes_the_fingerprint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    base = {"systematic": "nominal", "jes_factor": 1.0, "min_btag": 1}
    assert _fp(tmp_path / "a", config=base) != _fp(tmp_path / "b", config={**base, "systematic": "down"})


def test_a_payload_change_also_changes_the_canonical_ir(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # the External descriptor's content hash is part of the IR, so the *computation* identity is
    # sensitive to the correction/model content too (defence in depth for cache-poisoning)
    a, _ = agc.build_agc(tmp_path / "a")
    b, _ = agc.build_agc(tmp_path / "b", sf_scale=1.10)
    assert a.manifest["analysis"]["ir"] != b.manifest["analysis"]["ir"]
