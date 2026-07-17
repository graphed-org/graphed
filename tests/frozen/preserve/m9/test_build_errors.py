"""M9 — build/evaluate fail honestly rather than producing an incomplete or wrong bundle."""

from __future__ import annotations

import agc
import pytest

from graphed_preserve import PreserveError, build_bundle
from graphed_preserve.externals import evaluate_external


def test_build_requires_payload_bytes_for_non_opaque_externals(tmp_path) -> None:  # type: ignore[no-untyped-def]
    events = agc.make_events_for()
    s, value, weight = agc.record(
        events, config=agc._DEFAULT_CONFIG, corr_bytes=agc.correctionlib_json(), model_bytes=agc.onnx_model()
    )
    # a correctionlib/ONNX external whose bytes were not supplied must not silently vanish
    with pytest.raises(PreserveError, match="no payload bytes"):
        build_bundle(
            tmp_path / "bundle",
            session=s,
            value=value,
            weight=weight,
            datasets={"events": events},
            payloads={},  # missing
            histogram=agc.HIST,
        )


def test_build_rejects_payload_that_does_not_match_its_hash(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from graphed_preserve import CORRECTIONLIB_PLUGIN, ONNX_PLUGIN  # noqa: PLC0415

    events = agc.make_events_for()
    corr, model = agc.correctionlib_json(), agc.onnx_model()
    s, value, weight = agc.record(events, config=agc._DEFAULT_CONFIG, corr_bytes=corr, model_bytes=model)
    # supply the RIGHT keys but WRONG bytes for the correction -> hash mismatch is caught
    poisoned = {
        CORRECTIONLIB_PLUGIN.content_hash(corr): agc.correctionlib_json(scale=9.9),  # bytes != recorded hash
        ONNX_PLUGIN.content_hash(model): model,
    }
    with pytest.raises(PreserveError, match="hashes to"):
        build_bundle(
            tmp_path / "bundle",
            session=s,
            value=value,
            weight=weight,
            datasets={"events": events},
            payloads=poisoned,
            histogram=agc.HIST,
        )


def test_evaluate_external_rejects_an_unregistered_kind() -> None:
    node = {"descriptor": {"kind": "mystery", "io_schema": "x"}, "params": {}}
    with pytest.raises(PreserveError, match="no plugin"):
        evaluate_external(node, [None], b"")
