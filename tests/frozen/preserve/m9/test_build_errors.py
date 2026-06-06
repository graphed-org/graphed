"""M9 — build/evaluate fail honestly rather than producing an incomplete or wrong bundle."""

from __future__ import annotations

import agc
import pytest

from graphed_preserve import PreserveError, build_bundle
from graphed_preserve.externals import evaluate_external


def test_build_requires_payload_bytes_for_non_opaque_externals(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pdir = tmp_path / "payloads"
    agc.write_payloads(pdir)
    events = agc.make_events_for()
    s, value, weight = agc.record(events, pdir, config=agc._DEFAULT_CONFIG)
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


def test_evaluate_external_rejects_an_unknown_kind() -> None:
    node = {"descriptor": {"kind": "mystery", "io_schema": "x"}, "params": {}}
    with pytest.raises(PreserveError, match="no evaluator"):
        evaluate_external(node, [None], b"")
