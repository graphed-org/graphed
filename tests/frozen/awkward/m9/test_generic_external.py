"""M9 support — a generic, user-defined External op carries a caller-supplied deterministic hash.

The legacy `correction` / `onnx` external ops content-hash a specific file format. The generic
`external` op lets a user record ANY external whose deterministic content hash is computed elsewhere
(e.g. a graphed-preserve plugin: ONNX -> hash of weights, correctionlib -> hash of contents, or a
user's own scheme). The backend is just the conduit: it records the descriptor verbatim.
"""

from __future__ import annotations

from graphed import Session
from graphed_corpus import make_events

from graphed_awkward import AwkwardBackend, from_awkward


def test_generic_external_descriptor_is_built_from_params() -> None:
    desc = AwkwardBackend().external_payload(
        "external",
        {
            "kind": "my_model",
            "content_hash": "sha256:deadbeef",
            "framework": "mylib",
            "version": "1",
            "io_schema": "x->y",
        },
    )
    assert desc is not None
    assert desc.kind == "my_model"
    assert desc.content_hash == "sha256:deadbeef"
    assert desc.framework == "mylib" and desc.version == "1" and desc.io_schema == "x->y"


def test_generic_external_records_a_node_with_its_hash() -> None:
    s = Session(AwkwardBackend())
    events = from_awkward(s, "events", make_events(n_events=50))
    jet_pt = events.Jet.pt
    n0 = s.node_count()
    s.record_external(
        "external",
        lambda *vals: vals[0],
        [jet_pt],
        {"kind": "my_model", "content_hash": "sha256:c0ffee", "io_schema": "pt->score"},
    )
    assert s.node_count() == n0 + 1  # exactly one External node
    assert "sha256:c0ffee" in s.to_dot()  # the user-supplied content hash is embedded in the graph


def test_legacy_external_ops_are_unchanged() -> None:
    b = AwkwardBackend()
    assert b.external_payload("add", {}) is None  # non-external op
    assert b.external_payload("map", {"fn": "f"}).kind == "opaque_callable"  # type: ignore[union-attr]
