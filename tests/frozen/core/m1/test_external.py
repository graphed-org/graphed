"""External nodes carry a PayloadDescriptor that participates in the structural hash (M1)."""

from __future__ import annotations

import graphed.core as gc


def _descriptor(**overrides: str) -> dict[str, str]:
    base = {
        "kind": "onnx_model",
        "content_hash": "sha256:abc123",
        "framework": "onnxruntime",
        "version": "1.17.0",
        "io_schema": "f32[1,4]->f32[1,2]",
        "preprocessing_ref": "scale_v1",
    }
    base.update(overrides)
    return base


def test_identical_descriptors_intern_to_one() -> None:
    s = gc.GraphStore()
    src = s.add_source("e")
    a = s.add_external(_descriptor(), [src])
    b = s.add_external(_descriptor(), [src])
    assert a == b


def test_changing_content_hash_yields_distinct_node() -> None:
    s = gc.GraphStore()
    src = s.add_source("e")
    a = s.add_external(_descriptor(), [src])
    b = s.add_external(_descriptor(content_hash="sha256:def456"), [src])
    assert a != b


def test_changing_any_descriptor_field_yields_distinct_node() -> None:
    s = gc.GraphStore()
    src = s.add_source("e")
    base = s.add_external(_descriptor(), [src])
    for field in ("kind", "framework", "version", "io_schema", "preprocessing_ref"):
        other = s.add_external(_descriptor(**{field: "CHANGED"}), [src])
        assert other != base, f"changing {field} must change the NodeId"


def test_payload_descriptor_class_roundtrips() -> None:
    d = gc.PayloadDescriptor(
        kind="correctionlib",
        content_hash="sha256:777",
        framework="correctionlib",
        version="2.6",
        io_schema="json",
        preprocessing_ref=None,
    )
    s = gc.GraphStore()
    src = s.add_source("e")
    a = s.add_external(d, [src])
    b = s.add_external(d, [src])
    assert a == b
    assert d.kind == "correctionlib"
    assert d.content_hash == "sha256:777"
