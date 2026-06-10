"""M8 — the canonical IR is serializable, versioned, deterministic, and byte-identical.

Plan A.3.1: "the serializable IR — not cloudpickle — is the canonical durable representation." Plan
M8 gate: "Identical plan -> byte-identical serialization." These tests pin the IR-level guarantees
the durable Plan builds on; the higher-level DurablePlan guarantees live in ``test_durable_plan.py``.
"""

from __future__ import annotations

import pytest

from graphed_core import GraphStore, PayloadDescriptor


def _analysis() -> GraphStore:
    g = GraphStore()
    src = g.add_source("events", {"uri": "data.root"})
    pt = g.add_op("pt", [src])
    cut = g.add_op("gt", [pt], {"thr": 30.0})
    nn = g.add_external(
        PayloadDescriptor(
            kind="onnx",
            content_hash="deadbeef",
            framework="onnxruntime",
            version="1.17",
            io_schema="f32[4]->f32",
            preprocessing_ref="pre://x",
        ),
        [cut],
        {"batch": 32},
    )
    red = g.add_reduction("sum", [nn])
    return g, red


def test_roundtrip_preserves_structure() -> None:
    g, out = _analysis()
    blob = g.serialize(outputs=[out])
    back = GraphStore.deserialize(blob)
    assert back.node_count() == g.node_count()
    assert back.to_dot() == GraphStore.deserialize(blob).to_dot()


def test_reserialize_is_byte_identical() -> None:
    g, out = _analysis()
    blob = g.serialize(outputs=[out])
    assert GraphStore.deserialize(blob).serialize() == blob  # carried marks reserialize identically


def test_identical_graphs_serialize_identically() -> None:
    # determinism gate at the IR level: same construction -> same bytes
    g1, o1 = _analysis()
    g2, o2 = _analysis()
    assert g1.serialize(outputs=[o1]) == g2.serialize(outputs=[o2])


def test_serialization_is_versioned() -> None:
    g, out = _analysis()
    blob = g.serialize(outputs=[out])
    assert blob[:4] == b"GIR1", "the durable form carries a version magic (plan M8: versioned)"


def test_reduced_graph_with_stages_roundtrips() -> None:
    # the durable plan carries the *reduced* IR, which contains fused Stage nodes
    g, out = _analysis()
    reduced, _report = g.reduce(outputs=[out])
    blob = reduced.serialize()
    back = GraphStore.deserialize(blob)
    assert back.to_dot() == reduced.to_dot()
    assert back.serialize() == blob


def test_external_payload_descriptor_survives_roundtrip() -> None:
    # External payload descriptors are the reproducibility metadata (A.3.1) and must not be lost
    g, out = _analysis()
    back = GraphStore.deserialize(g.serialize(outputs=[out]))
    dot = back.to_dot()
    assert "onnx" in dot and "deadbeef" in dot and "onnxruntime" in dot


def test_bad_magic_is_rejected() -> None:
    with pytest.raises(ValueError, match="magic"):
        GraphStore.deserialize(b"NOPEnot a real blob")


def test_truncation_is_rejected() -> None:
    g, out = _analysis()
    blob = g.serialize(outputs=[out])
    with pytest.raises(ValueError, match="truncated"):
        GraphStore.deserialize(blob[:-3])
