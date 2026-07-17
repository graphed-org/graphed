"""M8 support — `Session.serialized_ir` is the "compile" step of a DurablePlan deployment.

A user records an analysis once and serializes it to the canonical, versioned, byte-identical
durable IR (optionally optimized by the M4 reducer first). That byte string is what a
`graphed.core.DurablePlan` carries and re-targets at many datasets. These tests pin: the bytes
round-trip through graphed-core, are deterministic, and the optimized form is the reduced graph.
"""

from __future__ import annotations

import pytest
from backends import ListBackend, from_list

import graphed.core
from graphed import Session


def _record() -> tuple[Session, object]:
    s = Session(ListBackend())
    a = from_list(s, "a", [1, 2, 3])
    b = from_list(s, "b", [4, 5, 6])
    c = (a + b) * a
    result = c + (a + b)  # a repeated sub-expression -> exercises interning before serialization
    return s, result


def test_serialized_ir_roundtrips_through_graphed_core() -> None:
    s, result = _record()
    blob = s.serialized_ir(result, optimize=False)
    back = graphed.core.GraphStore.deserialize(blob)
    assert back.node_count() == s.node_count()
    assert back.serialize() == blob  # byte-identical round trip


def test_serialized_ir_is_deterministic() -> None:
    s1, r1 = _record()
    s2, r2 = _record()
    assert s1.serialized_ir(r1) == s2.serialized_ir(r2)


def test_optimized_ir_keeps_the_output_and_is_the_reduced_graph() -> None:
    # [freeze-M22-1, user-authorized respin: serializing no longer writes output marks into the
    # store — the reference is the store reduced FOR the same request]
    s, result = _record()
    optimized = s.serialized_ir(result, optimize=True)
    expected = s._store.reduce(outputs=[result.node_id])[0].serialize()
    assert optimized == expected
    assert graphed.core.GraphStore.deserialize(optimized).node_count() > 0  # not DCE'd to nothing
    assert s._store.outputs() == []  # and serializing left NO state behind


def test_optimize_without_an_output_is_rejected() -> None:
    s, _ = _record()
    with pytest.raises(ValueError, match="needs at least one output"):
        s.serialized_ir(optimize=True)


def test_serialized_ir_feeds_a_durable_plan() -> None:
    # the headline usability path: record -> serialize -> DurablePlan -> retarget at a dataset
    s, result = _record()
    plan = graphed.core.DurablePlan(
        ir=s.serialized_ir(result),
        process=graphed.core.OpSpec.from_ref("builtins:len"),
        combine=graphed.core.OpSpec.from_ref("operator:add"),
        empty=graphed.core.OpSpec.from_ref("builtins:int"),
    )
    deployed = plan.for_dataset(graphed.core.Dataset("file://x.root", n_events=100), chunk_size=40)
    assert len(deployed.partitions) == 3
    assert deployed.ir_fingerprint() == plan.ir_fingerprint()  # same compiled computation
