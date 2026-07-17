"""External payload descriptors flag opaque callables as a preservation risk (M2)."""

from __future__ import annotations

import numpy as np

from graphed import Session
from graphed.core import PayloadDescriptor
from graphed.numpy import NumpyBackend, from_array


def test_external_payload_describes_opaque_callable() -> None:
    backend = NumpyBackend()
    desc = backend.external_payload("map", {"fn": "myfn"})
    assert isinstance(desc, PayloadDescriptor)
    assert desc.kind == "opaque_callable"
    assert "myfn" in desc.content_hash
    assert desc.framework == "python"
    # the descriptor's kind/hash mark it un-content-addressed -> a preservation risk (M9 fixes this)
    assert desc.content_hash.startswith("unhashed-opaque:")


def test_non_external_op_has_no_payload() -> None:
    assert NumpyBackend().external_payload("add", {}) is None


def test_map_records_external_node_with_descriptor() -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", np.array([1.0, 2.0, 3.0]))
    n0 = s.node_count()
    out = a.map(lambda x: np.asarray(x) * 2, name="double")
    assert s.node_count() == n0 + 1
    assert np.allclose(s.materialize(out), [2.0, 4.0, 6.0])


def test_distinct_callables_distinct_nodes() -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", np.array([1.0, 2.0, 3.0]))
    m1 = a.map(lambda x: x, name="f1")
    m2 = a.map(lambda x: x, name="f2")
    assert m1.node_id != m2.node_id
