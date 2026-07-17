"""Cover the remaining op dispatch and backend protocol surface (plan M3)."""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import pytest
from analyses import q1, record
from graphed import Session
from graphed_corpus import make_events

from graphed_awkward import AwkwardBackend, from_awkward, gak, payloads
from graphed_awkward._ops import apply


def _events() -> tuple[Session, object]:
    s = Session(AwkwardBackend())
    return s, from_awkward(s, "events", make_events(n_events=200))


def test_extra_reductions_and_helpers_record() -> None:
    s, events = _events()
    jets = events.Jet
    for out in (
        gak.count(jets.pt, axis=1),
        gak.argmax(jets.pt, axis=1, keepdims=True),
        gak.argsort(jets.pt, axis=1, ascending=False),
        gak.ones_like(jets.pt, dtype="int64"),
        gak.local_index(jets, axis=1),
    ):
        assert s.form(out).is_typetracer


def test_values_astype_and_minimum_via_apply() -> None:
    arr = ak.Array([[1.0, 2.0], [3.0]])
    assert str(ak.type(apply("ak.values_astype", [arr], {"dtype": "int64"}))).count("int64")
    assert ak.to_list(apply("minimum", [ak.Array([1, 5]), ak.Array([3, 2])], {})) == [1, 2]


def test_unsupported_op_raises() -> None:
    with pytest.raises(TypeError):
        apply("ak.nonexistent", [], {})


def test_backend_protocol_surface() -> None:
    b = AwkwardBackend()
    assert "source" in b.boundary_ops()
    assert b.project("filter", "USED", {}) == "USED"
    assert b.external_payload("add", {}) is None
    # opaque map callable still gets a preservation-risk descriptor
    desc = b.external_payload("map", {"fn": "f"})
    assert desc is not None and desc.kind == "opaque_callable"


def test_map_opaque_callable_records_external() -> None:
    s, events = _events()
    n0 = s.node_count()
    events.Jet.pt.map(lambda a: a, name="opaque")
    assert s.node_count() > n0


def test_dataset_descriptor(tmp_path: object) -> None:
    p = Path(str(tmp_path)) / "f.root"
    p.write_bytes(b"ROOTDATA")
    desc = payloads.dataset_descriptor("DAS:/my/dataset", [str(p)])
    assert desc.kind == "dataset"
    assert "sha256:" in desc.content_hash
    assert desc.io_schema == "DAS:/my/dataset"


def test_q1_records(shared_events: ak.Array) -> None:
    s, out = record(q1)
    assert s.form(out).is_typetracer
