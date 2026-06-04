"""External nodes content-hash the correctionlib JSON / ONNX model (plan M3 / A.3.1)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from graphed import Session
from graphed_corpus import make_events

from graphed_awkward import AwkwardBackend, from_awkward, gak


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_correction(path: Path, weight: float = 1.0) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "corrections": [{"name": "sf", "version": 1, "data": weight}],
            }
        )
    )


def _write_onnx(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.save(model, str(path))


def test_correction_descriptor_content_hashes_json(tmp_path: Path) -> None:
    jpath = tmp_path / "sf.json"
    _write_correction(jpath)
    desc = AwkwardBackend().external_payload("correction", {"path": str(jpath), "name": "sf"})
    assert desc is not None
    assert desc.kind == "correctionlib"
    assert desc.content_hash == _sha(jpath)
    assert desc.version == "2"
    assert desc.io_schema == "sf"


def test_correction_external_node_carries_hash_in_graph(tmp_path: Path) -> None:
    jpath = tmp_path / "sf.json"
    _write_correction(jpath)
    s = Session(AwkwardBackend())
    events = from_awkward(s, "events", make_events(n_events=50))
    gak.apply_correction(str(jpath), "sf", [events.Jet.pt], evaluator=lambda x: x)
    assert _sha(jpath) in s.to_dot()  # the External node embeds the content-hashed descriptor


def test_changing_correction_changes_hash(tmp_path: Path) -> None:
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    _write_correction(p1, weight=1.0)
    _write_correction(p2, weight=2.0)
    b = AwkwardBackend()
    h1 = b.external_payload("correction", {"path": str(p1), "name": "sf"})
    h2 = b.external_payload("correction", {"path": str(p2), "name": "sf"})
    assert h1 is not None and h2 is not None
    assert h1.content_hash != h2.content_hash


def test_onnx_descriptor_content_hashes_model(tmp_path: Path) -> None:
    mpath = tmp_path / "model.onnx"
    _write_onnx(mpath)
    desc = AwkwardBackend().external_payload("onnx", {"path": str(mpath)})
    assert desc is not None
    assert desc.kind == "onnx_model"
    assert desc.content_hash == _sha(mpath)
    assert "13" in desc.version  # opset
    assert "x->y" in desc.io_schema


def test_onnx_external_node_recorded(tmp_path: Path) -> None:
    mpath = tmp_path / "model.onnx"
    _write_onnx(mpath)
    s = Session(AwkwardBackend())
    events = from_awkward(s, "events", make_events(n_events=50))
    jet_pt = events.Jet.pt
    n0 = s.node_count()
    gak.onnx_inference(str(mpath), [jet_pt], runner=lambda x: x)
    assert s.node_count() == n0 + 1  # exactly one External node
    assert _sha(mpath) in s.to_dot()
