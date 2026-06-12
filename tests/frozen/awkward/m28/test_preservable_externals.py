"""M28 — the external-recording seam aligned with preservation (additive; M3 path untouched).

Three defects this fixes, each pinned here:

1. **Hash-convention divergence**: the M3 descriptors hash RAW FILE BYTES, while the
   preservation plugins hash CONTENT IDENTITY (canonical JSON for correctionlib, weights +
   graph structure for ONNX) — same ``kind``, two identities, so an M3-recorded External
   cannot pass a bundle's payload-integrity check. The new descriptor builders use the
   preservation conventions (identical domain strings and algorithms).
2. **No call template**: record-time evaluation used the caller's callable blindly while
   replay used a hard-wired legacy shape — record and replay could disagree. The new path
   takes ``args=``/``kwargs=`` templates, stores them in the node params (canonical JSON),
   and **record-time evaluation materializes the SAME template** — agreement by construction.
3. **Path leakage**: the M3 params embed filesystem paths in the IR; the new path's params
   carry no paths (the payload travels by content hash).

The cross-repo acceptance (gak-recorded -> build_bundle integrity -> reproduce bit-for-bit)
lives in graphed-preserve's frozen suite; this module pins everything provable here without a
graphed-preserve dependency.
"""

from __future__ import annotations

import json
from typing import Any

import awkward as ak
import numpy as np
import pytest
from graphed import Session

from graphed_awkward import AwkwardBackend, from_awkward, gak
from graphed_awkward.payloads import (
    correctionlib_contents_descriptor,
    correctionlib_contents_hash,
    onnx_weights_descriptor,
    onnx_weights_hash,
)


def _node(s: Session, arr: Any) -> dict[str, Any]:
    """The interned node behind ``arr`` (via the session's core store)."""
    return next(n for n in s._store.nodes() if n["id"] == arr.node_id)


CSET = json.dumps(
    {
        "schema_version": 2,
        "corrections": [
            {
                "name": "jetsf",
                "version": 1,
                "inputs": [
                    {"name": "systematic", "type": "string"},
                    {"name": "pt", "type": "real"},
                    {"name": "eta", "type": "real"},
                ],
                "output": {"name": "sf", "type": "real"},
                "data": {
                    "nodetype": "category",
                    "input": "systematic",
                    "content": [{"key": "nominal", "value": 1.5}],
                },
            }
        ],
    }
).encode()


def _onnx_bytes(weight: float, *, extra_node: bool = False) -> bytes:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    w = numpy_helper.from_array(np.array([[weight]], dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="B")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 1])
    nodes = [helper.make_node("Gemm", ["x", "W", "B"], ["z" if extra_node else "y"])]
    if extra_node:
        nodes.append(helper.make_node("Relu", ["z"], ["y"]))
    graph = helper.make_graph(nodes, "m", [x], [y], initializer=[w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    onnx.checker.check_model(model)
    return model.SerializeToString()  # type: ignore[no-any-return]


# ---------------------------------- the aligned hash conventions ---------------------------------
def test_correctionlib_contents_hash_ignores_formatting_but_not_content() -> None:
    import hashlib  # noqa: PLC0415

    reordered = json.dumps(json.loads(CSET), sort_keys=True).encode()  # same content, new bytes
    whitespaced = json.dumps(json.loads(CSET), indent=3).encode()
    changed = CSET.replace(b'"value": 1.5', b'"value": 1.6')
    assert reordered != CSET
    assert correctionlib_contents_hash(CSET) == correctionlib_contents_hash(reordered)
    assert correctionlib_contents_hash(CSET) == correctionlib_contents_hash(whitespaced)
    assert correctionlib_contents_hash(CSET) != correctionlib_contents_hash(changed)
    # ... and it is NOT the raw-bytes hash (the M3 convention this seam diverged from)
    assert correctionlib_contents_hash(CSET) != "sha256:" + hashlib.sha256(CSET).hexdigest()


def test_onnx_weights_hash_is_content_identity() -> None:
    a, b_weights, c_structure = _onnx_bytes(0.5), _onnx_bytes(0.9), _onnx_bytes(0.5, extra_node=True)
    assert onnx_weights_hash(a) == onnx_weights_hash(_onnx_bytes(0.5))
    assert onnx_weights_hash(a) != onnx_weights_hash(b_weights)  # weights are content
    assert onnx_weights_hash(a) != onnx_weights_hash(c_structure)  # graph structure is content


def test_descriptors_carry_the_aligned_hash_and_metadata() -> None:
    d = correctionlib_contents_descriptor(CSET, "jetsf")
    assert d.kind == "correctionlib"
    assert d.content_hash == correctionlib_contents_hash(CSET)
    assert d.version == "2"  # schema_version, from the payload itself
    m = onnx_weights_descriptor(_onnx_bytes(0.5))
    assert m.kind == "onnx_model"
    assert m.content_hash == onnx_weights_hash(_onnx_bytes(0.5))


# ---------------------------------- the template-bearing recorders --------------------------------
def _session() -> tuple[Session, Any]:
    events = ak.Array(
        {"Jet": [[{"pt": 30.0, "eta": 0.5}, {"pt": 50.0, "eta": 1.0}], [], [{"pt": 80.0, "eta": 2.0}]]}
    )
    s = Session(AwkwardBackend())
    return s, from_awkward(s, "events", events)


def test_apply_correction_with_a_template_records_path_free_and_obeys_it() -> None:
    s, ev = _session()
    seen: list[tuple[Any, ...]] = []

    def evaluator(*call: Any) -> Any:
        seen.append(call)
        assert call[0] == "nominal"  # the CONSTANT, routed by the template
        return call[1] * 0.0 + 1.5  # jagged-shaped SF

    sf = gak.apply_correction(CSET, "jetsf", [ev.Jet.pt, ev.Jet.eta], evaluator, args=["nominal", "$0", "$1"])
    out = ak.Array(s.materialize(sf))
    assert ak.num(out, axis=1).tolist() == [2, 0, 1]  # jagged structure all the way through

    (call,) = seen
    assert len(call) == 3
    assert ak.num(ak.Array(call[1]), axis=1).tolist() == [2, 0, 1]  # inputs passed NATIVELY (jagged)

    node = _node(s, sf)
    assert "path" not in node["params"]  # no filesystem leakage into the IR
    assert json.loads(str(node["params"]["args"])) == ["nominal", "$0", "$1"]  # the preserved template
    assert node["descriptor"]["content_hash"] == correctionlib_contents_hash(CSET)


def test_apply_correction_accepts_payload_bytes_or_a_path(tmp_path) -> None:  # type: ignore[no-untyped-def]
    s, ev = _session()
    p = tmp_path / "cset.json"
    p.write_bytes(CSET)
    a = gak.apply_correction(CSET, "jetsf", [ev.Jet.pt], lambda *c: c[-1], args=["nominal", "$0"])
    b = gak.apply_correction(str(p), "jetsf", [ev.Jet.pt], lambda *c: c[-1], args=["nominal", "$0"])
    na, nb = _node(s, a), _node(s, b)
    assert na["descriptor"]["content_hash"] == nb["descriptor"]["content_hash"]  # identity is content


def test_onnx_inference_with_a_group_template_stacks_features() -> None:
    pytest.importorskip("onnx")
    s, ev = _session()
    seen: list[Any] = []

    def runner(x: Any) -> Any:
        seen.append(np.asarray(x))
        return np.asarray(x)[:, 0].astype("float64")

    njet = gak.num(ev.Jet, axis=1)
    ht = gak.sum(ev.Jet.pt, axis=1)
    out = gak.onnx_inference(_onnx_bytes(0.5), [njet, ht], runner, args=[["$0", "$1"]])
    s.materialize(out)
    (x,) = seen
    assert x.shape == (3, 2) and x.dtype == np.float32  # ONE stacked (n_events, 2) float32 matrix

    node = _node(s, out)
    assert "path" not in node["params"]
    assert node["descriptor"]["content_hash"] == onnx_weights_hash(_onnx_bytes(0.5))


def test_the_legacy_m3_path_is_byte_for_byte_unchanged(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # no template -> the original recorder: backend-built descriptor, raw-file hash, path param
    import hashlib  # noqa: PLC0415

    s, ev = _session()
    p = tmp_path / "cset.json"
    p.write_bytes(CSET)
    sf = gak.apply_correction(str(p), "jetsf", [ev.Jet.pt], evaluator=lambda x: x)
    node = _node(s, sf)
    assert node["params"]["path"] == str(p)  # the M3 convention, untouched
    assert node["descriptor"]["content_hash"] == "sha256:" + hashlib.sha256(CSET).hexdigest()
