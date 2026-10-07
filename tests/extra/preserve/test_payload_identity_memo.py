"""Recording one payload many times derives its identity once; another payload gets its own."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import awkward as ak
import numpy as np
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.awkward.payloads import correctionlib_contents_hash, onnx_weights_hash
from graphed.preserve import CORRECTIONLIB_PLUGIN, XGBOOST_PLUGIN
from graphed.preserve.externals import ExternalPlugin, record_external
from graphed.preserve.externals._helpers import _HASH_MEMO

N = 5
CSET = json.dumps({"schema_version": 2, "corrections": []}).encode()
OTHER_CSET = json.dumps({"schema_version": 2, "corrections": [], "description": "other"}).encode()


@pytest.fixture(autouse=True)
def _cold_memo() -> None:
    _HASH_MEMO.clear()


def _x() -> tuple[Session, Any]:
    s = Session(AwkwardBackend())
    return s, from_awkward(s, "e", ak.zip({"x": np.array([0.5, 2.0])}, depth_limit=1)).x


def _descriptor(s: Session, arr: Any) -> dict[str, Any]:
    node: dict[str, Any] = next(n for n in s._store.nodes() if n["id"] == arr.node_id)
    return dict(node["descriptor"])


def _count_parses(monkeypatch: pytest.MonkeyPatch, module: Any, name: str, payload: bytes) -> list[None]:
    """Patch ``module.name`` to log each call whose first argument is ``payload``."""
    calls: list[None] = []
    real: Callable[..., Any] = getattr(module, name)

    def counting(arg: Any, *args: Any, **kwargs: Any) -> Any:
        if (arg.encode() if isinstance(arg, str) else bytes(arg)) == payload:
            calls.append(None)
        return real(arg, *args, **kwargs)

    monkeypatch.setattr(module, name, counting)
    return calls


def _onnx_model(weight: float) -> bytes:
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array(np.array([[weight]], dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="B")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 1])
    graph = helper.make_graph([helper.make_node("Gemm", ["x", "W", "B"], ["y"])], "m", [x], [y], initializer=[w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    out: bytes = model.SerializeToString()
    return out


def test_apply_correction_parses_a_set_once_across_recorded_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    s, x = _x()
    parses = _count_parses(monkeypatch, json, "loads", CSET)
    first = gak.apply_correction(CSET, "src0", [x], lambda *a: a[0], args=["$0"])
    after_one = len(parses)
    rest = [gak.apply_correction(CSET, f"src{i}", [x], lambda *a: a[0], args=["$0"]) for i in range(1, N)]

    assert after_one > 0
    assert len(parses) == after_one
    for i, arr in enumerate([first, *rest]):
        d = _descriptor(s, arr)
        assert (d["content_hash"], d["version"], d["io_schema"]) == (
            correctionlib_contents_hash(CSET),
            "2",
            f"src{i}",
        )


def test_a_different_correction_set_gets_its_own_identity() -> None:
    s, x = _x()
    a = gak.apply_correction(CSET, "sf", [x], lambda *v: v[0], args=["$0"])
    b = gak.apply_correction(OTHER_CSET, "sf", [x], lambda *v: v[0], args=["$0"])
    ha, hb = _descriptor(s, a)["content_hash"], _descriptor(s, b)["content_hash"]

    assert ha == correctionlib_contents_hash(CSET)
    assert hb == correctionlib_contents_hash(OTHER_CSET)
    assert ha != hb


def test_onnx_inference_loads_a_model_once_across_recorded_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    onnx = pytest.importorskip("onnx")
    model, other = _onnx_model(0.5), _onnx_model(2.0)
    s, x = _x()
    loads = _count_parses(monkeypatch, onnx, "load_from_string", model)
    gak.onnx_inference(model, [x], lambda m: m, args=[["$0"]])
    after_one = len(loads)
    arrs = [gak.onnx_inference(model, [x], lambda m: m, args=[["$0"]]) for _ in range(N)]
    distinct = gak.onnx_inference(other, [x], lambda m: m, args=[["$0"]])

    assert after_one > 0
    assert len(loads) == after_one
    assert {_descriptor(s, a)["content_hash"] for a in arrs} == {onnx_weights_hash(model)}
    assert _descriptor(s, distinct)["content_hash"] == onnx_weights_hash(other) != onnx_weights_hash(model)


@pytest.mark.parametrize(
    ("plugin", "params"),
    [(CORRECTIONLIB_PLUGIN, {"name": "sf", "args": ["$0"]}), (XGBOOST_PLUGIN, {"args": [["$0"]]})],
    ids=["correctionlib", "xgboost"],
)
def test_record_external_hashes_a_payload_once_across_recorded_calls(
    monkeypatch: pytest.MonkeyPatch, plugin: ExternalPlugin, params: dict[str, Any]
) -> None:
    s, x = _x()
    expected, other_expected = plugin.content_hash(CSET), plugin.content_hash(OTHER_CSET)
    _HASH_MEMO.clear()
    parses = _count_parses(monkeypatch, json, "loads", CSET)
    record_external(s, plugin, CSET, [x], params=params, output_type="float64")
    after_one = len(parses)
    arrs = [record_external(s, plugin, CSET, [x], params=params, output_type="float64") for _ in range(N)]
    other = record_external(s, plugin, OTHER_CSET, [x], params=params, output_type="float64")

    assert after_one > 0
    assert len(parses) == after_one
    assert {_descriptor(s, a)["content_hash"] for a in arrs} == {expected}
    assert _descriptor(s, other)["content_hash"] == other_expected != expected
