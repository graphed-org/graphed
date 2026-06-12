"""The ``onnx_model`` plugin: hash of WEIGHTS (+ graph op structure), not file bytes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from ._base import ExternalPlugin
from ._helpers import ml_matrix, parse_call_template


def onnx_content_hash(payload: bytes) -> str:
    import onnx  # noqa: PLC0415
    from onnx import numpy_helper  # noqa: PLC0415

    model = onnx.load_from_string(payload)
    h = hashlib.sha256()
    h.update(b"onnx-weights-v1")
    for init in sorted(model.graph.initializer, key=lambda t: t.name):
        h.update(init.name.encode("utf-8"))
        h.update(numpy_helper.to_array(init).tobytes())
    for node in model.graph.node:  # structure too: same weights, different graph -> different hash
        h.update(node.op_type.encode("utf-8"))
    return "sha256:" + h.hexdigest()


def load_onnx(payload: bytes, params: Mapping[str, Any]) -> Any:
    """Create the ONNX Runtime session once (per worker) — not per call."""
    import onnxruntime as ort  # noqa: PLC0415

    return ort.InferenceSession(payload, providers=["CPUExecutionProvider"])


def eval_onnx(session: Any, params: Mapping[str, Any], inputs: list[Any]) -> Any:
    import awkward as ak  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    template = parse_call_template(params, len(inputs), allow_kwargs=False)
    if template is None:  # legacy: one (-1, 1) feed
        name = str(params.get("input_name", "")) or session.get_inputs()[0].name
        x = np.asarray(ak.to_numpy(ak.Array(inputs[0])), dtype="float32").reshape(-1, 1)
        out = session.run(None, {name: x})[0].reshape(-1)
        return ak.Array(np.asarray(out, dtype="float64"))
    args, _ = template
    if len(args) == 1 and args[0][0] == "named":  # ONNX feeds are named — the natural form
        feeds = {name: ml_matrix(entry, inputs) for name, entry in args[0][1].items()}
    else:  # positional: mapped to the graph's declared input order
        graph_inputs = [i.name for i in session.get_inputs()]
        feeds = {graph_inputs[k]: ml_matrix(entry, inputs) for k, entry in enumerate(args)}
    out = session.run(None, feeds)[0].reshape(-1)
    return ak.Array(np.asarray(out, dtype="float64"))


def _onnx_samples() -> list[bytes]:
    import numpy as np  # noqa: PLC0415
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    def _model(weight: float) -> bytes:
        w = numpy_helper.from_array(np.array([[weight]], dtype=np.float32), name="W")
        b = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="B")
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 1])
        graph = helper.make_graph(
            [helper.make_node("Gemm", ["x", "W", "B"], ["y"])], "m", [x], [y], initializer=[w, b]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
        return model.SerializeToString()  # type: ignore[no-any-return]

    return [_model(0.5), _model(0.9)]


ONNX_PLUGIN = ExternalPlugin(
    kind="onnx_model",
    content_hash=onnx_content_hash,
    evaluate=eval_onnx,
    samples=_onnx_samples,
    load=load_onnx,  # build the inference session once per worker
    framework="onnxruntime",
)
