"""The ``onnx_model`` plugin: hash of WEIGHTS (+ graph op structure), not file bytes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from ._base import ExternalPlugin
from ._helpers import _as_event_array, memoized_model_hash, ml_matrix, parse_call_template

#: onnxruntime's input type strings whose element name is not numpy's
_ORT_DTYPES = {"tensor(float)": "float32", "tensor(double)": "float64"}


def onnx_content_hash(payload: bytes) -> str:
    """Return the content hash of an ONNX payload."""
    return memoized_model_hash("onnx", payload, _onnx_content_hash_impl)


def _onnx_content_hash_impl(payload: bytes) -> str:
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
    """Run the ONNX session on ``inputs``, one value per event."""
    import awkward as ak  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    # onnxruntime takes no implicit cast: feed each input the dtype the graph declares for it
    declared = {i.name: i.type for i in session.get_inputs()}
    template = parse_call_template(params, len(inputs), allow_kwargs=False)
    if template is None:  # legacy: one (-1, 1) feed
        name = str(params.get("input_name", "")) or session.get_inputs()[0].name
        feeds = {name: np.asarray(ak.to_numpy(ak.Array(inputs[0]))).reshape(-1, 1)}
    else:
        args, _ = template
        if len(args) == 1 and args[0][0] == "named":  # ONNX feeds are named — the natural form
            feeds = {name: ml_matrix(entry, inputs) for name, entry in args[0][1].items()}
        else:  # positional: mapped to the graph's declared input order
            graph_inputs = [i.name for i in session.get_inputs()]
            feeds = {graph_inputs[k]: ml_matrix(entry, inputs) for k, entry in enumerate(args)}
    feeds = {
        name: x.astype(_ORT_DTYPES.get(declared[name], declared[name][7:-1]), copy=False)
        if name in declared
        else x
        for name, x in feeds.items()
    }
    return _as_event_array(session.run(None, feeds)[0])


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
    output_dtype="float64",
)
