"""Evaluators for `External` payload nodes — the single source of eval truth (plan M9).

The same functions are used at **build time** (the analysis records its corrections/inference with
these as the eval callables) and at **reproduce time** (the IR interpreter calls them), so a bundle
reproduces bit-for-bit: identical library + identical content-addressed payload bytes -> identical
result. Corrections stay correctionlib JSON and models stay ONNX (HEP standards; invent no formats).
"""

from __future__ import annotations

from typing import Any

from .errors import PreserveError


def eval_correctionlib(payload: bytes, *, name: str, systematic: str, x: Any) -> Any:
    """Evaluate a correctionlib correction set (a weight systematic): ``correction.evaluate(syst, x)``
    over a per-event scalar input, returning a per-event weight array."""
    import awkward as ak  # noqa: PLC0415
    import correctionlib  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    cset = correctionlib.CorrectionSet.from_string(payload.decode("utf-8"))
    arr = np.asarray(ak.to_numpy(ak.Array(x)), dtype="float64")
    weight = cset[name].evaluate(systematic, arr)
    return ak.Array(np.asarray(weight, dtype="float64"))


def eval_onnx(payload: bytes, *, input_name: str, x: Any) -> Any:
    """Run an ONNX model (per-event inference) over a single per-event feature, returning the score."""
    import awkward as ak  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415

    sess = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
    name = input_name or sess.get_inputs()[0].name
    arr = np.asarray(ak.to_numpy(ak.Array(x)), dtype="float32").reshape(-1, 1)
    out = sess.run(None, {name: arr})[0].reshape(-1)
    return ak.Array(np.asarray(out, dtype="float64"))


def evaluate_external(node: dict[str, Any], inputs: list[Any], payload: bytes) -> Any:
    """Dispatch an External node to its evaluator using its descriptor + the resolved payload bytes."""
    descriptor = node["descriptor"]
    kind = descriptor["kind"]
    params = node["params"]
    if kind == "correctionlib":
        return eval_correctionlib(
            payload,
            name=str(params.get("name", descriptor["io_schema"])),
            systematic=str(params.get("systematic", "nominal")),
            x=inputs[0],
        )
    if kind == "onnx_model":
        return eval_onnx(payload, input_name=str(params.get("input_name", "")), x=inputs[0])
    raise PreserveError(
        f"no evaluator for external kind {kind!r} (opaque/cloudpickled nodes are a preservation risk)"
    )
