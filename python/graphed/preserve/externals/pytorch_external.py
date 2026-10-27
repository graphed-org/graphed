"""The ``pytorch_model`` plugin (M26): TorchScript archives; hash = sorted state_dict + code."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from typing import Any

from ._base import ExternalPlugin
from ._helpers import (
    _as_event_array,
    _stack_feature_columns,
    memoized_model_hash,
    ml_matrix,
    parse_call_template,
)


def pytorch_content_hash(payload: bytes) -> str:
    """Return the content hash of a TorchScript payload."""
    return memoized_model_hash("pytorch", payload, _pytorch_content_hash_impl)


def _pytorch_content_hash_impl(payload: bytes) -> str:
    """Weights (sorted state_dict) + TorchScript code — stable across re-saves of one model."""
    import torch

    model = torch.jit.load(io.BytesIO(payload), map_location="cpu")
    h = hashlib.sha256(b"torch-weights-code-v1")
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().numpy().tobytes())
    h.update(model.code.encode("utf-8"))  # architecture is content too: same weights, new graph
    return "sha256:" + h.hexdigest()


def load_pytorch(payload: bytes, params: Mapping[str, Any]) -> Any:
    """Load a TorchScript payload on the CPU, in eval mode."""
    import torch

    model = torch.jit.load(io.BytesIO(payload), map_location="cpu")
    model.eval()
    return model


def eval_pytorch(model: Any, params: Mapping[str, Any], inputs: list[Any]) -> Any:
    """Run the TorchScript model on ``inputs``, one value per event."""
    import torch

    template = parse_call_template(params, len(inputs))
    # TorchScript takes no implicit cast: feed the dtype the model's weights declare
    dtype = next((p.dtype for p in model.parameters()), None)

    def tensor(matrix: Any) -> Any:
        return (
            torch.from_numpy(matrix).to(dtype) if dtype is not None else torch.from_numpy(matrix)
        )

    with torch.no_grad():
        if template is None:  # legacy: one stacked feature matrix
            out = model(tensor(_stack_feature_columns(inputs))).numpy()
        else:
            args, kwargs = template
            tensors = [tensor(ml_matrix(e, inputs)) for e in args]
            kwtensors = {k: tensor(ml_matrix(e, inputs)) for k, e in kwargs.items()}
            out = model(*tensors, **kwtensors).numpy()
    return _as_event_array(out)


def _pytorch_samples() -> list[bytes]:
    import torch

    def _model(weight: float) -> bytes:
        m = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Sigmoid())
        with torch.no_grad():
            m[0].weight.fill_(weight)
            m[0].bias.fill_(0.0)
        m.eval()
        buf = io.BytesIO()
        torch.jit.save(torch.jit.trace(m, torch.zeros(1, 1)), buf)
        return buf.getvalue()

    return [_model(0.5), _model(0.9)]


PYTORCH_PLUGIN = ExternalPlugin(
    kind="pytorch_model",
    content_hash=pytorch_content_hash,
    evaluate=eval_pytorch,
    samples=_pytorch_samples,
    load=load_pytorch,
    framework="torch",
    output_dtype="float64",
)
