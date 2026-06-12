"""The ``pytorch_model`` plugin (M26): TorchScript archives; hash = sorted state_dict + code."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from ._base import ExternalPlugin
from ._helpers import _as_event_array, _stack_feature_columns, ml_matrix, parse_call_template


def pytorch_content_hash(payload: bytes) -> str:
    """Weights (sorted state_dict) + TorchScript code — stable across re-saves of one model."""
    import io  # noqa: PLC0415

    import torch  # noqa: PLC0415

    model = torch.jit.load(io.BytesIO(payload), map_location="cpu")
    h = hashlib.sha256(b"torch-weights-code-v1")
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().numpy().tobytes())
    h.update(model.code.encode("utf-8"))  # architecture is content too: same weights, new graph
    return "sha256:" + h.hexdigest()


def load_pytorch(payload: bytes, params: Mapping[str, Any]) -> Any:
    import io  # noqa: PLC0415

    import torch  # noqa: PLC0415

    model = torch.jit.load(io.BytesIO(payload), map_location="cpu")
    model.eval()
    return model


def eval_pytorch(model: Any, params: Mapping[str, Any], inputs: list[Any]) -> Any:
    import torch  # noqa: PLC0415

    template = parse_call_template(params, len(inputs))
    with torch.no_grad():
        if template is None:  # legacy: one stacked feature matrix
            out = model(torch.from_numpy(_stack_feature_columns(inputs))).numpy()
        else:
            args, kwargs = template
            tensors = [torch.from_numpy(ml_matrix(e, inputs)) for e in args]
            kwtensors = {k: torch.from_numpy(ml_matrix(e, inputs)) for k, e in kwargs.items()}
            out = model(*tensors, **kwtensors).numpy()
    return _as_event_array(out)


def _pytorch_samples() -> list[bytes]:
    import io  # noqa: PLC0415

    import torch  # noqa: PLC0415

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
)
