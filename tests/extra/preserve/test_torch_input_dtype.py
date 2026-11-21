"""PyTorch inputs are cast only to a dtype numpy has, unless ``input_dtype=`` names one (#252)."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import awkward as ak
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.errors import GraphedTypeError
from graphed.preserve.externals import PYTORCH_PLUGIN, record_external

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")
X = 1.03


class Upcasting(torch.nn.Module):  # type: ignore[misc]
    """A weight stored in a narrow dtype, upcast inside ``forward``."""

    def __init__(self, dtype: Any) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.full((1, 1), 4.0, dtype=dtype), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() @ self.w.float()


class Plain(torch.nn.Module):  # type: ignore[misc]
    """A float32 weight with no cast in ``forward``: TorchScript refuses a float64 input."""

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.full((1, 1), 4.0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w


def _payload(module: Any) -> bytes:
    buf = io.BytesIO()
    torch.jit.save(torch.jit.script(module), buf)
    return buf.getvalue()


def _run(module: Any, **params: Any) -> tuple[float, dict[str, Any]]:
    session = Session(AwkwardBackend())
    ev = from_awkward("ev", ak.Array({"x": [X]}), session=session)
    out = record_external(session, PYTORCH_PLUGIN, _payload(module), [ev.x], params=params or None)
    return float(ak.to_numpy(session.materialize(out))[0]), session._nodes()[out.node_id]["params"]


def _eager(module: Any, dtype: Any) -> float:
    with torch.no_grad():
        return float(module(torch.tensor([[X]], dtype=torch.float64).to(dtype))[0, 0])


@pytest.mark.parametrize("weights", ["bfloat16", "float8_e4m3fn"])
def test_a_weight_dtype_numpy_lacks_leaves_the_inputs_unrounded(weights: str) -> None:
    module = Upcasting(getattr(torch, weights))
    got = _run(module)[0]
    assert got == _eager(module, torch.float32)
    assert got != _eager(module, getattr(torch, weights))


def test_a_declared_input_dtype_wins() -> None:
    module = Upcasting(torch.bfloat16)
    assert _run(module, input_dtype="bfloat16")[0] == _eager(module, torch.bfloat16)
    assert _run(module, input_dtype="float32")[0] == _eager(module, torch.float32)


def test_a_weight_dtype_numpy_has_still_casts_the_inputs() -> None:
    assert _run(Plain())[0] == _eager(Plain(), torch.float32)


def test_the_declared_dtype_is_part_of_the_node() -> None:
    module = Upcasting(torch.bfloat16)
    assert "input_dtype" not in _run(module)[1]
    assert _run(module, input_dtype="float32")[1]["input_dtype"] == "float32"


@pytest.mark.parametrize(
    ("spelling", "match"),
    [
        ("torch.float32", "names no torch dtype"),
        ("nope", "names no torch dtype"),
        ("Tensor", "names no torch dtype"),
        ("float", "spell it 'float32'"),
        ("half", "spell it 'float16'"),
    ],
)
def test_an_input_dtype_not_spelled_as_torch_does_is_refused(spelling: str, match: str) -> None:
    session = Session(AwkwardBackend())
    ev = from_awkward("ev", ak.Array({"x": [X]}), session=session)
    with pytest.raises(GraphedTypeError, match=match):
        record_external(
            session, PYTORCH_PLUGIN, _payload(Plain()), [ev.x], params={"input_dtype": spelling}
        )
