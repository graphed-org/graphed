"""An output dtype numpy lacks is cast to float64, or refused by name without one (#246)."""

from __future__ import annotations

import io
from typing import Any

import awkward as ak
import numpy as np
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.preserve import PYTORCH_PLUGIN
from graphed.preserve.errors import PreserveError
from graphed.preserve.externals import record_external
from graphed.preserve.externals._helpers import _as_event_array


def test_a_bfloat16_pytorch_model_runs() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
    buf = io.BytesIO()
    torch.jit.save(torch.jit.trace(model, torch.zeros(1, 2, dtype=torch.bfloat16)), buf)
    session = Session(AwkwardBackend())
    ev = from_awkward("ev", ak.Array({"a": [1.0, 2.0], "b": [3.0, 4.0]}), session=session)
    out = record_external(session, PYTORCH_PLUGIN, buf.getvalue(), [ev.a, ev.b])
    assert ak.to_list(session.materialize(out)) == [7.0, 10.0]


@pytest.mark.parametrize("dtype", ["bfloat16", "float8_e4m3fn", "float16"])
def test_a_torch_float_numpy_lacks_is_cast_to_float64(dtype: str) -> None:
    torch = pytest.importorskip("torch")
    values = _as_event_array(torch.tensor([[1.5], [2.0]]).to(getattr(torch, dtype)))
    assert (str(ak.type(values)), ak.to_list(values)) == ("2 * float64", [1.5, 2.0])


@pytest.mark.parametrize("dtype", ["complex32", "complex64", "qint8"])
def test_a_torch_output_with_no_float64_value_is_refused_by_name(dtype: str) -> None:
    torch = pytest.importorskip("torch")
    ones: Any = torch.ones(2, 1)
    tensor = (
        torch.quantize_per_tensor(ones, 0.1, 0, torch.qint8)
        if dtype == "qint8"
        else ones.to(getattr(torch, dtype))
    )
    with pytest.raises(PreserveError, match=f"output is torch.{dtype}, which has no float64"):
        _as_event_array(tensor)


def test_a_complex_numpy_output_is_refused_not_truncated() -> None:
    with pytest.raises(PreserveError, match="output is complex64, which has no float64 value"):
        _as_event_array(np.array([[1 + 2j], [3 + 4j]], dtype=np.complex64))
