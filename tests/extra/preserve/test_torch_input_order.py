"""The PyTorch plugin feeds a column-major matrix on ``order="F"``, as pandas does (#258)."""

from __future__ import annotations

import io
from typing import Any

import awkward as ak
import numpy as np
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.errors import GraphedTypeError
from graphed.preserve.externals import PYTORCH_PLUGIN, record_external

torch = pytest.importorskip("torch")

WIDTH, ROWS = 10, 9


def _model() -> tuple[Any, bytes]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(WIDTH, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)
    ).eval()
    buf = io.BytesIO()
    torch.jit.save(torch.jit.trace(model, torch.zeros(1, WIDTH)), buf)
    return torch.jit.load(io.BytesIO(buf.getvalue())), buf.getvalue()


def _inputs() -> Any:
    return np.random.default_rng(1).normal(size=(ROWS, WIDTH))


def _record(order: str | None) -> tuple[Session, Any]:
    _scripted, payload = _model()
    x = _inputs()
    session = Session(AwkwardBackend())
    ev = from_awkward("ev", ak.Array({f"c{i}": x[:, i] for i in range(WIDTH)}), session=session)
    params = None if order is None else {"order": order}
    out = record_external(
        session, PYTORCH_PLUGIN, payload, [ev[f"c{i}"] for i in range(WIDTH)], params=params
    )
    return session, out


def _eager(matrix: Any) -> Any:
    scripted, _payload = _model()
    with torch.no_grad():
        return np.asarray(scripted(torch.FloatTensor(matrix)).numpy(), dtype=np.float64)[:, 0]


def test_this_batch_rounds_differently_by_layout() -> None:
    x = _inputs()
    assert not np.array_equal(_eager(np.ascontiguousarray(x)), _eager(np.asfortranarray(x)))


def test_order_f_equals_a_pandas_fed_evaluation_bit_for_bit() -> None:
    session, out = _record("F")
    got = ak.to_numpy(session.materialize(out))
    assert np.array_equal(got, _eager(np.asfortranarray(_inputs())))


@pytest.mark.parametrize("order", [None, "C"])
def test_the_default_feeds_a_row_major_matrix_as_before(order: str | None) -> None:
    session, out = _record(order)
    got = ak.to_numpy(session.materialize(out))
    assert np.array_equal(got, _eager(np.ascontiguousarray(_inputs())))


def test_the_default_records_no_order_and_order_f_is_another_node() -> None:
    session, out = _record(None)
    params = session._nodes()[out.node_id]["params"]
    assert "order" not in params
    _session_f, out_f = _record("F")
    params_f = _session_f._nodes()[out_f.node_id]["params"]
    assert params_f["order"] == "F"
    assert {k: v for k, v in params_f.items() if k != "order"} == params


def test_the_two_orders_are_two_nodes_in_one_session() -> None:
    _scripted, payload = _model()
    session = Session(AwkwardBackend())
    ev = from_awkward("ev", ak.Array({"a": [1.0], "b": [2.0]}), session=session)

    def record(params: dict[str, str] | None) -> Any:
        return record_external(session, PYTORCH_PLUGIN, payload, [ev.a, ev.b], params=params)

    assert record(None).node_id == record(None).node_id != record({"order": "F"}).node_id


def test_an_unknown_order_is_refused_by_name() -> None:
    with pytest.raises(GraphedTypeError, match="order='A'"):
        _record("A")
