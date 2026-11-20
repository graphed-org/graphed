"""A model External records the row shape its payload states, and execution checks it (#250)."""

from __future__ import annotations

import io
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.debug import StageError
from graphed.errors import GraphedTypeError
from graphed.preserve import (
    JAX_PLUGIN,
    ONNX_PLUGIN,
    PYTORCH_PLUGIN,
    TENSORFLOW_PLUGIN,
    XGBOOST_PLUGIN,
)
from graphed.preserve.externals import record_external
from graphed.preserve.externals._helpers import _HASH_MEMO
from graphed.session import CheckedExternal

TEMPLATE = {"args": [["$0", "$1"]]}
X = np.random.default_rng(0).normal(size=(40, 2))


@pytest.fixture(autouse=True)
def _cold_memo() -> None:
    _HASH_MEMO.clear()


def _inputs(session: Session) -> list[Any]:
    """Two option-typed feature columns with no missing value: the model's value has no option."""
    col = ak.mask(ak.Array([1.0, 2.0, 3.0]), [True, True, True])
    ev = from_awkward("ev", ak.zip({"a": col, "b": col * 2}), session=session)
    return [ev.a, ev.b]


def _recorded(session: Session, out: Any) -> str:
    return str(session.form(out).describe()).removeprefix("## * ")


def _types(session: Session, out: Any) -> tuple[str, str]:
    return _recorded(session, out), str(ak.type(session.materialize(out)).content)


def _key(session: Session, out: Any) -> str:
    fn = session._externals[out.node_id][0]
    assert isinstance(fn, CheckedExternal)
    return fn.key


def _onnx(k: int, out_dims: list[Any] | None = None) -> bytes:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, out_dims or [None, k])],
        [numpy_helper.from_array(np.ones((2, k), dtype="float32"), "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return bytes(model.SerializeToString())


def _xgboost(params: dict[str, Any], label: Any) -> bytes:
    xgb = pytest.importorskip("xgboost")
    return bytes(xgb.train(params, xgb.DMatrix(X, label=label), 2).save_raw("json"))


def _jax(k: int) -> bytes:
    jax = pytest.importorskip("jax")
    from jax import export  # noqa: PLC0415

    spec = jax.ShapeDtypeStruct(export.symbolic_shape("b, 2"), jax.numpy.float32)
    w: Any = np.ones((2, k), dtype="float32")
    return bytes(export.export(jax.jit(lambda x: x @ w))(spec).serialize())


def _tensorflow(k: int) -> bytes:
    keras = pytest.importorskip("keras")
    model = keras.Sequential([keras.layers.Input(shape=(2,)), keras.layers.Dense(k)])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.keras"
        model.save(path)
        return path.read_bytes()


def _pytorch(k: int) -> bytes:
    torch = pytest.importorskip("torch")
    buf = io.BytesIO()
    torch.jit.save(torch.jit.trace(torch.nn.Linear(2, k), torch.zeros(1, 2)), buf)
    return buf.getvalue()


STATED: list[tuple[Any, Callable[[], bytes], dict[str, Any], str]] = [
    (ONNX_PLUGIN, lambda: _onnx(3), TEMPLATE, "3 * float64"),
    (ONNX_PLUGIN, lambda: _onnx(1), TEMPLATE, "float64"),
    (
        XGBOOST_PLUGIN,
        lambda: _xgboost({"objective": "multi:softprob", "num_class": 3}, np.arange(40) % 3),
        {},
        "3 * float64",
    ),
    (
        XGBOOST_PLUGIN,
        lambda: _xgboost({"objective": "multi:softmax", "num_class": 3}, np.arange(40) % 3),
        {},
        "float64",
    ),
    (XGBOOST_PLUGIN, lambda: _xgboost({"tree_method": "hist"}, X), {}, "2 * float64"),
    (JAX_PLUGIN, lambda: _jax(3), {}, "3 * float64"),
    (TENSORFLOW_PLUGIN, lambda: _tensorflow(3), {}, "3 * float64"),
]


@pytest.mark.parametrize(
    ("plugin", "build", "params", "expected"),
    STATED,
    ids=["onnx", "onnx-1", "xgb-softprob", "xgb-softmax", "xgb-targets", "jax", "tensorflow"],
)
def test_a_stated_row_shape_is_the_recorded_type_and_the_value(
    plugin: Any, build: Callable[[], bytes], params: dict[str, Any], expected: str
) -> None:
    session = Session(AwkwardBackend())
    out = record_external(session, plugin, build(), _inputs(session), params=params)
    assert _types(session, out) == (expected, expected)
    assert _key(session, out) == "array output_type"


@pytest.mark.parametrize(
    ("plugin", "build", "params"),
    [
        (PYTORCH_PLUGIN, lambda: _pytorch(1), {}),
        (ONNX_PLUGIN, lambda: _onnx(1, ["n", "k"]), TEMPLATE),
        (ONNX_PLUGIN, lambda: _onnx(1, [None, None]), TEMPLATE),
    ],
    ids=["pytorch", "onnx-symbolic", "onnx-unknown"],
)
def test_an_unstated_row_shape_records_one_value_per_event_checking_leaves(
    plugin: Any, build: Callable[[], bytes], params: dict[str, Any]
) -> None:
    session = Session(AwkwardBackend())
    out = record_external(session, plugin, build(), _inputs(session), params=params)
    assert _types(session, out) == ("float64", "float64")
    assert _key(session, out) == "output_dtype"


def _no_output_onnx() -> bytes:
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper  # noqa: PLC0415

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2])
    model = helper.make_model(
        helper.make_graph([], "g", [x], []), opset_imports=[helper.make_opsetid("", 17)]
    )
    return bytes(model.SerializeToString())


def _no_output_jax() -> bytes:
    jax = pytest.importorskip("jax")
    from jax import export  # noqa: PLC0415

    spec = jax.ShapeDtypeStruct(export.symbolic_shape("b, 2"), jax.numpy.float32)
    return bytes(export.export(jax.jit(lambda x: ()))(spec).serialize())


@pytest.mark.parametrize(
    ("plugin", "build", "params"),
    [(ONNX_PLUGIN, _no_output_onnx, TEMPLATE), (JAX_PLUGIN, _no_output_jax, {})],
    ids=["onnx", "jax"],
)
def test_a_model_with_no_output_records_a_leaf_claim(
    plugin: Any, build: Callable[[], bytes], params: dict[str, Any]
) -> None:
    session = Session(AwkwardBackend())
    out = record_external(session, plugin, build(), _inputs(session), params=params)
    assert (_recorded(session, out), _key(session, out)) == ("float64", "output_dtype")


def test_a_model_stating_a_width_it_does_not_give_is_refused() -> None:
    session = Session(AwkwardBackend())
    out = record_external(
        session, ONNX_PLUGIN, _onnx(2, [None, 3]), _inputs(session), params=TEMPLATE
    )
    assert _recorded(session, out) == "3 * float64"  # as the payload states
    with pytest.raises(StageError) as info:
        session.materialize(out)
    assert (
        "declares array output_type '3 * float64'; its value is '2 * float64'"
        in info.value.cause_message
    )


@pytest.mark.parametrize(
    ("dims", "expected", "key"),
    [
        ([None, 3], "3 * float64", "array output_type"),
        (["n", "k"], "float64", "output_dtype"),
    ],
    ids=["stated", "symbolic"],
)
def test_gak_onnx_inference_records_what_the_plugin_records(
    dims: list[Any], expected: str, key: str
) -> None:
    payload = _onnx(dims[1] if isinstance(dims[1], int) else 1, dims)
    session = Session(AwkwardBackend())
    from graphed.preserve.externals.onnx_external import eval_onnx, load_onnx  # noqa: PLC0415

    model = load_onnx(payload, {})
    runner = lambda x: eval_onnx(model, TEMPLATE, [x[:, 0], x[:, 1]])  # noqa: E731
    out = gak.onnx_inference(payload, _inputs(session), runner, args=[["$0", "$1"]])
    assert _types(session, out) == (expected, expected)
    assert _key(session, out) == key


def _count(monkeypatch: pytest.MonkeyPatch, module: Any, name: str, payload: bytes) -> list[None]:
    calls: list[None] = []
    real = getattr(module, name)

    def counting(arg: Any, *args: Any, **kwargs: Any) -> Any:
        if (arg.encode() if isinstance(arg, str) else bytes(arg)) == payload:
            calls.append(None)
        return real(arg, *args, **kwargs)

    monkeypatch.setattr(module, name, counting)
    return calls


def _onnx_parser() -> tuple[Any, str]:
    return pytest.importorskip("onnx"), "load_from_string"


def _jax_parser() -> tuple[Any, str]:
    pytest.importorskip("jax")
    from jax import export  # noqa: PLC0415

    return export, "deserialize"


@pytest.mark.parametrize(
    ("plugin", "build", "params", "parser"),
    [
        (ONNX_PLUGIN, lambda: _onnx(3), TEMPLATE, _onnx_parser),
        (XGBOOST_PLUGIN, lambda: _xgboost({}, X[:, 0]), {}, lambda: (json, "loads")),
        (JAX_PLUGIN, lambda: _jax(3), {}, _jax_parser),
    ],
    ids=["onnx", "xgboost", "jax"],
)
def test_the_row_shape_is_read_in_the_hash_parse(
    monkeypatch: pytest.MonkeyPatch,
    plugin: Any,
    build: Callable[[], bytes],
    params: dict[str, Any],
    parser: Callable[[], tuple[Any, str]],
) -> None:
    payload = build()
    session = Session(AwkwardBackend())
    inputs = _inputs(session)
    calls = _count(monkeypatch, *parser(), payload)
    record_external(session, plugin, payload, inputs, params=params)
    assert len(calls) == 1


def test_gak_onnx_inference_reads_the_row_shape_in_one_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _onnx(3)
    session = Session(AwkwardBackend())
    inputs = _inputs(session)
    calls = _count(monkeypatch, *_onnx_parser(), payload)
    gak.onnx_inference(payload, inputs, lambda x: x, args=[["$0", "$1"]])
    assert len(calls) == 1


@pytest.mark.parametrize("claim", [True, False])
def test_a_callers_form_is_checked_only_when_claimed(claim: bool) -> None:
    session = Session(AwkwardBackend())
    x = _inputs(session)[0]
    out = session.record_external(
        "map",
        lambda v: ak.unflatten(v, 1),
        [x],
        {"fn": "lists"},
        descriptor=AwkwardBackend().external_payload("map", {"fn": "lists"}),
        form=session.form(x),
        form_params={"claim": claim},
    )
    if not claim:
        assert ak.to_list(session.materialize(out)) == [[1.0], [2.0], [3.0]]
        return
    with pytest.raises(StageError) as info:
        session.materialize(out)
    assert "declares array output_type '?float64'" in info.value.cause_message


class _Untyped:
    """A form the backend cannot type, as another package's own form is."""

    def describe(self) -> str:
        return "untyped"


def test_a_claimed_form_the_backend_cannot_type_is_refused_at_its_line() -> None:
    session = Session(AwkwardBackend())
    x = _inputs(session)[0]
    with pytest.raises(GraphedTypeError) as info:
        session.record_external(
            "map",
            lambda v: v,
            [x],
            {"fn": "untyped"},
            descriptor=AwkwardBackend().external_payload("map", {"fn": "untyped"}),
            form=_Untyped(),
            form_params={"claim": True},
        )
    assert info.value.provenance.filename == __file__
