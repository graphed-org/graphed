"""An External's recorded type keeps the parameters its value keeps (#248, #310)."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

import awkward as ak
import correctionlib
import numpy as np
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.errors import GraphedTypeError
from graphed.preserve import (
    JAX_PLUGIN,
    ONNX_PLUGIN,
    PYTORCH_PLUGIN,
    TENSORFLOW_PLUGIN,
    XGBOOST_PLUGIN,
)
from graphed.preserve.externals import record_external
from graphed.preserve.externals.correctionlib_external import CORRECTIONLIB_PLUGIN

DOUBLE = json.dumps(
    {
        "schema_version": 2,
        "corrections": [
            {
                "name": "c",
                "version": 1,
                "inputs": [{"name": "x", "type": "real"}],
                "output": {"name": "w", "type": "real"},
                "data": {
                    "nodetype": "formula",
                    "expression": "2*x",
                    "parser": "TFormula",
                    "variables": ["x"],
                },
            }
        ],
    }
).encode()
JAGGED = ak.Array([[1.0, 2.0], [], [3.0]]).layout


def _types(session: Session, out: Any) -> tuple[str, str]:
    recorded = str(session.form(out).describe()).removeprefix("## * ")
    return recorded, str(ak.type(session.materialize(out)).content)


_product = json.loads(DOUBLE)
_product["corrections"][0]["inputs"].append({"name": "y", "type": "real"})
_product["corrections"][0]["data"].update(expression="x*y", variables=["x", "y"])
PRODUCT = json.dumps(_product).encode()


def _jagged(leaf: Any = None, lists: Any = None) -> ak.Array:
    content = ak.contents.NumpyArray(JAGGED.content.data, parameters=leaf)
    return ak.Array(ak.contents.ListOffsetArray(JAGGED.offsets, content, parameters=lists))


def _option(array: ak.Array, parameters: Any = None) -> ak.Array:
    index = ak.index.Index64(np.arange(len(array)))
    return ak.Array(ak.contents.IndexedOptionArray(index, array.layout, parameters=parameters))


FLAT = ak.Array([1.0, 2.0, 3.0])
SHAPES = {
    "leaf": [_jagged(leaf={"typename": "float[]"})],
    "list": [_jagged(lists={"__doc__": "a"})],
    "differing-lists": [_jagged(lists={"__doc__": "a"}), _jagged(lists={"__doc__": "b"})],
    "list-and-flat": [_jagged(lists={"__doc__": "a"}), FLAT],
    "flat-and-list": [FLAT, _jagged(lists={"__doc__": "a"})],
    "list-and-option": [_jagged(lists={"__doc__": "a"}), ak.Array([1.0, None, 3.0])],
    "shared-list": [_jagged(lists={"__doc__": "a"}), _jagged(lists={"__doc__": "a"})],
    "multidimensional-leaf": [
        ak.Array(ak.contents.NumpyArray(np.ones((3, 2)), parameters={"typename": "f"}))
    ],
    "option-list": [_option(_jagged(lists={"__doc__": "a"}))],
    "option-parameter": [_option(FLAT, {"__doc__": "o"})],
}


def _cset_type(arrays: list[ak.Array]) -> str:
    cset = correctionlib.CorrectionSet.from_string(
        (PRODUCT if len(arrays) == 2 else DOUBLE).decode()
    )
    return str(ak.type(cset["c"].evaluate(*arrays)).content)


def _record_plugin(session: Session, xs: list[Any]) -> Any:
    payload = PRODUCT if len(xs) == 2 else DOUBLE
    args = [f"${i}" for i in range(len(xs))]
    return record_external(
        session, CORRECTIONLIB_PLUGIN, payload, xs, params={"name": "c", "args": args}
    )


@pytest.mark.parametrize(
    "record",
    [
        lambda s, xs: gak.apply_correction(PRODUCT if len(xs) == 2 else DOUBLE, "c", xs),
        _record_plugin,
    ],
    ids=["apply_correction", "record_external"],
)
@pytest.mark.parametrize("arrays", SHAPES.values(), ids=SHAPES.keys())
def test_a_correction_records_the_type_the_installed_correctionlib_gives(
    arrays: list[ak.Array], record: Any
) -> None:
    session = Session(AwkwardBackend())
    xs = [from_awkward(f"x{i}", a, session=session) for i, a in enumerate(arrays)]
    expected = _cset_type(arrays)
    assert _types(session, record(session, xs)) == (expected, expected)


def test_a_scalar_input_of_a_correction_broadcasts_as_a_number() -> None:
    session = Session(AwkwardBackend())
    x = from_awkward("x", _jagged(lists={"__doc__": "a"}), session=session)
    total = gak.sum(from_awkward("n", FLAT, session=session))
    expected = _cset_type([_jagged(lists={"__doc__": "a"}), np.float64(6.0)])
    assert _types(session, gak.apply_correction(PRODUCT, "c", [x, total])) == (expected, expected)


@pytest.mark.parametrize(
    "record",
    [lambda s, x: gak.apply_correction(DOUBLE, "c", [x]), lambda s, x: _record_plugin(s, [x])],
    ids=["apply_correction", "record_external"],
)
def test_an_input_the_correction_cannot_take_is_refused_where_it_is_recorded(record: Any) -> None:
    session = Session(AwkwardBackend())
    x = from_awkward("x", ak.Array(["a", "b"]), session=session)
    with pytest.raises(GraphedTypeError, match="string") as info:
        record(session, x)
    assert info.value.provenance.filename == __file__


SYSTEMATIC = json.dumps(
    {
        "schema_version": 2,
        "corrections": [
            {
                "name": "sf",
                "version": 1,
                "inputs": [{"name": "s", "type": "string"}, {"name": "x", "type": "real"}],
                "output": {"name": "w", "type": "real"},
                "data": {
                    "nodetype": "category",
                    "input": "s",
                    "content": [{"key": "nominal", "value": 1.5}],
                },
            }
        ],
    }
).encode()


def test_a_systematic_call_records_the_flat_value_it_gives_an_option_input() -> None:
    session = Session(AwkwardBackend())
    x = from_awkward("x", ak.Array([1.0, None, 3.0]), session=session)
    out = record_external(
        session,
        CORRECTIONLIB_PLUGIN,
        SYSTEMATIC,
        [x],
        params={"name": "sf", "systematic": "nominal"},
    )
    assert _types(session, out) == ("float64", "float64")


def _onnx() -> bytes:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "two"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n", 1])],
        [numpy_helper.from_array(np.array([2.0], dtype="float32"), "two")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return bytes(model.SerializeToString())


def _xgboost() -> bytes:
    xgb = pytest.importorskip("xgboost")
    data = xgb.DMatrix(np.arange(8.0).reshape(-1, 1), label=np.arange(8.0))
    return bytes(xgb.train({"max_depth": 1}, data, 1).save_raw("json"))


def _jax() -> bytes:
    jax = pytest.importorskip("jax")
    from jax import export  # noqa: PLC0415

    spec = jax.ShapeDtypeStruct(export.symbolic_shape("b, 1"), jax.numpy.float32)
    return bytes(export.export(jax.jit(lambda x: x * 2))(spec).serialize())


def _pytorch() -> bytes:
    torch = pytest.importorskip("torch")
    buf = io.BytesIO()
    torch.jit.save(torch.jit.trace(torch.nn.Linear(1, 1), torch.zeros(1, 1)), buf)
    return buf.getvalue()


def _tensorflow() -> bytes:
    tf = pytest.importorskip("tensorflow")
    x = tf.keras.Input((1,))
    model = tf.keras.Model(x, tf.keras.layers.Dense(1)(x))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.keras"
        model.save(path)
        return path.read_bytes()


@pytest.mark.parametrize(
    ("plugin", "build"),
    [
        (ONNX_PLUGIN, _onnx),
        (XGBOOST_PLUGIN, _xgboost),
        (JAX_PLUGIN, _jax),
        (PYTORCH_PLUGIN, _pytorch),
        (TENSORFLOW_PLUGIN, _tensorflow),
    ],
    ids=["onnx", "xgboost", "jax", "pytorch", "tensorflow"],
)
def test_a_model_output_drops_its_inputs_leaf_parameters(plugin: Any, build: Any) -> None:
    payload = build()
    session = Session(AwkwardBackend())
    x = from_awkward(
        "x", ak.with_parameter(ak.Array([1.0, 2.0, 3.0]), "typename", "float"), session=session
    )
    assert _types(session, record_external(session, plugin, payload, [x])) == (
        "float64",
        "float64",
    )
