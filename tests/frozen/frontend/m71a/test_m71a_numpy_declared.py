"""m71a A-C4 (numpy): a declared `output_type=` is the recorded element type, one node per type."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from m71a_numpy_fixtures import (
    M2,
    NUMERIC,
    RA,
    X,
    Y,
    describe,
    f,
    params_of,
    record_of,
    recorded,
)

import graphed


def _float64(a: Any) -> Any:
    return a.astype("float64")


def test_a_float64_declaration_in_every_spelling_is_one_node() -> None:
    sn, x, _y, _m2 = recorded()
    n = x.map(_float64, name="n", output_type="f8")

    assert describe(sn, n) == "vector[float64]"
    assert params_of(sn, n)["output_type"] == "float64"
    for spec in ("float64", np.float64, np.dtype("f8"), float):
        assert x.map(_float64, name="n", output_type=spec).node_id == n.node_id
    value = sn.materialize(n)
    assert isinstance(value, np.ndarray) and value.dtype == np.float64
    np.testing.assert_array_equal(value, X.astype("float64"))
    assert describe(sn, n.reduce("sum")).startswith("scalar[")


def test_each_dtype_family_is_one_node_and_families_are_distinct() -> None:
    sn, x, _y, _m2 = recorded()
    families: dict[str, list[Any]] = {
        "bool": [bool, np.bool_, "?", "bool"],
        "int64": [int, np.int64, "int64"],
        "<U5": ["U5", np.dtype("U5"), "<U5"],
        "S5": ["S5", np.dtype("S5")],
        "<U0": [str],
        "object": [object],
        "float64": [float],
    }
    first: dict[str, Any] = {}
    for key, spellings in families.items():
        first[key] = x.map(f, name="n", output_type=spellings[0])
        for spec in spellings[1:]:
            assert x.map(f, name="n", output_type=spec).node_id == first[key].node_id, (key, spec)
    ids = {array.node_id for array in first.values()}
    assert len(ids) == len(families)

    assert params_of(sn, first["bool"])["output_type"] == "bool"
    assert params_of(sn, first["int64"])["output_type"] == "int64"
    assert params_of(sn, first["<U5"])["output_type"] == "<U5"
    assert describe(sn, first["<U5"]) == "vector[<U5]"
    assert params_of(sn, first["<U0"])["output_type"] == "<U0"
    assert params_of(sn, first["object"])["output_type"] == "object"
    assert describe(sn, first["object"]) == "vector[object]"

    undeclared = x.map(f, name="n")
    assert describe(sn, undeclared) == "vector[object]"
    assert "output_type" not in params_of(sn, undeclared)
    assert undeclared.node_id not in ids


@pytest.mark.parametrize("name", NUMERIC)
def test_every_numerical_dtype_is_declarable(name: str) -> None:
    sn, x, _y, _m2 = recorded()
    scalar_type = np.bool_ if name == "bool" else getattr(np, name)
    out = x.map(lambda a: a.astype(name), name="num", output_type=name)

    assert describe(sn, out) == f"vector[{name}]"
    assert params_of(sn, out)["output_type"] == name
    assert x.map(lambda a: a.astype(name), name="num", output_type=np.dtype(name)).node_id == out.node_id
    assert x.map(lambda a: a.astype(name), name="num", output_type=scalar_type).node_id == out.node_id
    assert x.map(lambda a: a.astype(name), name="num").node_id != out.node_id
    value = sn.materialize(out)
    assert isinstance(value, np.ndarray) and value.dtype == np.dtype(name)
    np.testing.assert_array_equal(value, X.astype(name))


def test_numerical_dtypes_are_pairwise_distinct_nodes() -> None:
    _sn, x, _y, _m2 = recorded()
    ids = {x.map(lambda a: a, name="num", output_type=name).node_id for name in NUMERIC}
    assert len(ids) == len(NUMERIC)


def test_a_structured_dtype_records_a_record_whose_fields_are_typed() -> None:
    sn, x, _y, _m2 = recorded()
    spec = [("pt", "f4"), ("eta", "f4")]

    def pack(a: Any) -> Any:
        out = np.zeros(len(a), dtype=spec)
        out["pt"], out["eta"] = a, a * 2
        return out

    rec = x.map(pack, name="n", output_type=spec)
    assert x.map(pack, name="n", output_type=np.dtype(spec)).node_id == rec.node_id
    assert describe(sn, rec) == "record[pt,eta]"
    eta = rec["eta"]
    assert describe(sn, eta) == "vector[float32]"
    np.testing.assert_array_equal(np.asarray(sn.materialize(eta)), X * 2)


def test_a_subarray_dtype_records_a_trailing_shape() -> None:
    sn, x, _y, _m2 = recorded()
    out = x.map(lambda a: np.repeat(a[:, None], 3, axis=1), name="n", output_type=("f4", (3,)))

    assert describe(sn, out) == "vector[float32, shape=(None, 3)]"
    value = np.asarray(sn.materialize(out))
    assert value.shape == (2, 3)
    np.testing.assert_array_equal(value, np.repeat(X[:, None], 3, axis=1))


def test_apply_scalar_2d_and_record_inputs_take_the_declared_element() -> None:
    sn, x, y, m2 = recorded()

    both = graphed.apply(lambda a, b: a > b, x, y, name="n2", output_type="bool")
    assert describe(sn, both) == "vector[bool]"
    np.testing.assert_array_equal(np.asarray(sn.materialize(both)), X > Y)

    total = x.reduce("sum")
    assert describe(sn, total.map(f, name="n3", output_type="bool")) == "scalar[bool]"
    assert describe(sn, total.map(lambda a: str(a), name="n3", output_type="U5")) == "scalar[<U5]"

    rows = m2.map(lambda a: a.sum(axis=1) > 1, name="n4", output_type="bool")
    assert describe(sn, rows) == "vector[bool]"
    np.testing.assert_array_equal(np.asarray(sn.materialize(rows)), M2.sum(axis=1) > 1)

    picked = record_of(sn).map(lambda t: t["a"] > 1, name="n5", output_type="bool")
    assert describe(sn, picked) == "vector[bool]"
    np.testing.assert_array_equal(np.asarray(sn.materialize(picked)), RA > 1)


def _program() -> bytes:
    sn, x, y, m2 = recorded()
    outs = [
        x.map(_float64, name="n", output_type="f8"),
        x.map(f, name="n", output_type=bool),
        x.map(f, name="n", output_type="U5"),
        x.map(f, name="n", output_type=[("pt", "f4"), ("eta", "f4")]),
        x.map(f, name="n", output_type=("f4", (3,))),
        graphed.apply(lambda a, b: a > b, x, y, name="n2", output_type="bool"),
        x.reduce("sum").map(f, name="n3", output_type="bool"),
        m2.map(f, name="n4", output_type="bool"),
        record_of(sn).map(f, name="n5", output_type="bool"),
        *(x.map(f, name="num", output_type=name) for name in NUMERIC),
    ]
    return sn.serialized_ir(*outs, optimize=False)


def test_declared_recordings_serialize_byte_identically_across_sessions() -> None:
    first, second = _program(), _program()
    assert first == second
    assert b"float16" in first and b"<U5" in first
