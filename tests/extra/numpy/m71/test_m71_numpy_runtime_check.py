"""m71 on numpy: a declared `map`/`apply` type is checked against the value at run time — dtype, the
trailing shape a subarray pins, the leading axis of the first input, record fields."""

from __future__ import annotations

import pickle
import sys
from typing import Any

import numpy as np
import pytest

import graphed
from graphed import GraphedTypeError, Session
from graphed.aggregate import external_evaluators
from graphed.execute import compile_ir, evaluate_ir
from graphed.numpy import NumpyBackend, from_array, from_record

X = np.array([0.5, 1.5], dtype=np.float32)


def here() -> int:
    return sys._getframe(1).f_lineno


def _x() -> tuple[Session, Any]:
    s = Session(NumpyBackend())
    return s, from_array(s, "x", X)


def _value(v: Any) -> Any:
    return lambda a: v


def _refused(call: Any, line: int) -> GraphedTypeError:
    with pytest.raises(GraphedTypeError) as info:
        call()
    assert type(info.value).__name__ == "OutputTypeError"
    assert info.value.provenance.lineno == line
    return info.value


def test_a_map_value_of_another_dtype_raises_at_the_declaring_line() -> None:
    s, x = _x()
    out, line = x.map(np.sqrt, output_type="float64"), here()
    err = _refused(lambda: s.materialize(out), line)
    assert "declares output_type 'float64'; its value is 'vector[float32]'" in err.detail


def test_an_apply_value_of_another_dtype_raises_at_the_declaring_line() -> None:
    s, x = _x()
    out, line = graphed.apply(np.add, x, x, output_type=bool), here()
    assert "its value is 'vector[float32]'" in _refused(lambda: s.materialize(out), line).detail


@pytest.mark.parametrize(
    ("declared", "value", "fits"),
    [
        (("f4", (3,)), np.zeros((2, 3), dtype="f4"), True),
        (("f4", (3,)), np.zeros((2, 2), dtype="f4"), False),  # the subarray pins the trailing shape
        ("float32", np.zeros((2, 3), dtype="f4"), False),  # a plain dtype pins no trailing axes
        (str, np.array(["ab", "c"]), True),  # unsized: any length of its kind
        ("U5", np.array(["ab", "c"]), False),
        ("S0", np.array([b"ab"]), True),
        ([("pt", "f4"), ("eta", "f4")], np.zeros(2, dtype=[("pt", "f4"), ("eta", "f4")]), True),
        ([("pt", "f4"), ("eta", "f4")], np.zeros(2, dtype=[("pt", "f4"), ("phi", "f4")]), False),
        ([("pt", "f4"), ("eta", "f4")], {"pt": X, "eta": X}, True),  # the backend's own record value
        ([("pt", "f4"), ("eta", "f4")], {"pt": X}, False),
        ("float32", {"pt": X}, False),
    ],
)
def test_what_matches_at_the_edges(declared: Any, value: Any, fits: bool) -> None:
    s, x = _x()
    out, line = x.map(_value(value), name=f"edge {declared}", output_type=declared), here()
    if fits:
        assert s.materialize(out) is value
    else:
        _refused(lambda: s.materialize(out), line)


def test_a_scalar_input_takes_no_leading_axis() -> None:
    s, x = _x()
    total = x.reduce("sum")
    assert s.materialize(total.map(_value(np.bool_(True)), name="s", output_type=bool)) == np.bool_(True)
    out, line = total.map(_value(np.array([True])), name="v", output_type=bool), here()
    assert "its value is 'vector[bool]'" in _refused(lambda: s.materialize(out), line).detail


def _a_above_one(t: Any) -> Any:
    return t["a"] > 1


def test_a_record_input_takes_the_leading_axis() -> None:
    s = Session(NumpyBackend())
    rec = from_record(s, "r", a=np.array([0.5, 2.0]), b=np.array([1, 2]))
    value = np.asarray(s.materialize(rec.map(_a_above_one, name="n", output_type=bool)))
    assert value.tolist() == [False, True]


def test_evaluate_ir_raises_the_located_error_and_it_pickles() -> None:
    s, x = _x()
    out, line = x.map(np.sqrt, output_type="float64"), here()
    compiled = compile_ir(s, out)
    err = _refused(
        lambda: evaluate_ir(compiled, NumpyBackend(), {"x": X}, externals=external_evaluators(s, compiled)),
        line,
    )
    clone = pickle.loads(pickle.dumps(err))
    assert (type(clone), clone.op, clone.provenance, clone.detail) == (
        type(err),
        err.op,
        err.provenance,
        err.detail,
    )


def test_numpy_records_and_checks_no_output_dtype() -> None:
    s, x = _x()
    out = s.record_external("map", np.sqrt, [x], {"fn": "sqrt"}, form_params={"output_dtype": "float64"})
    assert s.form(out).describe() == "vector[object]"
    assert np.asarray(s.materialize(out)).dtype == np.float32
