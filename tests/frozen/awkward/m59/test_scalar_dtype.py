"""m59 integ-m59-S — a numpy scalar operand keeps its dtype and its exact value.

`float(value)` loses both: the dtype the operand would have imposed and, above 2**53, the value
itself. The recorded form's dtype and the executed result must equal eager awkward's, Python
operands must record exactly as they do today, and the dtype must reach the IR without the
frontend importing numpy or the store growing a `ParamValue` kind.
"""

from __future__ import annotations

import subprocess
import sys

import awkward as ak
import numpy as np
import pytest
from m59_idiom_fixtures import (
    ARRAYS,
    BOOLS,
    FLOATS,
    INTS,
    PY_SCALARS,
    SCALARS,
    dtype_of,
    eager_form,
    recorded,
    session_over,
    tracer,
)

#: a frontend that reads a scalar's dtype by importing numpy would fail the first assertion; one
#: that still collapses the dtype to a float would fail the second.
PURITY = """
import sys
import graphed
banned = [m for m in ("numpy", "awkward") if m in sys.modules]
assert not banned, banned

import numpy as np
import graphed.core


class Form:
    def describe(self):
        return "stub"


class Backend:
    def op_form(self, op, inputs, params):
        return Form()

    def boundary_ops(self):
        return frozenset()


session = graphed.Session(Backend())
a = session.source("a", form=Form(), data=None)
ids = {name: (a * value).node_id for name, value in
       (("uint64", np.uint64(1)), ("int32", np.int32(1)), ("python", 1))}
assert len(set(ids.values())) == 3, ids

graph = graphed.core.GraphStore.deserialize(session.serialized_ir(a * np.uint64(1), optimize=False))
values = [v for node in graph.nodes() for v in node["params"].values()]
assert values and all(isinstance(v, bool | int | float | str) for v in values), values
"""


@pytest.mark.parametrize("scalar", list(SCALARS))
@pytest.mark.parametrize("array", list(ARRAYS))
def test_a_numpy_scalar_operand_keeps_its_dtype_on_either_side(array: str, scalar: str) -> None:
    data, value = ARRAYS[array], SCALARS[scalar]
    session, arr = session_over(data)
    # a numpy scalar on the LEFT of a comparison is ill-typed in eager awkward itself, so the
    # comparison leg is array-side only; the arithmetic legs cover both sides.
    legs = [
        (arr * value, tracer(data) * value, data * value),
        (value * arr, value * tracer(data), value * data),
        (arr + value, tracer(data) + value, data + value),
        (value + arr, value + tracer(data), value + data),
        (arr > value, tracer(data) > value, data > value),
    ]
    if array != "bool":  # numpy forbids boolean subtract, so the asymmetric leg is int/float only
        legs += [
            (arr - value, tracer(data) - value, data - value),
            (value - arr, value - tracer(data), value - data),
        ]
    for out, abstract, concrete in legs:
        assert session.form(out).describe() == eager_form(abstract)
        got = session.materialize(out)
        assert dtype_of(got) == dtype_of(concrete)
        assert ak.to_list(got) == ak.to_list(concrete)


def test_bitwise_operators_and_ufuncs_keep_the_scalar_dtype_too() -> None:
    bools, arr_b = session_over(BOOLS)
    ints, arr_i = session_over(INTS)
    floats, arr_f = session_over(FLOATS)
    for session, out, concrete in (
        (bools, arr_b & np.bool_(True), BOOLS & np.bool_(True)),
        (bools, arr_b * np.uint64(8), BOOLS * np.uint64(8)),  # PackedSelection's bit-packing shape
        (ints, arr_i | np.int32(1), INTS | np.int32(1)),
        (ints, arr_i ^ np.int32(3), INTS ^ np.int32(3)),
        (ints, arr_i << np.int32(1), INTS << np.int32(1)),
        (ints, np.add(arr_i, np.int32(2)), np.add(INTS, np.int32(2))),
        (ints, np.subtract(np.int32(2), arr_i), np.subtract(np.int32(2), INTS)),
        (floats, np.maximum(arr_f, np.float32(4.0)), np.maximum(FLOATS, np.float32(4.0))),
    ):
        got = session.materialize(out)
        assert dtype_of(got) == dtype_of(concrete)
        assert ak.to_list(got) == ak.to_list(concrete)


def test_a_scalar_value_wider_than_a_float_round_trips_exactly() -> None:
    session, arr = session_over(BOOLS)
    for value, expected in ((np.uint64(2**63), 2**63), (np.uint64(2**64 - 1), 2**64 - 1)):
        out = arr * value
        assert session.form(out).describe() == eager_form(tracer(BOOLS) * value)
        assert dtype_of(session.materialize(out)) == "uint64"
        assert ak.to_list(session.materialize(out)) == [[expected, 0], [expected], [0, expected, expected]]

    inexact, floats = session_over(FLOATS)
    widened = inexact.materialize(floats * np.float32(0.1))
    assert ak.to_list(widened) == ak.to_list(FLOATS * np.float32(0.1))
    assert ak.to_list(widened)[0][0] == 0.10000000149011612  # float32(0.1) widened, not float64 0.1


def test_python_scalars_record_exactly_as_before() -> None:
    session, arr = session_over(INTS)
    for name, params in (
        ("int", {"scalar": 2, "side": "r"}),
        ("float", {"scalar": 0.5, "side": "r"}),
        ("bool", {"scalar": True, "side": "r"}),
    ):
        node = recorded(session, arr * PY_SCALARS[name])
        assert (node["kind"], node["name"], node["params"]) == ("op", "mul", params)
        assert session.form(arr * PY_SCALARS[name]).describe() == eager_form(tracer(INTS) * PY_SCALARS[name])
    assert recorded(session, 2 - arr)["params"] == {"scalar": 2, "side": "l"}

    other, elsewhere = session_over(INTS)
    assert session.serialized_ir(arr * 2) == other.serialized_ir(elsewhere * 2)


def test_dtype_and_value_together_decide_a_scalar_operands_node() -> None:
    _session, arr = session_over(INTS)
    one = {
        "uint64": np.uint64(1),
        "int32": np.int32(1),
        "float32": np.float32(1.0),
        "python_int": 1,
        "python_float": 1.0,
    }
    ids = {name: (arr * value).node_id for name, value in one.items()}

    assert len(set(ids.values())) == len(ids), ids
    assert (arr * np.uint64(1)).node_id == (arr * np.uint64(1)).node_id
    assert (arr * np.uint64(1)).node_id != (arr * np.uint64(2)).node_id


def test_the_frontend_still_imports_neither_numpy_nor_awkward() -> None:
    result = subprocess.run([sys.executable, "-c", PURITY], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
