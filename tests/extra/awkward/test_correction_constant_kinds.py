"""A correction refuses at the recording line a constant its input's type rejects (#311)."""

from __future__ import annotations

import json
from typing import Any

import awkward as ak
import correctionlib
import numpy as np
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.errors import GraphedTypeError
from graphed.preserve.externals import record_external
from graphed.preserve.externals.correctionlib_external import CORRECTIONLIB_PLUGIN

X = [1.0, 2.0]


def _payload(kind: str) -> bytes:
    if kind == "real":
        data: dict[str, Any] = {
            "nodetype": "formula",
            "expression": "x*y",
            "parser": "TFormula",
            "variables": ["k", "x"],
        }
    else:
        key: object = "a" if kind == "string" else 1
        data = {
            "nodetype": "category",
            "input": "k",
            "content": [{"key": key, "value": 1.0}],
            "default": 2.0,
        }
    doc = {
        "schema_version": 2,
        "corrections": [
            {
                "name": "c",
                "version": 1,
                "inputs": [{"name": "k", "type": kind}, {"name": "x", "type": "real"}],
                "output": {"name": "w", "type": "real"},
                "data": data,
            }
        ],
    }
    return json.dumps(doc).encode()


def _accepted(kind: str, constant: object) -> list[float] | None:
    corr = correctionlib.CorrectionSet.from_string(_payload(kind).decode())["c"]
    try:
        return list(corr.evaluate(constant, ak.Array(X)))
    except RuntimeError:
        return None


RECORDS = {
    "args": lambda p, x, k: gak.apply_correction(p, "c", [x], args=[k, "$0"]),
    "keywords": lambda p, x, k: gak.apply_correction(p, "c", k=k, x=x),
    "record_external": lambda p, x, k: record_external(
        x.session, CORRECTIONLIB_PLUGIN, p, [x], params={"name": "c", "args": [k, "$0"]}
    ),
}


@pytest.mark.parametrize("record", RECORDS.values(), ids=RECORDS.keys())
@pytest.mark.parametrize("constant", ["a", 1, 1.0, True], ids=["str", "int", "float", "bool"])
@pytest.mark.parametrize("kind", ["real", "int", "string"])
def test_a_constant_records_exactly_when_correctionlib_takes_it(
    kind: str, constant: object, record: Any
) -> None:
    session = Session(AwkwardBackend())
    x = from_awkward("x", ak.Array(X), session=session)
    expected = _accepted(kind, constant)
    if expected is None:
        with pytest.raises(GraphedTypeError, match=f"input 'k' is {kind}") as info:
            record(_payload(kind), x, constant)
        assert info.value.provenance.filename == __file__
    else:
        assert (
            ak.Array(session.materialize(record(_payload(kind), x, constant))).tolist() == expected
        )


@pytest.mark.parametrize(("kind", "refused"), [("string", False), ("real", True), ("int", True)])
def test_a_systematic_call_needs_a_string_first_input(kind: str, refused: bool) -> None:
    session = Session(AwkwardBackend())
    x = from_awkward("x", ak.Array(X), session=session)
    params = {"name": "c", "systematic": "a"}
    if refused:
        with pytest.raises(GraphedTypeError, match=f"input 'k' is {kind}; got 'a'"):
            record_external(session, CORRECTIONLIB_PLUGIN, _payload(kind), [x], params=params)
    else:
        out = record_external(session, CORRECTIONLIB_PLUGIN, _payload(kind), [x], params=params)
        assert ak.Array(session.materialize(out)).tolist() == [1.0, 1.0]


@pytest.mark.parametrize(
    ("kind", "constant", "refused"),
    [
        ("real", np.float64(1.0), False),
        ("string", np.str_("a"), False),
        ("int", np.float64(1.0), True),
    ],
)
def test_a_numpy_constant_checks_as_the_value_it_records(
    kind: str, constant: object, refused: bool
) -> None:
    session = Session(AwkwardBackend())
    x = from_awkward("x", ak.Array(X), session=session)
    params = {"name": "c", "args": [constant, "$0"]}
    if refused:
        with pytest.raises(GraphedTypeError, match=f"input 'k' is {kind}"):
            record_external(session, CORRECTIONLIB_PLUGIN, _payload(kind), [x], params=params)
    else:
        out = record_external(session, CORRECTIONLIB_PLUGIN, _payload(kind), [x], params=params)
        assert ak.Array(session.materialize(out)).tolist() == _accepted(kind, constant)
