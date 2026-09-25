"""m71: a declared type the installed awkward parser does not rebuild as itself (a parameterised
``float16`` on awkward 2.14 reads back as an empty named tuple) is refused, never recorded wrong."""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pytest
from awkward.types import ListType, NumpyType

from graphed import GraphedTypeError, Session
from graphed.awkward import AwkwardBackend, from_awkward

HALF = NumpyType("float16", parameters={"k": "v"})
EVENTS = ak.zip(
    {
        "x": np.array([0.5, 1.5, 2.5], dtype=np.float32),
        "Jet": ak.zip({"pt": ak.Array([[30.0, 20.0], [], [40.0]])}),
    },
    depth_limit=1,
)


def _parser_rebuilds_parameterised_float16() -> bool:
    spelling = str(HALF)
    try:
        return str(ak.types.from_datashape(spelling, highlevel=False)) == spelling
    except Exception:
        return False


@pytest.mark.parametrize(("field", "spec"), [("x", HALF), ("Jet", ListType(HALF))])
def test_a_parameterised_float16_records_its_eager_type_or_is_refused(field: str, spec: Any) -> None:
    session = Session(AwkwardBackend())
    ev = from_awkward(session, "events", EVENTS)
    column = ev.x if field == "x" else ev.Jet.pt
    eager_column = EVENTS.x if field == "x" else EVENTS.Jet.pt

    def fn(a: Any) -> Any:
        return ak.enforce_type(ak.values_astype(a, "f2"), spec)

    if _parser_rebuilds_parameterised_float16():
        out = column.map(fn, name="h16p", output_type=spec)
        assert str(session.form(out).describe()) == f"## * {spec}"
        assert str(ak.type(session.materialize(out))) == str(ak.type(fn(eager_column)))
    else:
        with pytest.raises(GraphedTypeError) as info:
            column.map(fn, name="h16p", output_type=spec)
        assert repr(spec) in info.value.detail
