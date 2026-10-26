"""`gak.apply_correction`: a compound correction named like a regular one (graphed#96).

And the recorded type of a flat-then-jagged call (graphed#97), checked against eager correctionlib.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import awkward as ak
import pytest

from graphed import Array, Session
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward, gak
from graphed.preserve.errors import PreserveError
from graphed.preserve.externals import get_plugin
from graphed.preserve.externals._base import record_external

correctionlib = pytest.importorskip("correctionlib")


def _formula(name: str, expression: str, variables: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "version": 1,
        "inputs": [{"name": v, "type": "real"} for v in variables],
        "output": {"name": "sf", "type": "real"},
        "data": {
            "nodetype": "formula",
            "expression": expression,
            "parser": "TFormula",
            "variables": variables,
        },
    }


#: a regular and a compound correction both named "X" (legal: names are unique per list only); "Y"
#: is compound only
CSET = json.dumps(
    {
        "schema_version": 2,
        "corrections": [
            _formula("X", "x+100", ["pt"]),
            _formula("a", "2*x", ["pt"]),
            _formula("two", "x*y", ["rho", "pt"]),
        ],
        "compound_corrections": [
            {
                "name": "X",
                "inputs": [{"name": "pt", "type": "real"}],
                "output": {"name": "sf", "type": "real"},
                "inputs_update": [],
                "input_op": "*",
                "output_op": "*",
                "stack": ["a"],
            },
            {
                "name": "Y",
                "inputs": [{"name": "pt", "type": "real"}],
                "output": {"name": "sf", "type": "real"},
                "inputs_update": [],
                "input_op": "*",
                "output_op": "*",
                "stack": ["a"],
            },
        ],
    }
).encode("utf-8")
EAGER = correctionlib.CorrectionSet.from_string(CSET.decode())
EVENTS = ak.Array({"pt": [[10.0, 20.0], [], [30.0]], "rho": [1.0, 2.0, 3.0]})


def _unusable(*call: Any) -> Any:
    raise AssertionError("the template path evaluates through the plugin")


def test_a_compound_correction_sharing_a_regular_name_evaluates_the_compound() -> None:
    session = Session(AwkwardBackend())
    pt = from_awkward(session, "events", EVENTS).pt
    regular = gak.apply_correction(CSET, "X", [pt], _unusable, args=["$0"])
    compound = gak.apply_correction(CSET, "X", [pt], _unusable, args=["$0"], compound=True)

    assert regular.node_id != compound.node_id  # the kind is node identity, so CSE keeps both
    assert (
        ak.Array(session.materialize(compound)).tolist()
        == EAGER.compound["X"].evaluate(EVENTS.pt).tolist()
    )
    assert (
        ak.Array(session.materialize(regular)).tolist() == EAGER["X"].evaluate(EVENTS.pt).tolist()
    )
    assert (
        EAGER.compound["X"].evaluate(EVENTS.pt).tolist() != EAGER["X"].evaluate(EVENTS.pt).tolist()
    )


def test_a_plugin_recorded_compound_correction_evaluates_the_compound() -> None:
    session = Session(AwkwardBackend())
    pt = from_awkward(session, "events", EVENTS).pt
    plugin = get_plugin("correctionlib")
    assert plugin is not None
    out = record_external(
        session, plugin, CSET, [pt], params={"name": "X", "args": ["$0"], "compound": True}
    )

    assert (
        ak.Array(session.materialize(out)).tolist()
        == EAGER.compound["X"].evaluate(EVENTS.pt).tolist()
    )


def test_a_compound_only_name_without_the_flag_falls_back_to_the_compound_list() -> None:
    session = Session(AwkwardBackend())
    pt = from_awkward(session, "events", EVENTS).pt
    out = gak.apply_correction(CSET, "Y", [pt], _unusable, args=["$0"])

    assert (
        ak.Array(session.materialize(out)).tolist()
        == EAGER.compound["Y"].evaluate(EVENTS.pt).tolist()
    )


def test_the_compound_flag_on_a_regular_only_name_is_refused() -> None:
    session = Session(AwkwardBackend())
    pt = from_awkward(session, "events", EVENTS).pt
    out = gak.apply_correction(CSET, "a", [pt], _unusable, args=["$0"], compound=True)

    with pytest.raises(PreserveError, match=r"no compound correction named 'a'"):
        session.materialize(out)


def _template(rho: Array, pt: Array) -> Array:
    return gak.apply_correction(CSET, "two", [rho, pt], _unusable, args=["$0", "$1"])


def _plugin(rho: Array, pt: Array) -> Array:
    plugin = get_plugin("correctionlib")
    assert plugin is not None
    out: Array = record_external(
        rho.session, plugin, CSET, [rho, pt], params={"name": "two", "args": ["$0", "$1"]}
    )
    return out


@pytest.mark.parametrize(
    "record", [_template, _plugin], ids=["apply_correction", "record_external"]
)
def test_a_flat_then_jagged_call_records_the_broadcast_type(
    record: Callable[[Array, Array], Array],
) -> None:
    session = Session(AwkwardBackend())
    events = from_awkward(session, "events", EVENTS)
    out = record(events.rho, events.pt)
    eager = EAGER["two"].evaluate(EVENTS.rho, EVENTS.pt)

    form = session.form(out)
    assert isinstance(form, AwkwardForm)
    assert str(form.tt.type.content) == str(eager.type.content) == "var * float64"
    assert (
        ak.Array(session.materialize(gak.sum(out, axis=1))).tolist()
        == ak.sum(eager, axis=1).tolist()
    )
    assert (
        ak.Array(session.materialize(gak.num(out, axis=1))).tolist()
        == ak.num(eager, axis=1).tolist()
    )
    assert ak.Array(session.materialize(gak.flatten(out))).tolist() == ak.flatten(eager).tolist()
