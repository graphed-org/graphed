"""#144: `gak.apply_correction` keyword form, and mistakes refused at the recording line."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.awkward._templates import _payload_bytes
from graphed.errors import GraphedTypeError
from graphed.varied import member_of

pytest.importorskip("correctionlib")

_INPUTS = [
    {"name": "syst", "type": "string"},
    {"name": "pt", "type": "real"},
    {"name": "eta", "type": "real"},
]
_BY_SYST = {
    "nodetype": "category",
    "input": "syst",
    "content": [
        {"key": "nominal", "value": {"nodetype": "formula", "expression": "x+y",
                                     "parser": "TFormula", "variables": ["pt", "eta"]}},
        {"key": "up", "value": {"nodetype": "formula", "expression": "2*x+y",
                                "parser": "TFormula", "variables": ["pt", "eta"]}},
    ],
}  # fmt: skip
_SCALE = {"nodetype": "formula", "expression": "x", "parser": "TFormula", "variables": ["pt"]}
_CORRECTION = {"version": 1, "output": {"name": "sf", "type": "real"}}
CSET = json.dumps(
    {
        "schema_version": 2,
        "corrections": [
            {**_CORRECTION, "name": "jet_sf", "inputs": _INPUTS, "data": _BY_SYST},
            {
                **_CORRECTION,
                "name": "l1",
                "inputs": [{"name": "pt", "type": "real"}],
                "data": _SCALE,
            },
        ],
        "compound_corrections": [
            {
                "name": "stack",
                "output": {"name": "sf", "type": "real"},
                "inputs": [{"name": "pt", "type": "real"}],
                "inputs_update": [],
                "input_op": "*",
                "output_op": "*",
                "stack": ["l1"],
            }
        ],
    }
).encode()


def _events() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    events = from_awkward(
        "events",
        ak.Array({"pt": [[1.0, 2.0], [], [3.0]], "eta": [[0.5, 0.25], [], [0.125]]}),
        session=session,
    )
    return session, events


def _values(session: Session, array: Any) -> Any:
    return ak.to_list(session.materialize(array))


def _refused(call: Any) -> GraphedTypeError:
    with pytest.raises(GraphedTypeError) as caught:
        call()
    return caught.value


def test_a_wrong_name_is_refused_at_the_recording_line_listing_the_names() -> None:
    session, ev = _events()
    pt, eta = ev.pt, ev.eta
    before = session.node_count()
    line = sys._getframe().f_lineno + 1
    exc = _refused(lambda: gak.apply_correction(CSET, "jet_SF", syst="up", pt=pt, eta=eta))
    assert "no correction 'jet_SF' in this set; it has: jet_sf, l1, stack" in str(exc)
    assert (exc.provenance.filename, exc.provenance.lineno) == (__file__, line)
    assert session.node_count() == before


@pytest.mark.parametrize(
    "call",
    [
        lambda ev: gak.apply_correction(CSET, "nope", [ev.pt]),
        lambda ev: gak.apply_correction(CSET, "nope", [ev.pt], None, args=["$0"]),
    ],
)
def test_a_wrong_name_is_refused_on_the_positional_forms(call: Any) -> None:
    _, ev = _events()
    assert "no correction 'nope'" in str(_refused(lambda: call(ev)))


def test_compound_true_searches_only_the_compound_list() -> None:
    _, ev = _events()
    exc = _refused(lambda: gak.apply_correction(CSET, "l1", compound=True, pt=ev.pt))
    assert "no compound correction 'l1' in this set; it has: stack" in str(exc)


def test_a_compound_only_name_resolves_with_and_without_the_flag() -> None:
    session, ev = _events()
    bare = gak.apply_correction(CSET, "stack", pt=ev.pt)
    flagged = gak.apply_correction(CSET, "stack", compound=True, pt=ev.pt)
    assert _values(session, bare) == _values(session, flagged)
    assert _values(session, bare) == [[1.0, 2.0], [], [3.0]]


@pytest.mark.parametrize(
    ("given", "got"),
    [
        ({"pt": "pt"}, "(pt)"),
        ({"syst": "up", "pt": "pt", "etaa": "eta"}, "(syst, pt, etaa)"),
        ({"syst": "up", "pt": "pt", "eta": "eta", "extra": "eta"}, "(syst, pt, eta, extra)"),
    ],
)
def test_wrong_keywords_are_refused_naming_both_lists(given: dict[str, str], got: str) -> None:
    _, ev = _events()
    named: dict[str, Any] = {k: (v if v == "up" else getattr(ev, v)) for k, v in given.items()}
    exc = _refused(lambda: gak.apply_correction(CSET, "jet_sf", **named))
    assert f"correction 'jet_sf' takes inputs (syst, pt, eta); got {got}" in str(exc)


@pytest.mark.parametrize(
    "call",
    [
        lambda ev: gak.apply_correction(CSET, "jet_sf", [ev.pt]),
        lambda ev: gak.apply_correction(CSET, "jet_sf", [ev.pt], None, args=["$0"]),
        lambda ev: gak.apply_correction(CSET, "jet_sf", [ev.pt], len, args=["$0"]),
    ],
)
def test_a_wrong_count_is_refused_on_the_positional_forms(call: Any) -> None:
    _, ev = _events()
    exc = _refused(lambda: call(ev))
    assert "correction 'jet_sf' takes inputs (syst, pt, eta); got 1 positional" in str(exc)


def test_keywords_are_the_template_form_bit_for_bit_and_node_for_node() -> None:
    session, ev = _events()
    keyword = gak.apply_correction(CSET, "jet_sf", eta=ev.eta, syst="up", pt=ev.pt)
    template = gak.apply_correction(CSET, "jet_sf", [ev.pt, ev.eta], args=["up", "$0", "$1"])
    assert keyword.node_id == template.node_id
    assert _values(session, keyword) == [[2.5, 4.25], [], [6.125]]


def test_a_correction_of_arrays_only_needs_no_template() -> None:
    session, ev = _events()
    assert _values(session, gak.apply_correction(CSET, "l1", [ev.pt])) == [
        [1.0, 2.0],
        [],
        [3.0],
    ]


def test_a_path_payload_takes_keywords(tmp_path: Path) -> None:
    session, ev = _events()
    path = tmp_path / "set.json"
    path.write_bytes(CSET)
    sf = gak.apply_correction(str(path), "l1", pt=ev.pt)
    assert _values(session, sf) == [[1.0, 2.0], [], [3.0]]


def test_keywords_refuse_the_positional_arguments() -> None:
    _, ev = _events()
    for extra in ({"args": ["$0"]}, {"evaluator": len}, {"inputs": [ev.pt]}):
        exc = _refused(lambda e=extra: gak.apply_correction(CSET, "l1", pt=ev.pt, **e))
        assert "as keywords or as inputs/args=, not both" in str(exc)


def test_keywords_of_constants_alone_are_refused() -> None:
    exc = _refused(lambda: gak.apply_correction(CSET, "l1", pt=1.0))
    assert "correction 'l1' needs at least one Array input" in str(exc)


def test_a_template_of_constants_alone_is_refused() -> None:
    _, ev = _events()
    exc = _refused(lambda: gak.apply_correction(CSET, "l1", [ev.pt], args=[1.0]))
    assert "correction 'l1' needs at least one Array input" in str(exc)


@pytest.mark.parametrize(
    ("syst", "shown"),
    [
        ("$0", "'$0'"),
        (None, "None"),
        ([1.0], "[1.0]"),
        (np.int64(1), "np.int64(1)"),
        (np.ones(3), "array([1., 1., 1.])"),
    ],
)
def test_a_keyword_that_is_no_input_value_is_refused_naming_it(syst: Any, shown: str) -> None:
    _, ev = _events()
    exc = _refused(lambda: gak.apply_correction(CSET, "jet_sf", syst=syst, pt=ev.pt, eta=ev.eta))
    assert "correction 'jet_sf' input 'syst' takes an Array" in str(exc)
    assert f"got {shown}" in str(exc)


def test_a_lone_array_where_a_list_belongs_names_the_iteration() -> None:
    _, ev = _events()
    with pytest.raises(TypeError, match="not iterable"):
        gak.apply_correction(CSET, "l1", ev.pt)


def test_no_inputs_at_all_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="needs inputs"):
        gak.apply_correction(CSET, "l1")


def test_a_varied_keyword_expands_per_universe() -> None:
    session, ev = _events()
    pt = graphed.vary(ev.pt, "jes", up=ev.pt * 2.0, down=ev.pt * 0.5)
    sf = gak.apply_correction(CSET, "l1", pt=pt)
    assert graphed.labels(sf) == ("nominal", "jes_up", "jes_down")
    assert _values(session, member_of(sf, "jes_up")) == [[2.0, 4.0], [], [6.0]]


def test_a_path_is_read_once_and_reread_when_the_file_changes(tmp_path: Path) -> None:
    path = tmp_path / "set.json"
    path.write_bytes(CSET)
    assert _payload_bytes(str(path)) is _payload_bytes(str(path))
    path.write_bytes(CSET + b" ")
    assert _payload_bytes(str(path)) == CSET + b" "
