"""m71b B-C1/B-C2/B-C5: `record_external(output_type=)` records the declared type (graphed#57)."""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pytest
from awkward.types import ListType, NumpyType, RecordType
from m71b_fixtures import (
    JSEL,
    LUMI,
    PHO,
    PHOTON,
    describe,
    eager,
    here,
    jet_select,
    lumi,
    numpy_recorded,
    params_of,
    photons,
    recorded,
    refusal,
    same,
)

from graphed import GraphedTypeError
from graphed.awkward import gak
from graphed.preserve.externals import record_external


def test_a_declared_lumi_mask_is_a_bool_mask() -> None:
    s, ev = recorded()
    e = eager()
    m = record_external(s, LUMI, b"g", [ev.run], output_type="bool")

    assert describe(s, m) == "## * bool"
    assert describe(s, ~m) == "## * bool"
    assert describe(s, ev.x[m]) == "## * float32"
    assert same(s.materialize(m), lumi(None, {}, [e.run]))
    assert same(s.materialize(ev.x[m]), e.x[lumi(None, {}, [e.run])])


def test_a_declared_jagged_selection_indexes_its_jets() -> None:
    s, ev = recorded()
    e = eager()
    jm = record_external(s, JSEL, b"j", [ev.Jet.pt], output_type="var * bool")

    assert describe(s, jm) == "## * var * bool"
    assert describe(s, ev.Jet.pt[jm]) == "## * var * float64"
    assert same(s.materialize(ev.Jet.pt[jm]), e.Jet.pt[jet_select(None, {}, [e.Jet.pt])])


def test_a_scalar_input_records_the_declared_scalar() -> None:
    s, ev = recorded()
    assert describe(s, record_external(s, LUMI, b"g", [gak.sum(ev.x)], output_type="bool")) == "bool"


def test_a_declared_named_record_resolves_its_behavior() -> None:
    s, ev = recorded()
    e = eager()
    g = record_external(s, PHO, b"p", [ev.Jet.pt], output_type=PHOTON)
    expected = ak.Array(photons(None, {}, [e.Jet.pt]), behavior=e.behavior)

    assert describe(s, g) == f"## * {PHOTON}"
    assert describe(s, g.pt2) == "## * var * float32"
    assert same(s.materialize(g.pt2), expected.pt2)


def test_the_declared_type_is_identity_and_the_undeclared_form_is_unchanged() -> None:
    s, ev = recorded()
    declared = record_external(s, LUMI, b"g", [ev.run], output_type="bool")
    undeclared = record_external(s, LUMI, b"g", [ev.run])

    assert declared.node_id != undeclared.node_id
    assert describe(s, undeclared) == "## * uint32"
    assert "output_type" not in params_of(s, undeclared)
    as_int8 = record_external(s, LUMI, b"g", [ev.run], output_type="int8")
    assert as_int8.node_id not in {declared.node_id, undeclared.node_id}


def test_spellings_of_one_declared_type_are_one_node() -> None:
    s, ev = recorded()
    flat = {
        record_external(s, LUMI, b"g", [ev.run], output_type=spec).node_id
        for spec in ("bool", bool, np.bool_)
    }
    assert len(flat) == 1
    assert params_of(s, record_external(s, LUMI, b"g", [ev.run], output_type=bool))["output_type"] == "bool"

    jagged = {
        record_external(s, JSEL, b"j", [ev.Jet.pt], output_type=spec).node_id
        for spec in ("var*bool", "var * bool", ListType(NumpyType("bool")))
    }
    assert len(jagged) == 1

    f32 = NumpyType("float32")
    photon_type = ListType(RecordType([f32, f32], ["pt", "eta"], parameters={"__record__": "Photon"}))
    named = {
        record_external(s, PHO, b"p", [ev.Jet.pt], output_type=spec).node_id for spec in (PHOTON, photon_type)
    }
    assert len(named) == 1


@pytest.mark.parametrize("spec", ["nope", 3])
def test_an_unbuildable_declaration_is_refused_at_the_call(spec: Any) -> None:
    s, ev = recorded()
    line = here()
    exc = refusal(lambda: record_external(s, LUMI, b"g", [ev.run], output_type=spec))
    assert (exc.provenance.filename, exc.provenance.lineno) == (__file__, line)
    assert repr(spec) in exc.detail


def test_a_numpy_session_still_has_no_payload_descriptor() -> None:
    sn, x = numpy_recorded()
    with pytest.raises(GraphedTypeError, match="no payload descriptor"):
        record_external(sn, LUMI, b"g", [x], output_type="bool")


def _program() -> bytes:
    s, ev = recorded()
    outs = [
        record_external(s, LUMI, b"g", [ev.run], output_type="bool"),
        record_external(s, LUMI, b"g", [ev.run]),
        record_external(s, LUMI, b"g", [ev.run], output_type="int8"),
        record_external(s, JSEL, b"j", [ev.Jet.pt], output_type="var * bool"),
        record_external(s, PHO, b"p", [ev.Jet.pt], output_type=PHOTON).pt2,
    ]
    return bytes(s.serialized_ir(*outs, optimize=False))


def test_declared_recordings_serialize_byte_identically_across_sessions() -> None:
    first, second = _program(), _program()
    assert first == second
    assert PHOTON.encode() in first
