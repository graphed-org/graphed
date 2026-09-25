"""m71a A-C5/A-C8 (awkward) and A-C4's awkward-object legs on numpy: a type the backend cannot build
is refused at the calling line, naming the spelling it was given."""

from __future__ import annotations

from typing import Any

import awkward as ak
import pytest
from awkward.types import ListType, NumpyType
from m71a_awkward_fixtures import (
    describe,
    eager,
    f,
    here,
    numpy_recorded,
    recorded,
    refusal,
    same,
)

from graphed.awkward import gak


@pytest.mark.parametrize("spec", ["nope", "var * nope", object, "V8", ">f4", 3, "unknown"])
def test_a_type_awkward_cannot_build_is_refused_at_the_call(spec: Any) -> None:
    _s, ev = recorded()
    line = here()
    exc = refusal(lambda: ev.x.map(f, name="bad", output_type=spec))
    assert (exc.provenance.filename, exc.provenance.lineno) == (__file__, line)
    assert repr(spec) in exc.detail


def test_a_scalar_input_refuses_a_record_element() -> None:
    _s, ev = recorded()
    total = gak.sum(ev.x)
    line = here()
    exc = refusal(lambda: total.map(f, name="bad", output_type="{a: float32}"))
    assert (exc.provenance.filename, exc.provenance.lineno) == (__file__, line)
    assert repr("{a: float32}") in exc.detail


def test_numpy_reads_an_awkward_primitive_as_its_dtype() -> None:
    sn, x = numpy_recorded()
    by_name = x.map(lambda a: a, name="o", output_type="float32")

    assert describe(sn, by_name) == "vector[float32]"
    assert x.map(lambda a: a, name="o", output_type=NumpyType("float32")).node_id == by_name.node_id
    as_form = ak.forms.from_type(NumpyType("float32"))
    assert x.map(lambda a: a, name="o", output_type=as_form).node_id == by_name.node_id


def test_numpy_refuses_an_awkward_list_type_as_it_refuses_the_string() -> None:
    _sn, x = numpy_recorded()
    line = here()
    as_object = refusal(lambda: x.map(f, name="bad", output_type=ListType(NumpyType("float32"))))
    assert (as_object.provenance.filename, as_object.provenance.lineno) == (__file__, line)
    as_string = refusal(lambda: x.map(f, name="bad", output_type="var * float32"))
    assert as_object.detail == as_string.detail


def _nested_float16_parses() -> bool:
    try:
        ak.types.from_datashape("var * float16", highlevel=False)
    except Exception:
        return False
    return True


def test_a_float16_record_field_records_or_is_refused_by_the_installed_parser() -> None:
    s, ev = recorded()
    spec = [("pt", "f2")]

    def fn(a: Any) -> Any:
        return ak.zip({"pt": ak.values_astype(a, "f2")})

    if _nested_float16_parses():
        out = ev.x.map(fn, name="h16r", output_type=spec)
        assert describe(s, out) == "## * {pt: float16}"
        assert same(s.materialize(out), fn(eager().x))
    else:
        line = here()
        exc = refusal(lambda: ev.x.map(fn, name="h16r", output_type=spec))
        assert (exc.provenance.filename, exc.provenance.lineno) == (__file__, line)
        assert repr(spec) in exc.detail


def test_a_jagged_float16_records_or_is_refused_by_the_installed_parser() -> None:
    s, ev = recorded()
    spec = "var * float16"

    def fn(a: Any) -> Any:
        return ak.values_astype(a, "f2")

    if _nested_float16_parses():
        out = ev.Jet.pt.map(fn, name="h16j", output_type=spec)
        assert describe(s, out) == "## * var * float16"
        assert same(s.materialize(out), fn(eager().Jet.pt))
    else:
        line = here()
        exc = refusal(lambda: ev.Jet.pt.map(fn, name="h16j", output_type=spec))
        assert (exc.provenance.filename, exc.provenance.lineno) == (__file__, line)
        assert repr(spec) in exc.detail
