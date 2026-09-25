"""m71a A-C5 (numpy): a declaration the backend cannot represent is refused at the calling line."""

from __future__ import annotations

from typing import Any

import pytest
from backends import ListBackend, from_list
from m71a_numpy_fixtures import f, here, recorded, refusal

from graphed import Session
from graphed.numpy import NumpyBackend


@pytest.mark.parametrize("spec", ["var * float32", "{pt: float32}", "nope", 3])
def test_a_type_numpy_cannot_represent_is_refused_at_the_call(spec: Any) -> None:
    _sn, x, _y, _m2 = recorded()
    line = here()
    exc = refusal(lambda: x.map(f, name="bad", output_type=spec))
    assert (exc.provenance.filename, exc.provenance.lineno) == (__file__, line)
    assert repr(spec) in exc.detail


def test_a_nested_structured_dtype_is_refused() -> None:
    _sn, x, _y, _m2 = recorded()
    line = here()
    exc = refusal(lambda: x.map(f, name="bad", output_type=[("r", [("x", "i4")])]))
    assert (exc.provenance.filename, exc.provenance.lineno) == (__file__, line)
    assert "plain dtypes" in exc.detail


@pytest.mark.parametrize("spec", [[("pt", "f4")], ("f4", (3,))])
def test_a_scalar_input_refuses_fields_and_subarrays(spec: Any) -> None:
    _sn, x, _y, _m2 = recorded()
    total = x.reduce("sum")
    line = here()
    exc = refusal(lambda: total.map(f, name="bad", output_type=spec))
    assert (exc.provenance.filename, exc.provenance.lineno) == (__file__, line)
    assert "scalar input" in exc.detail


def test_a_backend_without_the_hook_refuses_any_declaration() -> None:
    session = Session(ListBackend())
    a = from_list(session, "a", [0.5, 1.5])
    assert not hasattr(ListBackend(), "canonical_output_type")
    line = here()
    exc = refusal(lambda: a.map(f, name="bad", output_type="bool"))
    assert (exc.provenance.filename, exc.provenance.lineno) == (__file__, line)
    assert "output_type" in exc.detail


def test_output_type_and_an_explicit_form_are_exclusive() -> None:
    sn, x, _y, _m2 = recorded()
    params = {"fn": "n"}
    d, fx = NumpyBackend().external_payload("map", params), sn.form(x)

    def call() -> object:
        return sn.record_external("map", f, [x], params, descriptor=d, form=fx, output_type="bool")

    exc = refusal(call)
    assert (exc.provenance.filename, exc.provenance.lineno) == (__file__, call.__code__.co_firstlineno + 1)
    assert "exclusive" in exc.detail
