"""m71c C-C8: an input from another Session is refused at the calling line on every External route
(the plugin route and both gak templates), before any form is computed."""

from __future__ import annotations

from m71c_fixtures import CSET, MODEL, SF, _evaluator, here, recorded, refusal

from graphed.awkward import gak
from graphed.preserve.externals import CORRECTIONLIB_PLUGIN, record_external


def test_a_plugin_external_refuses_a_foreign_input() -> None:
    s, _ev = recorded()
    _s2, ev2 = recorded()
    line = here()
    exc = refusal(lambda: record_external(s, CORRECTIONLIB_PLUGIN, CSET, [ev2.x], params=SF))
    assert (exc.provenance.filename, exc.provenance.lineno) == (__file__, line)
    assert "different Session" in str(exc)


def test_the_correction_template_refuses_a_foreign_input() -> None:
    _s, ev = recorded()
    _s2, ev2 = recorded()
    line = here()
    exc = refusal(lambda: gak.apply_correction(CSET, "sf", [ev.x, ev2.x], _evaluator, args=["nominal", "$1"]))
    assert (exc.provenance.filename, exc.provenance.lineno) == (__file__, line)
    assert "different Session" in str(exc)


def test_the_onnx_template_refuses_a_foreign_input() -> None:
    _s, ev = recorded()
    _s2, ev2 = recorded()
    line = here()
    exc = refusal(lambda: gak.onnx_inference(MODEL, [ev.x, ev2.x], _evaluator, args=[["$1"]]))
    assert (exc.provenance.filename, exc.provenance.lineno) == (__file__, line)
    assert "different Session" in str(exc)
