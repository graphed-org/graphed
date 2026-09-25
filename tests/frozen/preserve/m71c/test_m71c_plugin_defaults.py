"""m71c C-C3/C-C5: built-in plugins and the gak correction template record their float64 value type;
a plugin's `output_dtype` default is validated at the calling line, and any numerical dtype works."""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pytest
from m71c_fixtures import (
    CSET,
    EVENTS,
    NUMERIC,
    SF,
    _evaluator,
    describe,
    here,
    numpy_recorded,
    plugin,
    recorded,
    refusal,
    same,
)

from graphed import GraphedTypeError
from graphed.awkward import gak
from graphed.preserve import (
    CORRECTIONLIB_PLUGIN,
    HISTOGRAM_PLUGIN,
    JAX_PLUGIN,
    ONNX_PLUGIN,
    PYTORCH_PLUGIN,
    TENSORFLOW_PLUGIN,
    TRITON_PLUGIN,
    XGBOOST_PLUGIN,
)
from graphed.preserve.externals import record_external

FLOAT64_BUILTINS = (
    CORRECTIONLIB_PLUGIN,
    ONNX_PLUGIN,
    TENSORFLOW_PLUGIN,
    PYTORCH_PLUGIN,
    XGBOOST_PLUGIN,
    JAX_PLUGIN,
    TRITON_PLUGIN,
)


def test_the_float64_builtins_declare_float64_and_the_histogram_plugin_nothing() -> None:
    assert [p.output_dtype for p in FLOAT64_BUILTINS] == ["float64"] * 7
    assert HISTOGRAM_PLUGIN.output_dtype is None


def test_a_correctionlib_external_records_float64_over_a_float32_input() -> None:
    s, ev = recorded()
    sf = record_external(s, CORRECTIONLIB_PLUGIN, CSET, [ev.x], params=SF)

    assert describe(s, sf) == "## * float64"
    assert str(ak.type(s.materialize(sf))) == "4 * float64"
    assert (
        describe(s, record_external(s, CORRECTIONLIB_PLUGIN, CSET, [gak.sum(ev.x)], params=SF)) == "float64"
    )


def test_the_correction_template_records_float64() -> None:
    s, ev = recorded()
    tc = gak.apply_correction(CSET, "sf", [ev.x], _evaluator, args=["nominal", "$0"])

    assert describe(s, tc) == "## * float64"
    total = gak.sum(ev.x)
    assert (
        describe(s, gak.apply_correction(CSET, "sf", [total], _evaluator, args=["nominal", "$0"]))
        == "float64"
    )


def test_an_explicit_output_type_wins_over_the_plugin_default() -> None:
    s, ev = recorded()
    default = record_external(s, CORRECTIONLIB_PLUGIN, CSET, [ev.x], params=SF)
    explicit = record_external(s, CORRECTIONLIB_PLUGIN, CSET, [ev.x], params=SF, output_type="float32")

    assert describe(s, explicit) == "## * float32"
    assert explicit.node_id != default.node_id
    assert describe(s, default) == "## * float64"


@pytest.mark.parametrize(
    ("kind", "spec"), [("m71c_float", float), ("m71c_np_float64", np.float64), ("m71c_f8", "f8")]
)
def test_every_float64_spelling_of_a_plugin_default_records_float64(kind: str, spec: Any) -> None:
    s, ev = recorded()
    custom = plugin(kind, "float64", output_dtype=spec)
    out = record_external(s, custom, b"c", [ev.x])

    assert describe(s, out) == "## * float64"
    assert same(s.materialize(out), ak.values_astype(EVENTS.x, "float64"))


def test_a_numpy_session_still_has_no_payload_descriptor() -> None:
    sn, x = numpy_recorded()
    with pytest.raises(GraphedTypeError, match="no payload descriptor"):
        record_external(sn, CORRECTIONLIB_PLUGIN, CSET, [x], params=SF)


def test_an_unknown_plugin_default_is_refused_at_the_call() -> None:
    s, ev = recorded()
    bad = plugin("m71c_nope", "float64", output_dtype="nope")
    line = here()
    exc = refusal(lambda: record_external(s, bad, b"c", [ev.run]))
    assert (exc.provenance.filename, exc.provenance.lineno) == (__file__, line)
    assert "nope" in exc.detail


def test_a_non_leaf_plugin_default_is_refused_at_the_call() -> None:
    s, ev = recorded()
    bad = plugin("m71c_list", "float32", output_dtype="var * float32")
    line = here()
    exc = refusal(lambda: record_external(s, bad, b"c", [ev.run]))
    assert (exc.provenance.filename, exc.provenance.lineno) == (__file__, line)
    assert "output_dtype" in exc.detail


def test_a_float16_plugin_default_records_float16() -> None:
    s, ev = recorded()
    half = plugin("m71c_half", "float16", output_dtype=np.float16)
    out = record_external(s, half, b"c", [ev.run])

    assert describe(s, out) == "## * float16"
    assert same(s.materialize(out), ak.values_astype(EVENTS.run, "float16"))


@pytest.mark.parametrize("name", NUMERIC)
def test_every_numerical_plugin_default_records_its_dtype(name: str) -> None:
    s, ev = recorded()
    custom = plugin(f"m71c_num_{name}", name, output_dtype=np.dtype(name))
    out = record_external(s, custom, b"c", [ev.run])

    assert describe(s, out) == f"## * {name}"
    assert same(s.materialize(out), ak.values_astype(EVENTS.run, name))
