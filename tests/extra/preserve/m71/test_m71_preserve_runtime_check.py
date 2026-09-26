"""A preserve External's declaration is checked at run time: ``output_type=``, else the plugin's
static leaf ``output_dtype``; a disagreeing value is refused at the declaring line."""

from __future__ import annotations

import hashlib
import sys
from typing import Any

import awkward as ak
import m71_triton_fixtures as fx
import numpy as np
import pytest

import graphed
from graphed import GraphedTypeError, Session
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward, gak
from graphed.core.execution import SequentialRunner
from graphed.debug import StageError
from graphed.preserve import TRITON_PLUGIN
from graphed.preserve.externals import ExternalPlugin, _base, record_external
from graphed.services import ServiceSpec, bind_services


def here() -> int:
    return sys._getframe(1).f_lineno


def _plugin(dtype: str, output_dtype: object) -> ExternalPlugin:
    return ExternalPlugin(
        kind=f"m71-check-{dtype}-{output_dtype}",
        content_hash=lambda b: hashlib.sha256(b).hexdigest(),
        evaluate=lambda resource, params, inputs: ak.values_astype(inputs[0] > 1, dtype),
        samples=lambda: [],
        output_dtype=output_dtype,
    )


def _x() -> tuple[Session, Any]:
    s = Session(AwkwardBackend())
    return s, from_awkward(s, "e", ak.zip({"x": np.array([0.5, 2.0], dtype=np.float32)}, depth_limit=1)).x


def test_a_served_mask_declared_bool_is_refused_at_its_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(_base._REGISTRY, TRITON_PLUGIN.kind, TRITON_PLUGIN)
    s = Session(AwkwardBackend())
    chunks = fx.Chunks()
    form = AwkwardForm(ak.Array(chunks.data.layout.to_typetracer(forget_length=True)))
    ev = s.source("events", form=form, data=chunks)
    s.declare_service(ServiceSpec("scorer", "http"))
    params = {"model": "scorer", "transport": "m71_triton_fixtures:transport", "service": "scorer"}
    payload = b'{"model": "scorer"}'
    mask, line = record_external(s, TRITON_PLUGIN, payload, [ev.x], output_type="bool", params=params), here()
    plan = graphed.aggregate_plan(gak.sum(ev.x[mask]), reduce=fx.total, combine=fx.add, empty=fx.zero)
    with pytest.raises(StageError) as info:
        SequentialRunner().run(bind_services(plan, {"scorer": fx.ENDPOINT}))
    assert info.value.user_frame.lineno == line
    assert "('triton_model') declares output_type 'bool'; its value is 'float64'" in info.value.cause_message


def test_a_plugin_breaking_its_output_dtype_is_refused() -> None:
    s, x = _x()
    out, line = record_external(s, _plugin("float32", "float64"), b"m", [x]), here()
    with pytest.raises(GraphedTypeError) as info:
        s.materialize(out)
    assert info.value.provenance.lineno == line
    assert "declares output_dtype 'float64'; its value is 'float32'" in info.value.detail


def test_output_type_wins_over_output_dtype() -> None:
    s, x = _x()
    out = record_external(s, _plugin("bool", "float64"), b"m", [x], output_type=bool)
    assert ak.to_list(s.materialize(out)) == [False, True]


def test_an_honest_plugin_and_an_undeclared_one_run() -> None:
    s, x = _x()
    assert ak.to_list(s.materialize(record_external(s, _plugin("float64", "float64"), b"m", [x]))) == [
        0.0,
        1.0,
    ]
    assert ak.to_list(s.materialize(record_external(s, _plugin("int8", None), b"m", [x]))) == [0, 1]
