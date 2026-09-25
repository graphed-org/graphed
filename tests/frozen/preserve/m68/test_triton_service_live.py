"""m68 D7 live — the shipped Triton plugin reaches a REAL server only through ``bind_services``, one leg per
wire, with no ``transport`` param: the endpoint's scheme picks ``tritonclient.http`` or ``tritonclient.grpc``.
The CI ``triton`` job serves ``tests/samples/triton_models/scorer``; each leg is gated on its variable like
``preserve/m9/test_triton_server.py``. The gRPC port answers only gRPC, so that leg witnesses the gRPC client."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from m68_services_fixtures import expected, new_events, values_plan

from graphed.core import SequentialRunner
from graphed.preserve import TRITON_PLUGIN, record_external
from graphed.services import ServiceSpec, bind_services

SERVERS = {"http": os.environ.get("GRAPHED_TRITON_HTTP"), "grpc": os.environ.get("GRAPHED_TRITON_GRPC")}
CHECKS = {"http": "http:/v2/health/ready", "grpc": "grpc:"}
# the served model's weights — they match tests/samples/triton_models/scorer/1/model.py
PAYLOAD = json.dumps({"model": "scorer", "version": "1", "weights": {"w": 0.45, "b": -0.1}}).encode()


def _leg(wire: str) -> object:
    reason = f"no real Triton server (set GRAPHED_TRITON_{wire.upper()}=host:port)"
    return pytest.param(wire, marks=pytest.mark.skipif(not SERVERS[wire], reason=reason))


@pytest.mark.parametrize("wire", [_leg("http"), _leg("grpc")])
def test_live_triton_through_a_bound_service(wire: str) -> None:
    pytest.importorskip(f"tritonclient.{wire}")
    ev = new_events()
    ev.session.declare_service(ServiceSpec("triton-live", "triton", check=CHECKS[wire]))
    params = {"service": "triton-live", "model": "scorer", "input_name": "x", "output_name": "y"}
    plan = values_plan(record_external(ev.session, TRITON_PLUGIN, PAYLOAD, [ev.x], params=params))
    bound = bind_services(plan, {"triton-live": f"{wire}://{SERVERS[wire]}"})
    assert bound.process.ir == plan.process.ir
    (row,) = SequentialRunner().run(bound).value
    assert np.allclose(row, expected(0.45, -0.1), atol=1e-6)  # FP32 over the wire
