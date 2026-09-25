"""m68 D7 — with no ``params["transport"]`` a bound endpoint's scheme picks the client module: ``http(s)`` →
``tritonclient.http``, ``grpc(s)`` → ``tritonclient.grpc``, given the bare ``host:port`` with ``ssl`` on the
``s`` schemes, and the requests are built from that module; ``params["transport"]`` still wins over any
scheme. The per-process resource cache keys on the endpoint and the params ``load`` reads, so nodes
differing only in evaluate-time params share one resource. Fake ``tritonclient`` modules sit in ``sys.modules``
(the m26 extra-test idiom)."""

from __future__ import annotations

import dataclasses
import json
import sys
import types
from typing import Any

import fake_triton_services as fake
import numpy as np
import pytest
from m68_services_fixtures import TRANSPORT, descriptor, expected, new_events, score, values_plan

from graphed.core import Plan, SequentialRunner
from graphed.debug import StageError
from graphed.preserve import CORRECTIONLIB_PLUGIN, TRITON_PLUGIN, record_external
from graphed.preserve.externals.correctionlib_external import load_correctionlib
from graphed.services import ServiceSpec, bind_services

PARAMS = {"model": "scorer", "input_name": "x", "output_name": "y"}


def _module(wire: str, log: list[Any]) -> types.ModuleType:
    """A ``tritonclient.<wire>`` stand-in logging connects as ``(url, ssl)`` and each request class built."""

    class InferenceServerClient(fake.FakeTritonClient):
        def __init__(self, url: str, *args: Any, ssl: bool = False, **kwargs: Any) -> None:
            log.append((url, ssl))
            super().__init__({"model": "scorer", "weights": {"w": 0.45, "b": -0.1}})

    class InferInput(fake.InferInput):
        def __init__(self, name: str, shape: list[int], datatype: str) -> None:
            log.append("InferInput")
            super().__init__(name, shape, datatype)

    class InferRequestedOutput(fake.InferRequestedOutput):
        def __init__(self, name: str) -> None:
            log.append("InferRequestedOutput")
            super().__init__(name)

    module = types.ModuleType(f"tritonclient.{wire}")
    module.InferenceServerClient = InferenceServerClient  # type: ignore[attr-defined]
    module.InferInput = InferInput  # type: ignore[attr-defined]
    module.InferRequestedOutput = InferRequestedOutput  # type: ignore[attr-defined]
    return module


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    logs: dict[str, list[Any]] = {"http": [], "grpc": []}
    package = types.ModuleType("tritonclient")
    monkeypatch.setitem(sys.modules, "tritonclient", package)
    for wire, log in logs.items():
        module = _module(wire, log)
        setattr(package, wire, module)
        monkeypatch.setitem(sys.modules, f"tritonclient.{wire}", module)
    return logs


def _served(tag: str) -> Plan[Any]:
    ev = new_events()
    ev.session.declare_service(ServiceSpec("scorer-svc", "triton"))
    params = {**PARAMS, "service": "scorer-svc"}
    return values_plan(record_external(ev.session, TRITON_PLUGIN, descriptor(tag), [ev.x], params=params))


@pytest.mark.parametrize(
    ("endpoint", "wire", "ssl"),
    [
        ("http://tr-http:8000", "http", False),
        ("https://tr-https:8443", "http", True),
        ("grpc://tr-grpc:8001", "grpc", False),
        ("grpcs://triton.fnal.gov:443", "grpc", True),
    ],
)
def test_the_scheme_picks_the_module_host_port_and_ssl(
    seen: dict[str, list[Any]], endpoint: str, wire: str, ssl: bool
) -> None:
    plan = bind_services(_served(f"scheme-{endpoint}"), {"scorer-svc": endpoint})
    (row,) = SequentialRunner().run(plan).value
    assert np.allclose(row, expected(0.45, -0.1), rtol=1e-6)
    other = "grpc" if wire == "http" else "http"
    assert seen[other] == []
    assert [e for e in seen[wire] if isinstance(e, tuple)] == [(endpoint.split("://")[1], ssl)]
    assert {"InferInput", "InferRequestedOutput"} <= set(seen[wire])


def test_a_tcp_endpoint_is_refused_naming_the_wires(seen: dict[str, list[Any]]) -> None:
    plan = bind_services(_served("scheme-tcp"), {"scorer-svc": "tcp://tr-tcp:8001"})
    with pytest.raises(StageError, match=r"(?s)(?=.*http)(?=.*grpc)") as err:
        SequentialRunner().run(plan)
    assert err.value.cause_type == "PreserveError"
    assert seen == {"http": [], "grpc": []}


def test_a_literal_url_without_a_scheme_stays_http(seen: dict[str, list[Any]]) -> None:
    ev = new_events()
    params = {**PARAMS, "url": "tr-literal:8000"}
    plan = values_plan(
        record_external(ev.session, TRITON_PLUGIN, descriptor("literal"), [ev.x], params=params)
    )
    (row,) = SequentialRunner().run(plan).value
    assert np.allclose(row, expected(0.45, -0.1), rtol=1e-6)
    assert seen["grpc"] == []
    assert [e for e in seen["http"] if isinstance(e, tuple)] == [("tr-literal:8000", False)]


def test_the_transport_param_wins_over_any_scheme(seen: dict[str, list[Any]]) -> None:
    literal, served = "triton://tr-literal-t:8000", "grpcs://tr-served-t:443"
    fake.serve(literal, descriptor("transport-literal", 0.45, -0.1))
    fake.serve(served, descriptor("transport-served", 0.8, 0.2))
    ev = new_events()
    ev.session.declare_service(ServiceSpec("scorer-svc", "triton"))
    plan = values_plan(
        score(ev, descriptor("transport-literal", 0.45, -0.1), url=literal),
        score(ev, descriptor("transport-served", 0.8, 0.2), service="scorer-svc"),
    )
    row_literal, row_served = SequentialRunner().run(bind_services(plan, {"scorer-svc": served})).value
    assert np.allclose(row_literal, expected(0.45, -0.1), rtol=1e-6)
    assert np.allclose(row_served, expected(0.8, 0.2), rtol=1e-6)
    assert fake.SERVERS[literal].infer_calls > 0 and fake.SERVERS[served].infer_calls > 0
    assert seen == {"http": [], "grpc": []}


def test_the_connection_key_is_the_endpoint_and_the_params_load_reads(seen: dict[str, list[Any]]) -> None:
    fake.serve("http://tr-key-a:8000", descriptor("key-load-params"))
    ev = new_events()
    for name in ("svc-a", "svc-b"):
        ev.session.declare_service(ServiceSpec(name, "triton"))
    nodes = [
        record_external(
            ev.session, TRITON_PLUGIN, descriptor("key-load-params"), [ev.x], params={**PARAMS, **extra}
        )
        for extra in (
            {"service": "svc-a"},
            {"service": "svc-a", "output_name": "z"},
            {"service": "svc-b"},
            {"service": "svc-a", "transport": TRANSPORT},
        )
    ]
    endpoints = {"svc-a": "http://tr-key-a:8000", "svc-b": "http://tr-key-b:8000"}
    for row in SequentialRunner().run(bind_services(values_plan(*nodes), endpoints)).value:
        assert np.allclose(row, expected(0.45, -0.1), rtol=1e-6)
    connects = sorted(e for e in seen["http"] if isinstance(e, tuple))
    assert connects == [("tr-key-a:8000", False), ("tr-key-b:8000", False)]
    assert fake.SERVERS["http://tr-key-a:8000"].infer_calls > 0


def test_correctionlib_universes_off_one_payload_load_once() -> None:
    loads: list[bytes] = []

    def counting(payload: bytes, params: Any) -> Any:
        loads.append(payload)
        return load_correctionlib(payload, params)

    plugin = dataclasses.replace(CORRECTIONLIB_PLUGIN, load=counting)
    cset = {
        "schema_version": 2,
        "description": "m68 load-once",
        "corrections": [
            {
                "name": "sf",
                "version": 1,
                "inputs": [{"name": "systematic", "type": "string"}, {"name": "x", "type": "real"}],
                "output": {"name": "sf", "type": "real"},
                "data": {
                    "nodetype": "category",
                    "input": "systematic",
                    "content": [{"key": "up", "value": 1.1}, {"key": "down", "value": 0.9}],
                },
            }
        ],
    }
    ev = new_events()
    nodes = [
        record_external(
            ev.session, plugin, json.dumps(cset).encode(), [ev.x], params={"name": "sf", "systematic": u}
        )
        for u in ("up", "down")
    ]
    up, down = SequentialRunner().run(values_plan(*nodes)).value
    assert np.allclose(up, 1.1) and np.allclose(down, 0.9) and len(up) == len(down) == 8
    assert len(loads) == 1


def test_equal_node_params_in_any_key_order_share_one_connection(seen: dict[str, list[Any]]) -> None:
    params = {**PARAMS, "service": "scorer-svc"}
    for order in (params, dict(reversed(params.items()))):  # two sessions, so two evaluators
        ev = new_events()
        ev.session.declare_service(ServiceSpec("scorer-svc", "triton"))
        node = record_external(ev.session, TRITON_PLUGIN, descriptor("key-order"), [ev.x], params=order)
        SequentialRunner().run(bind_services(values_plan(node), {"scorer-svc": "http://tr-order:8000"}))
    assert [e for e in seen["http"] if isinstance(e, tuple)] == [("tr-order:8000", False)]
