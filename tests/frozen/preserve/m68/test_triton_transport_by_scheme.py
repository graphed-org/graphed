"""m68 D7 — with no ``params["transport"]`` a bound endpoint's scheme picks the client module: ``http(s)`` →
``tritonclient.http``, ``grpc(s)`` → ``tritonclient.grpc``, given the bare ``host:port`` with ``ssl`` on the
``s`` schemes, and the requests are built from that module; ``params["transport"]`` still wins over any
scheme. Fake ``tritonclient`` modules sit in ``sys.modules`` (the m26 extra-test idiom)."""

from __future__ import annotations

import sys
import types
from typing import Any

import fake_triton_services as fake
import numpy as np
import pytest
from m68_services_fixtures import descriptor, expected, new_events, score, values_plan

from graphed.core import Plan, SequentialRunner
from graphed.debug import StageError
from graphed.preserve import TRITON_PLUGIN, record_external
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


def test_node_params_are_part_of_the_connection_key(seen: dict[str, list[Any]]) -> None:
    ev = new_events()
    ev.session.declare_service(ServiceSpec("scorer-svc", "triton"))
    payload = descriptor("key-params")
    nodes = [
        record_external(ev.session, TRITON_PLUGIN, payload, [ev.x], params={**PARAMS, **extra})
        for extra in ({"service": "scorer-svc"}, {"service": "scorer-svc", "output_name": "z"})
    ]
    plan = bind_services(values_plan(*nodes), {"scorer-svc": "http://tr-key:8000"})
    for row in SequentialRunner().run(plan).value:
        assert np.allclose(row, expected(0.45, -0.1), rtol=1e-6)
    assert [e for e in seen["http"] if isinstance(e, tuple)] == [("tr-key:8000", False)] * 2


def test_equal_node_params_in_any_key_order_share_one_connection(seen: dict[str, list[Any]]) -> None:
    params = {**PARAMS, "service": "scorer-svc"}
    for order in (params, dict(reversed(params.items()))):  # two sessions, so two evaluators
        ev = new_events()
        ev.session.declare_service(ServiceSpec("scorer-svc", "triton"))
        node = record_external(ev.session, TRITON_PLUGIN, descriptor("key-order"), [ev.x], params=order)
        SequentialRunner().run(bind_services(values_plan(node), {"scorer-svc": "http://tr-order:8000"}))
    assert [e for e in seen["http"] if isinstance(e, tuple)] == [("tr-order:8000", False)]
