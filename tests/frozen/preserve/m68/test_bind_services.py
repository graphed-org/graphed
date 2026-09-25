"""m68 D2/§3.2 — ``bind_services(plan, endpoints)`` returns a plan whose External evaluators (and a
``reduce`` with the hook) carry the endpoints; the original plan is untouched and still unbound. An
endpoint is ``scheme://host:port`` (``split_endpoint``), checked for every endpoint before binding."""

from __future__ import annotations

import dataclasses
from typing import Any

import fake_triton_services as fake
import numpy as np
import pytest
from m68_services_fixtures import _rows, descriptor, expected, mark, new_events, score, values_plan

from graphed.core import Plan, SequentialRunner
from graphed.services import ServiceSpec, UnboundService, bind_services, split_endpoint


def _endpoints(plan: Plan[Any]) -> list[str | None]:
    return [getattr(fn, "endpoint", None) for _key, fn in plan.process.externals]


def _served(tag: str, **plan_kw: Any) -> Plan[Any]:
    ev = new_events()
    ev.session.declare_service(ServiceSpec("scorer-svc", "triton"))
    return values_plan(score(ev, descriptor(tag), service="scorer-svc"), **plan_kw)


def test_a_bound_plan_carries_the_endpoint_and_the_original_stays_unbound() -> None:
    url = "grpc://bind-a:8000"
    server = fake.serve(url, descriptor("bind-a"))
    plan = _served("bind-a")
    bound = bind_services(plan, {"scorer-svc": url})
    assert _endpoints(bound) == [url]
    assert _endpoints(plan) == [None]
    assert bound.services == plan.services != ()
    assert dataclasses.replace(bound, process=plan.process) == plan
    (row,) = SequentialRunner().run(bound).value
    assert np.allclose(row, expected(0.45, -0.1), rtol=1e-6)
    assert server.infer_calls > 0
    with pytest.raises(UnboundService, match="scorer-svc"):
        SequentialRunner().run(plan)


def test_any_external_kind_is_bound() -> None:
    ev = new_events()
    ev.session.declare_service(ServiceSpec("gen-svc", "http"))
    plan = values_plan(mark(ev, "bind-generic", "gen-svc"))
    assert _endpoints(bind_services(plan, {"gen-svc": "http://gen:8000"})) == ["http://gen:8000"]
    assert _endpoints(plan) == [None]


def test_a_missing_endpoint_raises_naming_the_service() -> None:
    with pytest.raises(UnboundService, match="scorer-svc"):
        bind_services(_served("bind-missing"), {"other-svc": "grpc://other:8000"})
    with pytest.raises(UnboundService, match="scorer-svc"):
        bind_services(_served("r4-empty"), {})


class _BindableReduce:
    def __init__(self, endpoints: dict[str, str] | None = None) -> None:
        self.endpoints = endpoints

    def __call__(self, values: list[Any]) -> list[list[float]]:
        return _rows(values)

    def bind_services(self, endpoints: dict[str, str]) -> _BindableReduce:
        return _BindableReduce(dict(endpoints))


def test_a_reduce_with_the_hook_is_bound_too() -> None:
    endpoints = {"scorer-svc": "http://bind-reduce:8000"}
    plan = _served("bind-reduce", reduce=_BindableReduce())
    bound = bind_services(plan, endpoints)
    assert bound.process.reduce.endpoints == endpoints
    assert plan.process.reduce.endpoints is None


def test_a_reduce_only_service_is_bound() -> None:  # D8: histserv lives on the reduce, no External names it
    ev = new_events()
    ev.session.declare_service(ServiceSpec("hist-svc", "histserv"))
    plan = values_plan(ev.x, reduce=_BindableReduce(), services=("hist-svc",))
    endpoints = {"hist-svc": "tcp://hists:9000"}
    assert bind_services(plan, endpoints).process.reduce.endpoints == endpoints


def test_a_plan_without_services_binds_to_an_equal_process() -> None:
    ev = new_events()
    plan = values_plan(score(ev, descriptor("bind-none"), url="triton://bind-none:8000"))
    bound = bind_services(plan, {"scorer-svc": "http://elsewhere:8000"})
    assert bound.process == plan.process
    assert _endpoints(bound) == [None]
    plain: Plan[Any] = Plan(process=lambda part, res: [], combine=lambda a, b: a, empty=list)
    assert bind_services(plain, {"scorer-svc": "http://elsewhere:8000"}) is plain


@pytest.mark.parametrize(
    ("endpoint", "split"),
    [
        ("tcp://svc:9000", ("tcp", "svc:9000")),
        ("http://svc:8000", ("http", "svc:8000")),
        ("https://svc:8443", ("https", "svc:8443")),
        ("grpc://svc:8001", ("grpc", "svc:8001")),
        ("grpcs://triton.fnal.gov:443", ("grpcs", "triton.fnal.gov:443")),
        ("grpc://[::1]:8001", ("grpc", "[::1]:8001")),
    ],
)
def test_split_endpoint_gives_the_scheme_and_host_port(endpoint: str, split: tuple[str, str]) -> None:
    assert split_endpoint(endpoint) == split


def _names_the_schemes(err: pytest.ExceptionInfo[ValueError]) -> None:
    for scheme in ("tcp", "https", "grpcs"):  # http and grpc are their substrings
        assert scheme in str(err.value), scheme


@pytest.mark.parametrize(
    "endpoint",
    ["svc:8001", "triton://svc:1", "grpc://svc", "http://svc:8000/v2"],
    ids=["bare", "unknown-scheme", "no-port", "path"],
)
def test_split_endpoint_refuses_other_forms(endpoint: str) -> None:
    with pytest.raises(ValueError) as err:
        split_endpoint(endpoint)
    _names_the_schemes(err)


def test_bind_refuses_a_bare_endpoint_and_leaves_the_plan_unbound() -> None:
    plan = _served("bind-bare")
    with pytest.raises(ValueError) as err:
        bind_services(plan, {"scorer-svc": "bind-bare:8000"})
    _names_the_schemes(err)
    assert _endpoints(plan) == [None]
    plain: Plan[Any] = Plan(process=lambda part, res: [], combine=lambda a, b: a, empty=list)
    with pytest.raises(ValueError):
        bind_services(plain, {"unused": "bind-bare:8000"})
