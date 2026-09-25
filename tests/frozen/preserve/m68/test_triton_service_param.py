"""m68 D7 — a Triton node names a service (``service=``) or a literal ``url=``, never both; the endpoint
leaves graph identity, and the per-process connection cache keys on it."""

from __future__ import annotations

import hashlib

import fake_triton_services as fake
import numpy as np
import pytest
from m68_services_fixtures import descriptor, expected, new_events, score, values_plan

from graphed.core import GraphStore, SequentialRunner
from graphed.preserve import PreserveError
from graphed.services import ServiceSpec, bind_services

URL_IR_SHA256 = "02c61c847ce7d966070e1383069601aefd2481872a7ed6708133a56326f97d59"


def _externals(ir: bytes) -> list[dict[str, object]]:
    return [dict(n["params"]) for n in GraphStore.deserialize(ir).nodes() if n["kind"] == "external"]


def test_the_endpoint_never_enters_the_ir() -> None:
    ev = new_events()
    ev.session.declare_service(ServiceSpec("scorer-svc", "triton"))
    plan = values_plan(score(ev, descriptor("ir"), service="scorer-svc"))
    a = bind_services(plan, {"scorer-svc": "grpc://ir-a:8000"})
    b = bind_services(plan, {"scorer-svc": "http://ir-b:8000"})
    assert a.process.ir == b.process.ir == plan.process.ir
    (params,) = _externals(plan.process.ir)
    assert params["service"] == "scorer-svc" and "url" not in params


def test_the_url_path_records_the_0_0_6_bytes() -> None:
    ev = new_events()
    weight = score(ev, descriptor("url-pin"), url="triton://pinned:8000")
    ir = ev.session.serialized_ir(weight, optimize=False)
    assert hashlib.sha256(ir).hexdigest() == URL_IR_SHA256


def test_two_services_with_one_payload_connect_twice() -> None:
    fake.serve("grpc://two-a:8000", descriptor("two", 0.45, -0.1))
    fake.serve("grpc://two-b:8000", descriptor("two", 0.8, 0.2))
    ev = new_events()
    for name in ("svc-a", "svc-b"):
        ev.session.declare_service(ServiceSpec(name, "triton"))
    payload = descriptor("two")
    plan = values_plan(score(ev, payload, service="svc-a"), score(ev, payload, service="svc-b"))
    bound = bind_services(plan, {"svc-a": "grpc://two-a:8000", "svc-b": "grpc://two-b:8000"})
    row_a, row_b = SequentialRunner().run(bound).value
    assert np.allclose(row_a, expected(0.45, -0.1), rtol=1e-6)
    assert np.allclose(row_b, expected(0.8, 0.2), rtol=1e-6)
    assert fake.SERVERS["grpc://two-a:8000"].infer_calls > 0
    assert fake.SERVERS["grpc://two-b:8000"].infer_calls > 0


def test_rebinding_connects_to_the_new_endpoint() -> None:
    fake.serve("http://re-x:8000", descriptor("re", 0.45, -0.1))
    fake.serve("grpcs://re-y:443", descriptor("re", 0.8, 0.2))
    ev = new_events()
    ev.session.declare_service(ServiceSpec("scorer-svc", "triton"))
    plan = values_plan(score(ev, descriptor("re"), service="scorer-svc"))
    (row_x,) = SequentialRunner().run(bind_services(plan, {"scorer-svc": "http://re-x:8000"})).value
    (row_y,) = SequentialRunner().run(bind_services(plan, {"scorer-svc": "grpcs://re-y:443"})).value
    assert np.allclose(row_x, expected(0.45, -0.1), rtol=1e-6)
    assert np.allclose(row_y, expected(0.8, 0.2), rtol=1e-6)
    assert fake.SERVERS["grpcs://re-y:443"].infer_calls > 0


@pytest.mark.parametrize(
    "where",
    [{"url": "triton://both:8000", "service": "scorer-svc"}, {}],
    ids=["both", "neither"],
)
def test_url_and_service_are_exclusive(where: dict[str, str]) -> None:
    ev = new_events()
    ev.session.declare_service(ServiceSpec("scorer-svc", "triton"))
    with pytest.raises(PreserveError, match=r"(?s)(url.*service|service.*url)"):
        score(ev, descriptor("xor"), **where)
