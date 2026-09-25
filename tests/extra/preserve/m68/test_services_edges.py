"""Service-surface edges the frozen m68 suite leaves untaken: pickling (specs ride a stdlib-pickled
``Plan`` to a driver job; ``UnboundService`` crosses back from a worker), two more refused endpoint
forms, and a literal Triton url whose scheme no client speaks."""

from __future__ import annotations

import pickle
from typing import Any

import pytest

from graphed.core import Partition, Plan, WorkerResources
from graphed.preserve import PreserveError
from graphed.preserve.externals.triton_external import load_triton
from graphed.services import Launch, ServiceSpec, UnboundService, split_endpoint


def _process(partition: Partition, resources: WorkerResources) -> list[int]:
    return []


def _combine(a: list[int], b: list[int]) -> list[int]:
    return a + b


def test_a_plan_with_a_recipe_pickles() -> None:
    spec = ServiceSpec("svc", "http", launch=Launch(argv=("serve",), env={"A": "1"}, resources={"cpus": 2.0}))
    plan: Plan[list[int]] = Plan(process=_process, combine=_combine, empty=list, services=(spec,))
    back: Plan[Any] = pickle.loads(pickle.dumps(plan))
    assert back.services == (spec,)
    launch = back.services[0].launch
    assert launch is not None and dict(launch.env) == {"A": "1"}
    with pytest.raises(TypeError):
        launch.env["B"] = "2"  # type: ignore[index]


def test_unbound_service_pickles_with_its_name() -> None:
    err = pickle.loads(pickle.dumps(UnboundService("svc")))
    assert err.name == "svc"
    assert str(err) == str(UnboundService("svc"))


@pytest.mark.parametrize("endpoint", ["http://h:99999", "http://user@h:8000"], ids=["port-range", "userinfo"])
def test_split_endpoint_refuses_a_bad_port_and_userinfo(endpoint: str) -> None:
    with pytest.raises(ValueError, match="scheme://host:port"):
        split_endpoint(endpoint)


def test_a_literal_url_with_an_unknown_scheme_is_a_preserve_error() -> None:
    with pytest.raises(PreserveError, match="triton://h:1"):
        load_triton(b"", {"url": "triton://h:1"})


def test_a_run_report_with_endpoints_pickles() -> None:
    from graphed.core import ExecResult  # noqa: PLC0415
    from graphed.debug import RunRecorder  # noqa: PLC0415

    report = RunRecorder().report(
        result=ExecResult(value=0, n_partitions=0, n_combines=0), endpoints={"svc": "grpc://h:8001"}
    )
    back = pickle.loads(pickle.dumps(report))
    assert back == report and dict(back.endpoints) == {"svc": "grpc://h:8001"}
    with pytest.raises(TypeError):
        back.endpoints["other"] = "http://h:1"


def test_reproduce_refuses_a_node_that_calls_a_service(tmp_path: Any) -> None:
    import awkward as ak  # noqa: PLC0415

    from graphed import Session  # noqa: PLC0415
    from graphed.awkward import AwkwardBackend, from_awkward  # noqa: PLC0415
    from graphed.preserve import TRITON_PLUGIN, build_bundle, record_external, reproduce  # noqa: PLC0415

    s = Session(AwkwardBackend())
    s.declare_service(ServiceSpec("tagger", "triton"))
    data = ak.Array({"x": [0.5, 1.5]})
    events = from_awkward(s, "events", data)
    model = b'{"model": "tagger", "version": "1"}'
    score = record_external(
        s, TRITON_PLUGIN, model, [events.x], params={"service": "tagger", "model": "tagger"}
    )
    bundle = build_bundle(
        tmp_path,
        session=s,
        value=events.x,
        weight=score,
        datasets={"events": data},
        payloads={TRITON_PLUGIN.content_hash(model): model},
        histogram={"name": "x", "bins": 2, "lo": 0.0, "hi": 2.0},
    )
    with pytest.raises(PreserveError, match="calls service 'tagger'"):
        reproduce(bundle)
