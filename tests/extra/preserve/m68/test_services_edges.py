"""Service-surface edges the frozen m68 suite leaves untaken: pickling (specs ride a stdlib-pickled
``Plan`` to a driver job; ``UnboundService`` crosses back from a worker), two more refused endpoint
forms, a literal Triton url whose scheme no client speaks, a hashable spec with a recipe, and a
``RunReport`` pickled before it had ``endpoints``."""

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


def test_a_spec_with_a_recipe_hashes() -> None:
    def spec(env: str) -> ServiceSpec:
        return ServiceSpec(
            "svc", "http", launch=Launch(argv=("serve",), env={"A": env}, resources={"gpus": 1.0})
        )

    assert hash(spec("1")) == hash(spec("1"))
    assert len({spec("1"), spec("1"), spec("2")}) == 2


class _Pickled:
    """Pickles to a stream that builds ``cls`` from ``state`` without ``__init__``, as 0.0.6 wrote it."""

    def __init__(self, cls: type, state: dict[str, Any]) -> None:
        self.cls, self.state = cls, state

    def __reduce__(self) -> tuple[Any, ...]:
        return (object.__new__, (self.cls,), self.state)


def test_a_run_report_pickled_without_endpoints_loads() -> None:
    from graphed.core import ExecResult  # noqa: PLC0415
    from graphed.debug import RunRecorder, RunReport  # noqa: PLC0415

    report = RunRecorder().report(result=ExecResult(value=0, n_partitions=0, n_combines=0))
    state = {k: v for k, v in vars(report).items() if k != "endpoints"}
    old = pickle.loads(pickle.dumps(_Pickled(RunReport, state)))
    assert type(old) is RunReport and dict(old.endpoints) == {}
    assert old.to_json() == report.to_json()
    assert pickle.loads(pickle.dumps(old)) == report
