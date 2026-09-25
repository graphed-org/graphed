"""m68 §3.2 — the service surface is analysis data: a JSON-round-tripping spec, declared on the
Session, carried by the runtime Plan, the DurablePlan and the bundle, listed by ``inspect()``;
run endpoints ride the RunReport, outside the fingerprint."""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

import pytest
from m68_services_fixtures import HIST, X, descriptor, mark, new_events, score, values_plan

from graphed.core import DurablePlan, ExecResult, OpSpec, Partition
from graphed.debug import RunRecorder, RunReport
from graphed.preserve import TRITON_PLUGIN, Bundle, attach_run_report, build_bundle, inspect
from graphed.services import Launch, ServiceSpec

IMAGE = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/nvcr.io/tritonserver:25.11"


def _managed() -> ServiceSpec:
    return ServiceSpec(
        "scorer-svc",
        "triton-gpu",
        check="http:/v2/health/ready",
        ports=(10001, 10100),
        launch=Launch(
            argv=("tritonserver", "--http-port={port}", "--model-repository=models"),
            image=IMAGE,
            inputs=("models.tar",),
            env={"OMP_NUM_THREADS": "1"},
            resources={"cpus": 2.0, "memory_mb": 4096.0, "gpus": 1.0},
        ),
        timeout_s=900.0,
    )


def _site() -> ServiceSpec:
    return ServiceSpec("scorer-site", "triton-eaf")


def _unused() -> ServiceSpec:
    return ServiceSpec("unused-svc", "http")


def _json(doc: Any) -> Any:
    return json.loads(json.dumps(doc))


def test_defaults_are_the_planned_ones_and_specs_are_frozen() -> None:
    site = _site()
    assert (site.check, site.ports, site.launch, site.timeout_s) == ("tcp", (10000, 10100), None, 600.0)
    bare = Launch(argv=("serve",))
    assert (bare.image, bare.inputs, dict(bare.env), dict(bare.resources)) == (None, (), {}, {})
    with pytest.raises(dataclasses.FrozenInstanceError):
        site.name = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        bare.image = "x"  # type: ignore[misc]
    for mapping in (bare.env, bare.resources):  # §3 defaults rule: read-only, not a shared mutable dict
        with pytest.raises(TypeError):
            mapping["k"] = 1.0  # type: ignore[index]


@pytest.mark.parametrize("make", [_managed, _site], ids=["managed", "external-only"])
def test_spec_round_trips_through_json_text(make: Any) -> None:
    spec = make()
    back = ServiceSpec.from_json(_json(spec.to_json()))
    assert back == spec
    assert back.ports == spec.ports and isinstance(back.ports, tuple)


def test_session_registry_refuses_conflicts_and_undeclared_names() -> None:
    ev = new_events()
    s = ev.session
    s.declare_service(_managed())
    s.declare_service(_site())
    s.declare_service(_managed())  # an equal re-declaration is idempotent
    with pytest.raises(ValueError, match="scorer-svc"):
        s.declare_service(dataclasses.replace(_managed(), kind="other"))
    with pytest.raises(ValueError, match="scorer-svc"):
        s.declare_service(dataclasses.replace(_managed(), launch=None))
    assert s.services() == {"scorer-svc": _managed(), "scorer-site": _site()}
    assert s.service_for("scorer-site") == _site()
    declared = r"(?s)(scorer-site.*scorer-svc|scorer-svc.*scorer-site)"
    with pytest.raises(Exception, match=r"(?s)nope") as looked_up:
        s.service_for("nope")
    assert looked_up.match(declared)
    assert ev.x is not None  # interned before counting, so only a refused External could add a node
    before = s.node_count()
    with pytest.raises(Exception, match=r"(?s)nope") as recorded:
        score(ev, descriptor("undeclared"), service="nope")
    assert recorded.match(declared)
    with pytest.raises(Exception, match=r"(?s)nope") as generic:
        mark(ev, "undeclared", "nope")
    assert generic.match(declared)
    assert s.node_count() == before


def test_an_empty_registry_refuses_a_named_service() -> None:
    ev = new_events()
    assert ev.x is not None  # interned before counting
    before = ev.session.node_count()
    with pytest.raises(Exception, match="nope"):
        score(ev, descriptor("r6-empty"), service="nope")
    assert ev.session.node_count() == before


def test_services_is_a_copy_like_sources() -> None:
    ev = new_events()
    ev.session.declare_service(ServiceSpec("a-svc", "http"))
    ev.session.services()["a-svc"] = ServiceSpec("a-svc", "other")
    assert ev.session.service_for("a-svc") == ServiceSpec("a-svc", "http")


def test_plan_and_durable_plan_carry_the_referenced_specs() -> None:
    ev = new_events()
    for spec in (_managed(), _site(), _unused()):
        ev.session.declare_service(spec)
    weight = score(ev, descriptor("durable"), service="scorer-svc")
    score(ev, descriptor("dead"), service="scorer-site")
    assert values_plan(weight).services == (_managed(),)
    assert values_plan(weight, services=("unused-svc",)).services == (_managed(), _unused())  # sorted by name
    assert values_plan(weight, services=("scorer-svc",)).services == (_managed(),)  # a union
    with pytest.raises(Exception, match="nope") as err:
        values_plan(weight, services=("nope",))
    assert err.match(r"(?s)(scorer-site.*scorer-svc|scorer-svc.*scorer-site)")
    plan = values_plan(weight)
    parts = (Partition("mem://events", "", 0, 4), Partition("mem://events", "", 4, len(X)))
    durable = DurablePlan(
        ir=plan.process.ir,
        process=OpSpec.from_ref("builtins:list"),
        combine=OpSpec.from_ref("operator:add"),
        empty=OpSpec.from_ref("builtins:list"),
        partitions=parts,
        services=plan.services,
    )
    raw = durable.to_bytes()
    assert json.loads(raw)["services"] == [_json(_managed().to_json())]
    back = DurablePlan.from_bytes(raw)
    assert back.services == (_managed(),)
    assert back.to_bytes() == raw
    kept = {f.name: getattr(durable, f.name) for f in dataclasses.fields(durable) if f.name != "services"}
    bare = DurablePlan(**kept)
    assert bare.services == ()
    assert [durable.task_id(p) for p in parts] == [bare.task_id(p) for p in parts]  # the spec is not identity


def _bundle(root: Any) -> Bundle:
    ev = new_events()
    for spec in (_managed(), _site(), _unused()):
        ev.session.declare_service(spec)
    managed, site = descriptor("bundle-managed"), descriptor("bundle-site")
    weight = score(ev, managed, service="scorer-svc") * score(ev, site, service="scorer-site")
    payloads = {TRITON_PLUGIN.content_hash(p): p for p in (managed, site)}
    return build_bundle(
        root,
        session=ev.session,
        value=ev.x,
        weight=weight,
        datasets={"events": {"x": X}},
        payloads=payloads,
        histogram=HIST,
    )


def test_bundle_manifest_and_inspect_list_the_referenced_specs(tmp_path: Any) -> None:
    opened = Bundle.open(_bundle(tmp_path / "b").root)
    assert opened.manifest["services"] == [_json(s.to_json()) for s in (_site(), _managed())]  # name order
    lines = inspect(opened).splitlines()
    (header,) = [i for i, line in enumerate(lines) if line.strip().startswith("services")]
    assert header > next(i for i, line in enumerate(lines) if "external payloads" in line)
    block = "\n".join(lines[header:])
    for token in (
        "scorer-svc",
        "triton-gpu",
        "http:/v2/health/ready",
        IMAGE,
        "gpus",
        "scorer-site",
        "triton-eaf",
    ):
        assert token in block, token
    assert len(re.findall(r"\bexternal only\b", block)) == 1  # only scorer-site lacks a recipe
    assert "unused-svc" not in "\n".join(lines)


def test_specs_are_listed_in_name_order_whatever_the_kind(tmp_path: Any) -> None:
    names = [f"svc-{c}" for c in "hgfedcba"]  # declared and recorded against name order
    ev = new_events()
    for name in names:
        launch = Launch(argv=("http-serve", "--port={port}")) if name == "svc-c" else None
        ev.session.declare_service(ServiceSpec(name, "http", launch=launch))
    payloads = [descriptor(f"order-{name}") for name in names[:-1]]
    nodes = [score(ev, p, service=n) for p, n in zip(payloads, names[:-1], strict=True)]
    nodes.append(mark(ev, "order", "svc-a"))
    weight = nodes[0]
    for node in nodes[1:]:
        weight = weight * node
    assert [s.name for s in values_plan(weight).services] == sorted(names)
    assert [s.name for s in values_plan(nodes[0], services=names[1:]).services] == sorted(names)
    bundle = build_bundle(
        tmp_path / "b",
        session=ev.session,
        value=ev.x,
        weight=weight,
        datasets={"events": {"x": X}},
        payloads={TRITON_PLUGIN.content_hash(p): p for p in payloads},
        histogram=HIST,
    )
    opened = Bundle.open(bundle.root)
    assert [s["name"] for s in opened.manifest["services"]] == sorted(names)
    # an image-less recipe prints its argv
    assert re.search(r"(?<![\w-])http-serve(?![\w-])", inspect(opened))


def test_run_report_endpoints_are_run_provenance(tmp_path: Any) -> None:
    bundle = _bundle(tmp_path / "b")
    manifest_bytes = (bundle.root / "manifest.json").read_bytes()
    done = ExecResult(value=0, n_partitions=0, n_combines=0)
    endpoints = {"scorer-svc": "grpc://10.1.2.3:8001", "scorer-site": "http://eaf.example:8000"}
    report = RunRecorder().report(result=done, endpoints=endpoints)
    assert dict(report.endpoints) == endpoints
    assert report.to_json()["version"] == 1
    bare = RunReport(**{k: v for k, v in vars(report).items() if k != "endpoints"})
    assert dict(bare.endpoints) == {}
    with pytest.raises(TypeError):
        bare.endpoints["k"] = "v"  # type: ignore[index]
    assert dict(RunReport.from_json(_json(report.to_json())).endpoints) == dict(report.endpoints)
    legacy = report.to_json()
    legacy.pop("endpoints", None)  # a 0.0.6 report has no such key
    assert dict(RunReport.from_json(legacy).endpoints) == {}
    first = attach_run_report(bundle, report.to_json())
    second = attach_run_report(bundle, RunRecorder().report(result=done).to_json())
    opened = Bundle.open(bundle.root)
    assert (bundle.root / "manifest.json").read_bytes() == manifest_bytes
    assert opened.fingerprint() == bundle.fingerprint()
    lines = inspect(opened).splitlines()
    heads = [
        next(i for i, line in enumerate(lines) if line.startswith(f"    {d[:12]} ")) for d in (first, second)
    ]
    (arrow,) = [i for i, line in enumerate(lines) if "endpoints:" in line]
    for name, endpoint in endpoints.items():  # each pair a whole token on the one endpoints line
        assert re.search(rf"(?<![\w-]){name} → {re.escape(endpoint)}(?!\w)", lines[arrow]), name
    assert heads[0] < arrow < heads[1]
    assert sum("endpoints:" in line for line in lines) == 1
