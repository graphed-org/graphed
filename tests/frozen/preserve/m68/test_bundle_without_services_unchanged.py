"""m68 D1 — a plan or bundle that references no service is byte-identical to 0.0.6: ``services`` is
written only when the IR references a declared spec. The pins below were measured on 0.0.6."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import pickle
from typing import Any

from m68_services_fixtures import HIST, X, descriptor, new_events, score, values_plan

from graphed.core import DurablePlan, GraphStore, OpSpec, Partition, Plan
from graphed.preserve import TRITON_PLUGIN, Bundle, build_bundle

MANIFEST_KEYS = {
    "format_version",
    "analysis",
    "sources",
    "externals",
    "opaque_nodes",
    "provenance",
    "environment",
    "config",
    "seed",
}
PLAN_KEYS = {
    "format_version",
    "ir_b64",
    "process",
    "combine",
    "empty",
    "partitions",
    "read_columns",
    "stopping",
    "file_locality",
    "resource_hints",
}
PLAN_SHA256 = "53776410fa12314391d9929da9110354e47aee65ce16869fffb885e08bd499b2"
# pickle protocol 4 of Plan(process=list, combine=list, empty=list) and of
# _PluginEvaluator(TRITON_PLUGIN, b"", {}), both written by 0.0.6
PLAN_PICKLE_0_0_6 = (
    "gASViwAAAAAAAACMFmdyYXBoZWQuY29yZS5leGVjdXRpb26UjARQbGFulJOUKYGUfZQojAdwcm9jZXNzlIwIYnVpbHRpbnOUjARs"
    "aXN0lJOUjAdjb21iaW5llGgIjAVlbXB0eZRoCIwFdGFza3OUKYwKbmV4dF90YXNrc5ROjARzdG9wlE6MCW9wZW5fb25jZZSJdWIu"
)
EVALUATOR_PICKLE_0_0_6 = (
    "gASViAEAAAAAAACMIGdyYXBoZWQucHJlc2VydmUuZXh0ZXJuYWxzLl9iYXNllIwQX1BsdWdpbkV2YWx1YXRvcpSTlCmBlH2UKIwG"
    "cGx1Z2lulGgAjA5FeHRlcm5hbFBsdWdpbpSTlCmBlH2UKIwEa2luZJSMDHRyaXRvbl9tb2RlbJSMDGNvbnRlbnRfaGFzaJSMKmdy"
    "YXBoZWQucHJlc2VydmUuZXh0ZXJuYWxzLnRyaXRvbl9leHRlcm5hbJSME3RyaXRvbl9jb250ZW50X2hhc2iUk5SMCGV2YWx1YXRl"
    "lGgNjAtldmFsX3RyaXRvbpSTlIwHc2FtcGxlc5RoDYwPX3RyaXRvbl9zYW1wbGVzlJOUjARsb2FklGgNjAtsb2FkX3RyaXRvbpST"
    "lIwFY2xvc2WUaA2MDGNsb3NlX3RyaXRvbpSTlIwJZnJhbWV3b3JrlIwMdHJpdG9uY2xpZW50lIwKc3ludGhlc2l6ZZROdWKMB3Bh"
    "eWxvYWSUQwCUjAtub2RlX3BhcmFtc5R9lHViLg=="
)


def _durable() -> DurablePlan:
    g = GraphStore()
    src = g.add_source("events", {"uri": "mem://events"})
    out = g.add_reduction("sum", [g.add_op("x", [src])])
    return DurablePlan(
        ir=g.serialize(outputs=[out]),
        process=OpSpec.from_ref("builtins:list"),
        combine=OpSpec.from_ref("operator:add"),
        empty=OpSpec.from_ref("builtins:list"),
        partitions=(Partition("mem://events", "Events", 0, 4), Partition("mem://events", "Events", 4, 8)),
        read_columns=("x",),
    )


def _bundle(root: Any, *, declare: bool, service: bool) -> Bundle:
    ev = new_events()
    if declare:
        from graphed.services import ServiceSpec  # noqa: PLC0415  (the 0.0.6 pins collect without it)

        ev.session.declare_service(ServiceSpec("scorer-svc", "triton"))
    payload = descriptor("unchanged")
    where = {"service": "scorer-svc"} if service else {"url": "triton://unchanged:8000"}
    return build_bundle(
        root,
        session=ev.session,
        value=ev.x,
        weight=score(ev, payload, **where),
        datasets={"events": {"x": X}},
        payloads={TRITON_PLUGIN.content_hash(payload): payload},
        histogram=HIST,
    )


def test_a_bundle_without_services_keeps_the_0_0_6_manifest_keys(tmp_path: Any) -> None:
    bundle = _bundle(tmp_path / "b", declare=False, service=False)
    assert set(bundle.manifest) == MANIFEST_KEYS
    assert set(Bundle.open(bundle.root).manifest) == MANIFEST_KEYS


def test_a_durable_plan_without_services_is_byte_identical_to_0_0_6() -> None:
    raw = _durable().to_bytes()
    assert set(json.loads(raw)) == PLAN_KEYS
    assert hashlib.sha256(raw).hexdigest() == PLAN_SHA256
    assert DurablePlan.from_bytes(raw).to_bytes() == raw


def test_one_referenced_spec_adds_exactly_the_services_key(tmp_path: Any) -> None:
    from graphed.services import ServiceSpec  # noqa: PLC0415

    spec = ServiceSpec("scorer-svc", "triton")
    bundle = _bundle(tmp_path / "b", declare=True, service=True)
    assert set(Bundle.open(bundle.root).manifest) == MANIFEST_KEYS | {"services"}
    bare = _durable()
    assert DurablePlan.from_bytes(bare.to_bytes()).services == ()
    with_spec = json.loads(dataclasses.replace(bare, services=(spec,)).to_bytes())
    assert with_spec.pop("services") == [json.loads(json.dumps(spec.to_json()))]
    assert with_spec == json.loads(bare.to_bytes())


def test_a_declared_but_unreferenced_spec_is_not_written(tmp_path: Any) -> None:
    plain = _bundle(tmp_path / "plain", declare=False, service=False)
    declared = _bundle(tmp_path / "declared", declare=True, service=False)
    assert (declared.root / "manifest.json").read_bytes() == (plain.root / "manifest.json").read_bytes()
    ev = new_events()
    from graphed.services import ServiceSpec  # noqa: PLC0415

    ev.session.declare_service(ServiceSpec("scorer-svc", "triton"))
    assert values_plan(score(ev, descriptor("unreferenced"), url="triton://unchanged:8000")).services == ()
    assert Plan(process=list, combine=list, empty=list).services == ()


def test_0_0_6_pickles_load_without_services() -> None:
    plan = pickle.loads(base64.b64decode(PLAN_PICKLE_0_0_6))
    assert isinstance(plan, Plan) and plan.services == ()
    evaluator = pickle.loads(base64.b64decode(EVALUATOR_PICKLE_0_0_6))
    assert evaluator.plugin.kind == "triton_model" and evaluator.endpoint is None
