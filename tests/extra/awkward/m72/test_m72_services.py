"""Every m72 plan shape whose tasks evaluate a service-calling External carries that service and
binds its endpoint: a writes-only plan, writes beside reductions, and collated plans (the union)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import awkward as ak
import pytest
from m72_multiout_fixtures import (
    DATA_FILES,
    MC_FILES,
    FileSource,
    add,
    by_step,
    data_combine,
    data_empty,
    data_reduce,
    json_codec,
    no_paths,
    paths_only,
    read_part,
)

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, AwkwardForm, gak
from graphed.core import Plan
from graphed.core.execution import SequentialRunner
from graphed.preserve import ExternalPlugin, record_external, sha256_bytes
from graphed.services import ServiceSpec, UnboundService, bind_services
from graphed.write import PartWrite

SVC = ServiceSpec("svc", "http")
OTHER = ServiceSpec("other", "grpc")
ENDPOINTS = {"svc": "http://h:8000", "other": "grpc://h:8001"}


def _plus_one(resource: Any, params: Any, inputs: list[Any]) -> Any:
    return inputs[0] + 1


PLUGIN = ExternalPlugin(
    kind="m72_served", content_hash=sha256_bytes, evaluate=_plus_one, samples=lambda: [b"s"]
)


class AkFiles:
    """The fixtures' in-memory files, read as awkward chunks."""

    def __init__(self, files: dict[str, Any], tree: str) -> None:
        self.files = FileSource(files, tree)

    def partitions(self, steps_per_file: int = 1) -> Any:
        return self.files.partitions(steps_per_file)

    def read_partition(self, partition: Any, columns: Any, resources: Any) -> Any:
        return ak.Array(self.files.read_partition(partition, columns, resources))


def _served(files: dict[str, Any], tree: str, spec: ServiceSpec) -> Any:
    session = Session(AwkwardBackend())
    form = AwkwardForm(
        ak.Array(ak.Array(next(iter(files.values()))).layout.to_typetracer(forget_length=True))
    )
    x = session.source("x", form=form, data=AkFiles(files, tree))
    session.declare_service(spec)
    return x, record_external(session, PLUGIN, b"s", [x], params={"service": spec.name})


def _run_bound(plan: Plan[Any], services: tuple[ServiceSpec, ...]) -> Any:
    assert plan.services == services
    with pytest.raises(UnboundService):
        SequentialRunner().run(plan)
    return SequentialRunner().run(bind_services(plan, ENDPOINTS)).value


def test_a_writes_only_plan_carries_and_binds_the_written_service(tmp_path: Path) -> None:
    _x, y = _served(MC_FILES, "Events", SVC)
    write = PartWrite(array=y, destination=str(tmp_path), name=by_step, codec=json_codec)
    plan = graphed.aggregate_plan(reduce=paths_only, combine=add, empty=no_paths, writes=[write])
    paths = _run_bound(plan, (SVC,))
    assert read_part(paths[0])["values"] == (MC_FILES["mc-a"] + 1).tolist()


def test_writes_beside_a_plain_reduction_carry_the_written_service(tmp_path: Path) -> None:
    x, y = _served(DATA_FILES, "Runs", SVC)
    write = PartWrite(array=y, destination=str(tmp_path), name=by_step, codec=json_codec)
    plan = graphed.aggregate_plan(
        gak.sum(x), reduce=data_reduce, combine=data_combine, empty=data_empty, writes=[write]
    )
    total, paths = _run_bound(plan, (SVC,))
    assert total == sum(int(v.sum()) for v in DATA_FILES.values())
    assert read_part(paths[0])["values"] == (DATA_FILES["mc-a"] + 1).tolist()


def test_a_collated_plan_carries_the_union_and_binds_every_part(tmp_path: Path) -> None:
    _x, mc_y = _served(MC_FILES, "Events", SVC)
    write = PartWrite(array=mc_y, destination=str(tmp_path), name=by_step, codec=json_codec)
    mc = graphed.aggregate_plan(reduce=paths_only, combine=add, empty=no_paths, writes=[write])
    _x, data_y = _served(DATA_FILES, "Runs", OTHER)
    data = graphed.aggregate_plan(gak.sum(data_y), reduce=data_reduce, combine=data_combine, empty=data_empty)
    value = _run_bound(graphed.collate({"mc": graphed.collate({"mc": mc}), "data": data}), (OTHER, SVC))
    assert value["data"][0] == sum(int(v.sum()) + len(v) for v in DATA_FILES.values())
    assert read_part(value["mc"]["mc"][0])["values"] == (MC_FILES["mc-a"] + 1).tolist()


def test_collate_refuses_one_service_name_declared_two_ways() -> None:
    _x, a = _served(MC_FILES, "Events", SVC)
    _x, b = _served(DATA_FILES, "Runs", ServiceSpec("svc", "grpc"))
    plans = {
        n: graphed.aggregate_plan(gak.sum(y), reduce=data_reduce, combine=data_combine, empty=data_empty)
        for n, y in (("mc", a), ("data", b))
    }
    with pytest.raises(ValueError, match="'svc'"):
        graphed.collate(plans)
