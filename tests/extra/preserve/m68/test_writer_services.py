"""A ``to_parquet`` write plan carries the services its recording names and binds their endpoints,
on the plain and the varied awkward writers alike."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
import graphed.awkward as ga
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.core import Plan, SequentialRunner
from graphed.preserve import ExternalPlugin, record_external, sha256_bytes
from graphed.services import ServiceSpec, UnboundService, bind_services

pytest.importorskip("pyarrow")

X = np.arange(6.0)
SPEC = ServiceSpec("svc", "http")


def _first(resource: Any, params: Any, inputs: list[Any]) -> Any:
    return inputs[0]


PLUGIN = ExternalPlugin(kind="m68_writer", content_hash=sha256_bytes, evaluate=_first, samples=lambda: [b"w"])


def _served(session: Session, x: Any) -> Any:
    session.declare_service(SPEC)
    return record_external(session, PLUGIN, b"w", [x], params={"service": "svc"})


def _awkward(dest: str) -> Any:
    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", ak.Array({"x": X}))
    return ga.to_parquet(_served(s, ev.x), dest, compute=False)


def _awkward_varied(dest: str) -> Any:
    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", ak.Array({"x": X}))
    record = gak.zip({"x": ev.x, "y": _served(s, ev.x)})
    select = graphed.vary(ev.x > 1, "cut", up=ev.x > 2, down=ev.x > 0)
    return ga.to_parquet(record, dest, select=select, compute=False)


@pytest.mark.parametrize("write", [_awkward, _awkward_varied], ids=["plain", "varied"])
def test_a_write_plan_carries_and_binds_its_services(write: Callable[[str], Any], tmp_path: Any) -> None:
    plan: Plan[list[str]] = write(str(tmp_path / "out"))
    assert plan.services == (SPEC,)
    with pytest.raises(UnboundService, match="svc"):
        SequentialRunner().run(plan)
    bound = bind_services(plan, {"svc": "http://h:8000"})
    process: Any = bound.process
    assert [getattr(fn, "endpoint", None) for _key, fn in process.externals] == ["http://h:8000"]
    assert len(SequentialRunner().run(bound).value) == 1
