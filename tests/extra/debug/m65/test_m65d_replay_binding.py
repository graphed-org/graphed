"""Replay binds sources and externals as the run does, and fails where the run fails."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow")

import awkward as ak
import numpy as np

import graphed.debug as gd
from graphed import Array, Session
from graphed.aggregate import _PartitionReduce, aggregate_plan
from graphed.awkward import AwkwardBackend, from_awkward, from_parquet, gak
from graphed.core import LocalResources, PayloadDescriptor, Plan
from graphed.debug.replaying import Replay
from graphed.errors import GraphedError

N = 1000


def _first(v: list[Any]) -> float:
    return float(v[0])


def _add(a: float, b: float) -> float:
    return a + b


def _zero() -> float:
    return 0.0


def _ext_sum(x: Any) -> float:
    return float(ak.sum(x))


@dataclass(frozen=True)
class _ExtForm:
    def describe(self) -> str:
        return "sum"


def _events(tmp_path: Path) -> tuple[Session, Array]:
    path = tmp_path / "d.parquet"
    ak.to_parquet(ak.Array({"x": np.arange(float(N))}), path)
    s = Session(AwkwardBackend())
    return s, from_parquet(s, "events", str(path))


def _plan(y: Array, **kw: Any) -> Plan[Any]:
    return aggregate_plan(y, reduce=_first, combine=_add, empty=_zero, steps_per_file=4, **kw)


def _replay_error(r: Replay) -> BaseException:
    with pytest.raises(Exception) as info:
        r.value  # noqa: B018
    exc = info.value
    return exc if isinstance(exc, GraphedError) else exc.__cause__  # type: ignore[return-value]


def test_a_second_source_is_unbound_in_replay_as_in_the_run(tmp_path: Path) -> None:
    s, ev = _events(tmp_path)
    other = from_awkward(s, "other", ak.Array({"x": np.ones(N)}))
    y = gak.sum(ev.x * other.x, axis=None)
    plan = _plan(y)
    assert isinstance(plan.process, _PartitionReduce)
    with pytest.raises(GraphedError, match="no data bound for source 'other'") as run:
        plan.process(plan.tasks[0].partition, LocalResources())
    err = _replay_error(gd.replay(plan, 0, y))
    assert isinstance(err, GraphedError)
    assert str(err) == str(run.value)


def test_an_external_without_evaluator_fails_replay_as_in_the_run(tmp_path: Path) -> None:
    s, ev = _events(tmp_path)
    desc = PayloadDescriptor(
        kind="histogram",
        content_hash="sha256:m65d-binding",
        framework="boost_histogram",
        version="1",
        io_schema="uhi",
        preprocessing_ref=None,
    )
    h = s.record_external("histogram", _ext_sum, [ev.x], {"spec": "m65d"}, descriptor=desc, form=_ExtForm())
    plan = _plan(h)
    assert isinstance(plan.process, _PartitionReduce)
    process = dataclasses.replace(plan.process, externals=())
    with pytest.raises(GraphedError, match="needs an evaluator") as run:
        process(plan.tasks[0].partition, LocalResources())
    err = _replay_error(Replay(process, plan.tasks[0], (h,)))
    assert isinstance(err, GraphedError)
    assert str(err) == str(run.value)
