"""m68c edges the frozen suite does not take: the executors' payload slicer, a partial fold set, and
a join side that reads no partitioned source."""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import replace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "frozen", "awkward", "m68c"))
from m68c_services_harness import FOLD, two_sources

import graphed
from graphed.awkward import AwkwardForm
from graphed.shuffle import pick, split


def test_pick_keeps_one_dest_of_a_map_payload() -> None:
    ev, lu = two_sources()
    plan = graphed.join_plan(graphed.join(ev, lu, on=["run"]), steps_per_file=2)
    first = plan.stages[0]
    payload = first.process.resolve()(first.tasks[0], (), None)
    mapping = split(payload)
    assert set(mapping) == {0, 1}
    assert split(pick(mapping, 1)) == {1: mapping[1]}
    assert pickle.loads(pick(mapping, 0)) == {0: mapping[0]}


@pytest.mark.parametrize("given", [("reduce",), ("reduce", "combine"), ("empty",)])
def test_join_plan_refuses_a_partial_fold(given: tuple[str, ...]) -> None:
    ev, lu = two_sources()
    with pytest.raises(TypeError, match="together"):
        graphed.join_plan(graphed.join(ev, lu, on=["run"]), **{k: FOLD[k] for k in given})


def test_a_join_side_reading_no_partitioned_source_is_refused() -> None:
    ev, lu = two_sources()
    data = ak.Array({"run": np.array([1], dtype=np.int64), "w": np.array([2.0])})
    table = ev.session.source(
        "table", form=AwkwardForm(ak.Array(data.layout.to_typetracer(forget_length=True))), data=data
    )
    graphed.join_plan(graphed.join(ev, lu, on=["run"]))
    with pytest.raises(TypeError, match="exactly one partitioned source; one reads \\['table'\\]"):
        graphed.join_plan(graphed.join(ev, table, on=["run"]))


def test_a_main_reduce_stage_process_pickles_and_live_stays_out_of_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rows(values: list[object]) -> list[str]:
        return [str(v) for v in values]

    rows.__module__, rows.__qualname__ = "__main__", "m68c_rows"
    monkeypatch.setattr(sys.modules["__main__"], "m68c_rows", rows, raising=False)
    ev, lu = two_sources()
    plan = graphed.join_plan(graphed.join(ev, lu, on=["run"]), **{**FOLD, "reduce": rows})
    for stage in plan.stages:
        pickle.dumps(stage.process.resolve())
    bare = replace(plan, stages=tuple(replace(s, process=replace(s.process, live=None)) for s in plan.stages))
    assert plan.to_bytes() == bare.to_bytes()
    assert [plan.task_id(i, t) for i, s in enumerate(plan.stages) for t in s.tasks] == [
        bare.task_id(i, t) for i, s in enumerate(bare.stages) for t in s.tasks
    ]
