"""m68c edges the frozen suite does not take: the executors' payload slicer, a partial fold set, a
join side that reads no partitioned source, and a plan with a lambda or ``__main__`` reduce."""

from __future__ import annotations

import inspect
import os
import pickle
import subprocess
import sys
from typing import Any

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "frozen", "awkward", "m68c"))
from m68c_services_harness import FOLD, two_sources

import graphed
from graphed.awkward import AwkwardForm
from graphed.core import SequentialRunner
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


def _main_rows(values: list[Any]) -> list[str]:
    return [str(v) for v in values]


@pytest.mark.parametrize("kind", ["lambda", "main"])
def test_a_plan_with_an_unimportable_reduce_runs_in_a_fresh_process(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    reduce: Any = lambda values: [str(v) for v in values]  # noqa: E731
    if kind == "main":
        reduce = _main_rows
        monkeypatch.setattr(reduce, "__module__", "__main__")
        monkeypatch.setattr(reduce, "__qualname__", "m68c_rows")
        monkeypatch.setattr(sys.modules["__main__"], "m68c_rows", reduce, raising=False)
    ev, lu = two_sources()
    plan = graphed.join_plan(graphed.join(ev, lu, on=["run"]), **{**FOLD, "reduce": reduce})
    want = SequentialRunner().run(plan).value
    code = (
        "import pickle, sys; from graphed.core import SequentialRunner; "
        "sys.stdout.buffer.write(pickle.dumps(SequentialRunner().run(pickle.load(sys.stdin.buffer)).value))"
    )
    env = {**os.environ, "PYTHONPATH": os.path.dirname(inspect.getfile(two_sources))}
    done = subprocess.run(
        [sys.executable, "-c", code], input=pickle.dumps(plan), capture_output=True, env=env
    )
    assert done.returncode == 0, done.stderr.decode()[-400:]
    assert want and pickle.loads(done.stdout) == want
