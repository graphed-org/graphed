"""A plan's one-line repr, and ``Plan.save``/``Plan.load``."""

from __future__ import annotations

import dataclasses
import operator
import pickle
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from graphed.core import Partition
from graphed.core.execution import Plan, SequentialRunner, Task, capped


def _process(partition: Partition, resources: Any) -> int:
    return partition.entry_stop - partition.entry_start


def _plan(*uris: str, tree: str = "Events", **kwargs: Any) -> Plan[int]:
    tasks = tuple(Task(i, Partition(uri, tree, 0, 10)) for i, uri in enumerate(uris))
    return Plan(process=_process, combine=int.__add__, empty=int, tasks=tasks, **kwargs)


def test_a_hand_built_plan_names_its_tasks_file_and_process() -> None:
    plan = _plan("root://site//store/a.root", "root://site//store/a.root")
    assert repr(plan) == "Plan: 2 tasks over 1 file (a.root:Events), runs _process"


def test_a_file_with_no_tree_shows_no_colon() -> None:
    assert repr(_plan("/data/a.parquet", tree="")).startswith(
        "Plan: 1 task over 1 file (a.parquet), "
    )


def test_several_files_are_counted() -> None:
    assert repr(_plan("a.root", "b.root")) == "Plan: 2 tasks over 2 files, runs _process"


def test_two_trees_of_one_file_name_neither() -> None:
    plan = dataclasses.replace(
        _plan("a.root"),
        tasks=(Task(0, Partition("a.root", "Events")), Task(1, Partition("a.root", "Runs"))),
    )
    assert repr(plan) == "Plan: 2 tasks over 1 file, runs _process"


def test_a_windows_path_shows_its_file_name() -> None:
    assert repr(_plan(r"C:\data\a.root")).startswith("Plan: 1 task over 1 file (a.root:Events)")


def test_a_plan_with_no_tasks_and_a_callable_object() -> None:
    plan = Plan(process=dataclasses.make_dataclass("Proc", [])(), combine=max, empty=int)
    assert repr(plan) == "Plan: 0 tasks, runs Proc"


def test_an_adaptive_plan_says_so() -> None:
    assert ", adaptive, " in repr(_plan("a.root", next_tasks=lambda ctx: None))


def test_capped_counts_what_it_drops() -> None:
    assert capped(["a", "b"]) == "a, b"
    assert capped(list("abcdefgh")) == "a, b, c, d, e, f, +2 more"


def test_replace_still_works() -> None:
    plan = dataclasses.replace(_plan("a.root"), finalize=abs)
    assert plan.finalize is abs


def test_load_runs_as_the_saved_plan(tmp_path: Path) -> None:
    path = tmp_path / "p.plan"
    plan = _plan("a.root", "b.root")
    plan.save(path)
    assert (
        SequentialRunner().run(Plan.load(path)).value == SequentialRunner().run(plan).value == 20
    )


def test_load_in_a_fresh_isolated_interpreter(tmp_path: Path) -> None:
    path = tmp_path / "p.plan"
    tasks = (Task(0, Partition("a.root")),)
    Plan(process=operator.is_not, combine=operator.or_, empty=bool, tasks=tasks).save(path)
    code = f"""
from graphed.core.execution import Plan, SequentialRunner
print(SequentialRunner().run(Plan.load({str(path)!r})).value)
"""
    done = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "True"


def test_a_process_a_script_defines_loads_in_a_later_interpreter(tmp_path: Path) -> None:
    path = tmp_path / "p.plan"
    script = tmp_path / "analysis.py"
    script.write_text(
        f"""
from graphed.core import Partition
from graphed.core.execution import Plan, Task
def process(partition, resources):
    return 7
Plan(process, int.__add__, int, (Task(0, Partition("a.root")),)).save({str(path)!r})
"""
    )
    subprocess.run([sys.executable, "-I", str(script)], check=True)
    code = f"""
from graphed.core.execution import Plan, SequentialRunner
print(SequentialRunner().run(Plan.load({str(path)!r})).value)
"""
    done = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "7"


def test_a_save_that_cannot_pickle_keeps_the_plan_already_there(tmp_path: Path) -> None:
    path = tmp_path / "p.plan"
    _plan("a.root").save(path)
    saved = path.read_bytes()
    with pytest.raises(TypeError, match="lock"):
        _plan("a.root", services=(threading.Lock(),)).save(path)
    assert path.read_bytes() == saved


def test_a_header_that_does_not_decode_is_refused_by_name(tmp_path: Path) -> None:
    path = tmp_path / "p.plan"
    path.write_bytes(b"graphed-plan \xff\n")
    with pytest.raises(ValueError, match="saved by graphed"):
        Plan.load(path)


def test_another_versions_plan_is_refused_by_name(tmp_path: Path) -> None:
    path = tmp_path / "p.plan"
    path.write_bytes(b"graphed-plan 0.0.1\n" + b"\x80 not a pickle")
    with pytest.raises(ValueError, match=r"saved by graphed 0\.0\.1 .*rebuild it"):
        Plan.load(path)


def test_another_file_is_refused_before_unpickling(tmp_path: Path) -> None:
    path = tmp_path / "p.pkl"
    path.write_bytes(pickle.dumps(_plan("a.root")))
    with pytest.raises(ValueError, match=r"not a file Plan\.save wrote"):
        Plan.load(path)
