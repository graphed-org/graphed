"""A task's key and a saved plan's bytes do not depend on how a behavior was built."""

from __future__ import annotations

import json
import logging
import numbers
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import awkward as ak
import key_order_helpers as h
import numpy as np
import pytest

from graphed.awkward.backend import _BehaviorByRef, _canonical, shipped_behavior
from graphed.awkward.io import _keyed_behavior
from graphed.checkpoint import Store, resumable, run_resumable, run_shuffle_resumable
from graphed.checkpoint.resume import _key_digest
from graphed.checkpoint.runner import _SimulatedInterrupt
from graphed.core import SequentialRunner

HERE = Path(__file__).parent
LOGGER = "graphed.checkpoint"
BOTH_RUNNERS = pytest.mark.parametrize(
    ("build", "resume"), [(h.durable, run_resumable), (h.staged, run_shuffle_resumable)]
)
FIRST = (np.add, "P", numbers.Real)
SECOND = (np.add, "P", numbers.Number)
SIGNATURE = (np.add, "P", np.float64)


def real(a: Any, b: Any) -> str:
    return "real"


def number(a: Any, b: Any) -> str:
    return "number"


Warned = Callable[[], list[str]]


def leg(variant: str, root: Path, seed: str) -> dict[str, Any]:
    pytest.importorskip("pyarrow")
    env = {**os.environ, "PYTHONHASHSEED": seed}
    env["PYTHONPATH"] = os.pathsep.join([str(HERE), env.get("PYTHONPATH", "")])
    args = [str(root / "store"), str(root), str(root / "out")]
    done = subprocess.run(
        [sys.executable, str(HERE / "key_order_leg.py"), variant, *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    report: dict[str, Any] = json.loads(done.stdout.strip().splitlines()[-1])
    return report


@pytest.mark.parametrize(
    "variant", ["by_value", "by_ref", "by_factory", "write_dict", "write_ref", "shuffle"]
)
def test_a_new_process_reuses_what_another_hash_seed_stored(tmp_path: Path, variant: str) -> None:
    first = leg(variant, tmp_path, "1")
    second = leg(variant, tmp_path, "2")
    assert first["order"] != second["order"]
    assert first["reused"] == 0
    assert second["reused"] == second["tasks"] > 0
    assert first.get("bytes") == second.get("bytes")


def test_order_does_not_move_a_key_nor_the_shipped_bytes() -> None:
    shuffled = dict(reversed(h.BEHAVIOR.items()))
    assert list(shuffled) != list(h.BEHAVIOR)
    assert list(_canonical(shuffled)) == list(_canonical(h.BEHAVIOR))
    assert _key_digest(_canonical(shuffled)) == _key_digest(_canonical(h.BEHAVIOR))
    reordered = _BehaviorByRef(shuffled, "m:b")
    assert _key_digest(reordered) == _key_digest(_BehaviorByRef(h.BEHAVIOR, "m:b"))
    assert list(_keyed_behavior("key_order_helpers:BEHAVIOR")) == list(_canonical(h.BEHAVIOR))
    assert _keyed_behavior("key_order_helpers:NOTHING") is None


def test_a_key_does_not_move_with_which_of_its_equal_strings_are_one_object() -> None:
    name = "".join(["P", "Q"])
    distinct = "".join(["P", "Q"])
    assert name is not distinct
    assert list(_canonical({(np.add, name, name): 1, (np.add, name, "PA"): 2})) == list(
        _canonical({(np.add, name, distinct): 1, (np.add, name, "PA"): 2})
    )


def test_keys_differing_only_in_a_type_keep_their_relative_order() -> None:
    named = {(np.add, n, n): h.overload for n in "ba"}
    one = _canonical({FIRST: real, **named, SECOND: number})
    other = _canonical({SECOND: number, **named, FIRST: real})
    assert [k for k in one if k in (FIRST, SECOND)] == [FIRST, SECOND]
    assert list(one) != list(other)
    assert _key_digest(one) != _key_digest(other)
    assert ak._behavior.find_ufunc(one, SIGNATURE) is real
    assert ak._behavior.find_ufunc(other, SIGNATURE) is number


def test_a_shipped_behavior_keeps_the_order_of_an_equal_one_shipped_before() -> None:
    one = shipped_behavior({FIRST: real, SECOND: number})
    other = shipped_behavior({SECOND: number, FIRST: real})
    assert ak._behavior.find_ufunc(one, SIGNATURE).__name__ == "real"
    assert ak._behavior.find_ufunc(other, SIGNATURE).__name__ == "number"


def test_a_shipped_behavior_keeps_every_entry() -> None:
    shipped = shipped_behavior(dict(reversed(h.BEHAVIOR.items())))
    assert dict(shipped) == h.BEHAVIOR


def run(plan: Any, root: Path) -> int:
    resumed: Any = resumable(plan, str(root))
    SequentialRunner().run(resumed)
    reused: int = resumed.process.reused
    return reused


@pytest.fixture
def warned(caplog: pytest.LogCaptureFixture) -> Warned:
    caplog.set_level(logging.WARNING, LOGGER)
    return lambda: [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_a_total_miss_on_a_store_that_holds_these_partitions_warns_once(
    tmp_path: Path, warned: Warned
) -> None:
    partitions = h.parts(0, 10, 20)
    assert run(h.runtime(h.start, partitions), tmp_path) == 0
    assert warned() == []
    assert run(h.runtime(h.stop, partitions), tmp_path) == 0
    (message,) = warned()
    assert "0 of 3 tasks reused" in message
    assert f"{tmp_path} holds 3 results" in message


def test_no_warning_on_a_reuse_or_on_other_partitions(tmp_path: Path, warned: Warned) -> None:
    run(h.runtime(h.start, h.parts(0, 10)), tmp_path)
    assert run(h.runtime(h.start, h.parts(0, 10, 20)), tmp_path) == 2
    assert run(h.runtime(h.stop, h.parts(50)), tmp_path) == 0
    assert warned() == []


@BOTH_RUNNERS
def test_the_serial_runners_warn_on_a_total_miss(
    tmp_path: Path, warned: Warned, build: Any, resume: Any
) -> None:
    partitions = h.parts(0, 10)
    resume(build(b"one", partitions), Store(tmp_path))
    assert resume(build(b"one", partitions), Store(tmp_path)).report.skipped == 2
    assert warned() == []
    assert resume(build(b"two", partitions), Store(tmp_path)).report.skipped == 0
    (message,) = warned()
    assert "0 of 2 tasks reused" in message


@BOTH_RUNNERS
def test_a_total_miss_warns_before_the_run_is_interrupted(
    tmp_path: Path, warned: Warned, build: Any, resume: Any
) -> None:
    partitions = h.parts(0, 10)
    resume(build(b"one", partitions), Store(tmp_path))
    with pytest.raises(_SimulatedInterrupt):
        resume(build(b"two", partitions), Store(tmp_path), _kill_after=1)
    assert len(warned()) == 1


def test_a_shuffle_warns_for_its_input_partitions_and_not_for_its_destinations(
    tmp_path: Path, warned: Warned
) -> None:
    run_shuffle_resumable(h.shuffled(b"one", h.parts(0)), Store(tmp_path))
    run_shuffle_resumable(h.shuffled(b"two", h.parts(50)), Store(tmp_path))
    assert warned() == []
    run_shuffle_resumable(h.shuffled(b"three", h.parts(0)), Store(tmp_path))
    (message,) = warned()
    assert "0 of 2 tasks reused" in message
    assert "holds 1 results" in message


@pytest.mark.parametrize("build", [h.durable, h.staged])
def test_a_durable_plan_warns_through_resumable(
    tmp_path: Path, warned: Warned, build: Any
) -> None:
    resumable(build(b"one", h.parts(0)), str(tmp_path))
    Store(tmp_path).record_done("other", "f.root@0:10", Store(tmp_path).put(b"x"))
    resumable(build(b"two", h.parts(0)), str(tmp_path))
    assert len(warned()) == 1
