"""A task's key and a saved plan's bytes do not depend on how a behavior was built."""

from __future__ import annotations

import json
import logging
import numbers
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import awkward as ak
import key_order_helpers as h
import numpy as np
import pytest

from graphed.awkward.backend import _BehaviorByRef, _canonical, shipped_behavior
from graphed.checkpoint import Store, resumable, run_resumable, run_shuffle_resumable
from graphed.checkpoint.resume import _key_digest
from graphed.core import SequentialRunner

HERE = Path(__file__).parent
LOGGER = "graphed.checkpoint"


def leg(variant: str, root: Path, seed: str) -> dict[str, Any]:
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


@pytest.mark.parametrize("variant", ["by_value", "by_ref", "write_dict", "write_ref", "shuffle"])
def test_a_new_process_reuses_what_another_hash_seed_stored(tmp_path: Path, variant: str) -> None:
    first = leg(variant, tmp_path, "1")
    second = leg(variant, tmp_path, "2")
    assert first["order"] != second["order"]
    assert first["reused"] == 0
    assert second["reused"] == second["tasks"] > 0


def test_a_shuffle_plan_has_one_set_of_bytes_across_hash_seeds(tmp_path: Path) -> None:
    assert leg("shuffle", tmp_path, "1")["bytes"] == leg("shuffle", tmp_path, "2")["bytes"]


def test_order_does_not_move_a_key_nor_the_shipped_bytes() -> None:
    items = list(h.BEHAVIOR.items())
    shuffled = dict(reversed(items))
    assert list(shuffled) != list(h.BEHAVIOR)
    assert list(_canonical(shuffled)) == list(_canonical(h.BEHAVIOR))
    assert _key_digest(_canonical(shuffled)) == _key_digest(_canonical(h.BEHAVIOR))
    assert _key_digest(_BehaviorByRef(shuffled, "m:b")) == _key_digest(
        _BehaviorByRef(h.BEHAVIOR, "m:b")
    )


def test_a_key_holding_a_type_keeps_its_relative_order() -> None:
    def real(a: Any, b: Any) -> str:
        return "real"

    def number(a: Any, b: Any) -> str:
        return "number"

    first = (np.add, "P", numbers.Real)
    second = (np.add, "P", numbers.Number)
    named = {(np.add, n, n): h.overload for n in "ba"}
    one = _canonical({first: real, **named, second: number})
    other = _canonical({second: number, **named, first: real})
    assert [k for k in one if k in (first, second)] == [first, second]
    assert list(one) != list(other)
    assert _key_digest(one) != _key_digest(other)
    signature = (np.add, "P", np.float64)
    assert ak._behavior.find_ufunc(one, signature) is real
    assert ak._behavior.find_ufunc(other, signature) is number


def test_a_shipped_behavior_keeps_every_entry() -> None:
    shipped = shipped_behavior(dict(reversed(list(h.BEHAVIOR.items()))))
    assert dict(shipped) == h.BEHAVIOR


def run(plan: Any, root: Path) -> int:
    resumed: Any = resumable(plan, str(root))
    SequentialRunner().run(resumed)
    reused: int = resumed.process.reused
    return reused


def warnings_in(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_a_total_miss_on_a_store_that_holds_these_partitions_warns_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, LOGGER)
    partitions = h.parts(0, 10, 20)
    assert run(h.runtime(h.start, partitions), tmp_path) == 0
    assert warnings_in(caplog) == []
    assert run(h.runtime(h.stop, partitions), tmp_path) == 0
    (message,) = warnings_in(caplog)
    assert "0 of 3 tasks reused" in message
    assert f"{tmp_path} holds 3 results" in message


def test_no_warning_on_a_reuse_or_on_other_partitions(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, LOGGER)
    run(h.runtime(h.start, h.parts(0, 10)), tmp_path)
    assert run(h.runtime(h.start, h.parts(0, 10, 20)), tmp_path) == 2
    assert run(h.runtime(h.stop, h.parts(50)), tmp_path) == 0
    assert warnings_in(caplog) == []


@pytest.mark.parametrize(
    ("build", "resume"), [(h.durable, run_resumable), (h.staged, run_shuffle_resumable)]
)
def test_the_serial_runners_warn_on_a_total_miss(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, build: Any, resume: Any
) -> None:
    caplog.set_level(logging.WARNING, LOGGER)
    partitions = h.parts(0, 10)
    resume(build(b"one", partitions), Store(tmp_path))
    assert resume(build(b"one", partitions), Store(tmp_path)).report.skipped == 2
    assert warnings_in(caplog) == []
    assert resume(build(b"two", partitions), Store(tmp_path)).report.skipped == 0
    (message,) = warnings_in(caplog)
    assert "0 of 2 tasks reused" in message


def test_a_durable_plan_warns_through_resumable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, LOGGER)
    for build in (h.durable, h.staged):
        store = tmp_path / build.__name__
        resumable(build(b"one", h.parts(0)), str(store))
        Store(store).record_done("other", "f.root@0:10", Store(store).put(b"x"))
        resumable(build(b"two", h.parts(0)), str(store))
    assert len(warnings_in(caplog)) == 2
