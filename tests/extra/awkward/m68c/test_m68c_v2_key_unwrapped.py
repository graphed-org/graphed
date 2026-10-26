"""A join plan's stage keys do not move with the by-value wrapping of stdlib-pickled plans (GW33):
its stages pickle whole through cloudpickle, so their External evaluators stay raw."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "frozen", "awkward", "m68c")
)
from m68c_services_harness import two_sources

import graphed
import graphed.aggregate
from graphed.awkward import gak
from graphed.core import PayloadDescriptor

DESC = PayloadDescriptor(
    kind="v2key",
    content_hash="sha256:v2key",
    framework="none",
    version="1",
    io_schema="x",
    preprocessing_ref=None,
)


def double(x: Any) -> Any:
    return x * 2


def _task_ids(fn: Callable[[Any], Any]) -> list[str]:
    ev, lumi = two_sources()
    x = ev.session.record_external(
        "v2key", fn, [ev.x], descriptor=DESC, form=ev.session.form(ev.x)
    )
    plan = graphed.join_plan(
        graphed.join(gak.zip({"run": ev.run, "x": x}, depth_limit=1), lumi, on=["run"])
    )
    return [plan.task_id(i, task) for i, stage in enumerate(plan.stages) for task in stage.tasks]


@pytest.mark.parametrize("fn", [double, lambda x: x * 2], ids=["module", "lambda"])
def test_join_stage_keys_match_an_unwrapping_build(
    fn: Callable[[Any], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _task_ids(fn)
    monkeypatch.setattr(graphed.aggregate, "by_value", lambda f: f)
    assert built == _task_ids(fn)
