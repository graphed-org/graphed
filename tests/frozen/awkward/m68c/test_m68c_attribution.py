"""m68c §3.3 (attribution): an External that raises in a V2 map stage surfaces as ``StageError`` at
the user's recording line (M6), as in a V1 plan."""

from __future__ import annotations

import sys
from typing import Any

import pytest
from m68c_services_harness import two_sources

import graphed
from graphed.awkward import gak
from graphed.core.execution import SequentialRunner
from graphed.debug.errors import StageError
from graphed.preserve import ExternalPlugin, record_external, sha256_bytes


class SFUnreadable(RuntimeError):
    pass


def _boom(resource: Any, params: Any, inputs: list[Any]) -> Any:
    raise SFUnreadable("the scale-factor table is unreadable")


def _samples() -> list[bytes]:
    return [b"boom"]


BOOM = ExternalPlugin(kind="m68c_boom", content_hash=sha256_bytes, evaluate=_boom, samples=_samples)


def test_a_raising_external_before_a_join_points_at_its_recording_line() -> None:
    ev, lu = two_sources()
    line = sys._getframe().f_lineno + 1
    bad = record_external(ev.session, BOOM, b"boom", [ev.x])
    j = graphed.join(gak.zip({"run": ev.run, "x": bad}, depth_limit=1), lu, on=["run"])
    with pytest.raises(StageError) as info:
        SequentialRunner().run(graphed.join_plan(j, steps_per_file=2))
    frame = info.value.user_frame
    assert (frame.filename, frame.lineno) == (__file__, line)
    assert info.value.cause_type == "SFUnreadable"
