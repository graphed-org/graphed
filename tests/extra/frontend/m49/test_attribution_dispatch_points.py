"""§8.2(iii)'s attribution hook at the TOP-LEVEL dispatch point.

The frozen m49 debug anchors fail inside a fused stage, so they witness the inline member loop and
its `member_index`. This is the other dispatch point: a boundary reduction never fuses, so its
failure must arrive keyed `(node_id, None)` — a hook handed a member index there would mis-index the
label channel, and returning nothing must still leave the original exception alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
from backends import ListBackend, from_list

import graphed.core
from graphed import Session
from graphed.execute import Key, compile_ir, evaluate_ir


class Boom(RuntimeError):
    pass


class PoisonedSum(ListBackend):
    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        if op == "sum":
            raise Boom("the reduction failed")
        return super().eval_stage(op, inputs, params)


def _compiled() -> tuple[object, int]:
    session = Session(PoisonedSum())
    x = from_list(session, "x", [1.0, 2.0])
    compiled = compile_ir(session, (x * x).reduce("sum"))
    (output,) = graphed.core.GraphStore.deserialize(bytes(compiled.ir)).outputs()
    return compiled, output


def test_a_failing_boundary_reduction_is_keyed_with_no_member_index() -> None:
    compiled, output = _compiled()
    seen: list[tuple[Key, str]] = []

    def attribute(key: Key, op: str, ins: list[object], exc: BaseException) -> BaseException:
        seen.append((key, op))
        return ValueError(f"attributed {key}")

    with pytest.raises(ValueError, match=rf"attributed \({output}, None\)"):
        evaluate_ir(compiled, PoisonedSum(), {"x": [1.0, 2.0]}, on_failure=attribute)
    assert seen == [((output, None), "sum")]  # the map stage ran fine; only the reduction failed


def test_a_hook_that_attributes_nothing_leaves_the_original_exception() -> None:
    compiled, _output = _compiled()
    with pytest.raises(Boom):
        evaluate_ir(compiled, PoisonedSum(), {"x": [1.0, 2.0]}, on_failure=lambda *_: None)
