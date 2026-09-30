"""m68c §3.3: ``evaluate_ir`` dispatches ``exchange`` and ``join`` nodes through ``eval_stage``, so the
full IR of a join graph or a repartition graph evaluates to what ``Session.materialize`` gives."""

from __future__ import annotations

from typing import Any

import awkward as ak
import pytest
from m68c_services_harness import events_files, lumi_files, rows, two_sources

import graphed
from graphed.awkward import AwkwardBackend, gak
from graphed.execute import compile_ir, evaluate_ir


def _join_then_op(ev: Any, lu: Any) -> Any:
    j = graphed.join(ev, lu, on=["run"], how="outer")
    return gak.with_field(j, j.run * 10, "run10")


def _grouped_left(ev: Any, lu: Any) -> Any:
    return gak.join(ev, lu, on=["run"], how="left", grouped=True)


def _repartition_then_op(ev: Any, lu: Any) -> Any:
    r = graphed.repartition(lu, by="run")
    return gak.with_field(r, r.w * 2, "w2")


@pytest.mark.parametrize("build", [_join_then_op, _grouped_left, _repartition_then_op])
def test_evaluate_ir_over_the_full_ir_equals_materialize(build: Any) -> None:
    ev, lu = two_sources()
    out = build(ev, lu)
    sources = {"events": ak.concatenate(events_files()), "lumi": ak.concatenate(lumi_files())}
    (value,) = evaluate_ir(compile_ir(out.session, out), AwkwardBackend(), sources)
    expected = out.session.materialize(out)
    assert str(ak.Array(value).type) == str(expected.type)
    assert rows(value) == rows(expected)
