"""``graphed.debug.replay(plan, task)`` with no outputs, and ``graphed.debug.trace``."""

from __future__ import annotations

import pickle
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import awkward as ak
import cloudpickle
import pytest

import graphed
import graphed.debug as gd
from graphed import Session
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward, gak
from graphed.core import Partition
from graphed.core.execution import SequentialRunner
from graphed.errors import StageError
from graphed.preserve import build_bundle, reproduce

gh = pytest.importorskip("graphed_histogram")

DATA = ak.Array({"pt": [[1.0, 2.0], [], [3.0], [4.0, 5.0]], "met": [1.0, 2.0, 3.0, 4.0]})


class _File:
    """One partitioned file, recording the rows of each read."""

    def __init__(self) -> None:
        self.reads: list[int] = []

    def partitions(self, steps_per_file: int) -> tuple[Partition, ...]:
        return (Partition("f", "t", 0, len(DATA)),)

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> Any:
        part = partition.resolve(len(DATA))
        chunk = DATA[part.entry_start : part.entry_stop]
        self.reads.append(len(chunk))
        return chunk

    def __call__(self) -> ak.Array:
        return DATA


def _events() -> tuple[Any, _File]:
    file = _File()
    form = AwkwardForm(ak.Array(DATA.layout.to_typetracer(forget_length=True)))
    session = Session(AwkwardBackend())
    return session.source("ev", form=form, data=file), file


def _plan(*outputs: Any, **options: Any) -> Any:
    return graphed.aggregate_plan(
        *outputs,
        reduce=list,
        combine=lambda a, b: a + b,
        empty=list,
        **options,
    )


def _lead() -> tuple[Any, Any]:
    ev, _ = _events()
    kept = ev[ev.met > 1.5]  # the line each step names
    return kept, gak.num(kept.pt, axis=1)


def test_replay_needs_no_outputs_and_gives_the_runs_partial() -> None:
    _, counts = _lead()
    plan = _plan(counts)
    assert ak.to_list(gd.replay(plan, 0).value[0]) == [0, 1, 2]


def test_replay_steps_every_operation_of_a_fused_stage_at_its_line() -> None:
    _, counts = _lead()
    plan = _plan(counts)
    steps = list(gd.replay(plan, 0).steps())
    assert [s.node.op for s in steps] == ["ev", "field", "gt", "getitem", "field", "ak.num"]
    assert [s.node.op for s in gd.replay(plan, 0, counts).steps()] == [s.node.op for s in steps]
    getitem = steps[3]
    assert "ev[ev.met > 1.5]" in getitem.node.provenance.source
    assert getitem.node.provenance.filename == __file__
    assert len(getitem.value) == 3


def test_replay_at_opt_level_zero_needs_no_outputs() -> None:
    _, counts = _lead()
    plan = _plan(counts, opt_level=0)
    assert ak.to_list(gd.replay(plan, 0).value[0]) == [0, 1, 2]


def test_replay_of_two_outputs_keeps_their_order() -> None:
    ev, _ = _events()
    first, second = gak.sum(ev.pt, axis=1), gak.num(ev.pt, axis=1)
    values = gd.replay(_plan(first, second), 0).value
    assert ak.to_list(values[0]) == [3.0, 0.0, 3.0, 9.0]
    assert ak.to_list(values[1]) == [2, 0, 1, 2]


def test_replay_diffs_against_the_same_input() -> None:
    _, counts = _lead()
    diff = gd.replay(_plan(counts), 0).diff()
    assert diff.equal and diff.reference == "re-evaluated"


def test_replay_works_from_a_plan_that_crossed_a_process_boundary() -> None:
    _, counts = _lead()
    plan = pickle.loads(cloudpickle.dumps(_plan(counts)))
    assert ak.to_list(gd.replay(plan, 0).value[0]) == [0, 1, 2]


def test_a_failing_step_raises_at_its_line() -> None:
    ev, _ = _events()
    broken = ev.map(_explode)
    with pytest.raises(StageError) as raised:
        list(gd.replay(_plan(broken), 0).steps())
    assert raised.value.frames[0].filename == __file__
    assert (raised.value.op, raised.value.opt_level) == ("external", 0)


def test_rows_that_do_not_line_up_fail_in_replay_as_in_the_run() -> None:
    ev, _ = _events()
    unaligned = ev.met + ev[ev.met > 3.5].met
    plan = _plan(unaligned)
    with pytest.raises(StageError, match="different row sets"):
        SequentialRunner().run(plan)
    for given in ((), (unaligned,)):
        with pytest.raises(StageError, match="different row sets") as raised:
            list(gd.replay(plan, 0, *given).steps())
        assert raised.value.op == "add"
        assert raised.value.frames[0].filename == __file__
    chunk = plan.tasks[0].partition
    with pytest.raises(StageError, match="different row sets") as raised:
        gd.run(unaligned.session, unaligned, partition=chunk)
    assert raised.value.op == "add"


def test_a_reduction_needs_no_aligned_rows_in_replay() -> None:
    ev, _ = _events()
    joined = gak.concatenate([ev.met, ev[ev.met > 3.5].met])
    want = [1.0, 2.0, 3.0, 4.0, 4.0]
    plan = _plan(joined)
    assert ak.to_list(gd.replay(plan, 0).value[0]) == want
    assert ak.to_list(gd.run(joined.session, joined, partition=plan.tasks[0].partition)) == want


def test_a_bundle_reproduces_an_exchange_and_a_join(tmp_path: Path) -> None:
    data = {
        "ev": ak.Array({"k": [1, 2, 3], "a": [1.0, 2.0, 3.0]}),
        "ot": ak.Array({"k": [2, 3], "b": [5.0, 6.0]}),
    }
    session = Session(AwkwardBackend())
    ev = from_awkward("ev", data["ev"], session=session)
    ot = from_awkward("ot", data["ot"], session=session)
    for name, out in (("x", ev.a.repartition(n=2)), ("j", gak.join(ev, ot, on=["k"]))):
        bundle = build_bundle(tmp_path / name, session=session, value=out, datasets=data)
        assert ak.to_list(reproduce(bundle)) == ak.to_list(graphed.materialize(out))


def test_a_run_of_the_whole_source_takes_no_row_check() -> None:
    ev, _ = _events()
    assert ak.to_list(gd.run(ev.session, ev.met + ev[ev.met > 3.5].met)) == [5.0, 6.0, 7.0, 8.0]


def test_an_exchange_and_a_join_replay_as_the_run_ran_them() -> None:
    ev, _ = _events()
    kept = ev[ev.met > 3.5]
    left, right = gak.zip({"k": ev.met, "a": ev.met}), gak.zip({"k": kept.met, "b": kept.met})
    for output in (ev.met.repartition(n=2), gak.join(left, right, on=["k"])):
        plan = _plan(output)
        ran = ak.to_list(SequentialRunner().run(plan).value[0])
        assert ran
        for given in ((), (output,)):
            assert ak.to_list(gd.replay(plan, 0, *given).value[0]) == ran


def _explode(chunk: Any) -> Any:
    raise ValueError("boom")


def test_other_outputs_name_the_fix() -> None:
    kept, counts = _lead()
    with pytest.raises(ValueError, match="none to take them from the plan"):
        gd.replay(_plan(counts), 0, kept)


def test_a_histogram_as_an_output_names_the_fix() -> None:
    ev, _ = _events()
    h = gh.histogram(gak.flatten(ev.pt), bins=4, range=(0, 8))
    with pytest.raises(TypeError, match=r"gd\.replay\(plan, task\)"):
        gd.replay(gh.plan({"h": h}), 0, h)


def test_replay_of_histogram_plans_needs_no_outputs() -> None:
    ev, _ = _events()
    h = gh.histogram(gak.flatten(ev.pt), bins=4, range=(0, 8))
    assert gd.replay(gh.plan({"h": h}), 0).value["h"].sum() == 5
    assert gd.replay(h.plan(), 0).value.sum() == 5


def test_replay_adds_nothing_to_a_plans_pickle() -> None:
    _, counts = _lead()
    plan = _plan(counts)
    # the fixture's reader logs each read
    size = len(pickle.dumps(replace(plan.process, reader=None)))
    assert gd.replay(plan, 0).diff().equal
    assert len(pickle.dumps(replace(plan.process, reader=None))) == size


@pytest.mark.parametrize(
    "reduced",
    [lambda x: gak.sum(x, axis=None), lambda x: gak.max(x, axis=None), lambda x: gak.mean(x)],
    ids=["sum", "max", "mean"],
)
def test_replay_lowers_alike_with_and_without_outputs(reduced: Any) -> None:
    ev, _ = _events()
    total = reduced(ev.met.map(_double))
    plan = _plan(total)
    from_plan, as_written = (
        [(s.node.op, s.node.kind, s.node.boundary) for s in gd.replay(plan, 0, *given).steps()]
        for given in ((), (total,))
    )
    assert ("external", "external", True) in from_plan
    assert from_plan == as_written


def test_a_per_event_sum_is_a_boundary_with_and_without_outputs() -> None:
    ev, _ = _events()
    totals = gak.sum(ev.pt, axis=1)
    plan = _plan(totals)
    from_plan, as_written = (
        [(s.node.op, s.node.boundary) for s in gd.replay(plan, 0, *given).steps()]
        for given in ((), (totals,))
    )
    assert ("ak.sum", True) in from_plan
    assert from_plan == as_written


def _double(chunk: Any) -> Any:
    return chunk * 2


def test_trace_prints_a_row_per_line_and_marks_the_emptying_step(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ev, _ = _events()
    empty = ev[ev.met > 100.0]
    counts = gak.num(empty.pt, axis=1)
    gd.trace(counts, entries=4)
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert len(lines) == 3
    assert re.search(
        r"getitem +4 -> 0 +ev\[ev\.met > 100\.0\] +<-- removed every event$", lines[1]
    )
    assert "removed every event" not in lines[0] + lines[2]
    assert re.search(r"ak\.num +0 -> 0 ", lines[2])


def test_trace_marks_a_cut_chained_on_one_line(capsys: pytest.CaptureFixture[str]) -> None:
    ev, _ = _events()
    gd.trace(gak.num(ev[ev.met > 100.0].pt, axis=1), entries=4)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert re.search(r"getitem +4 -> 0 .*<-- removed every event$", lines[1])


def _cut(ev: Any, low: float) -> Any:
    return ev[ev.met > low]


def test_trace_marks_a_helper_line_at_the_call_that_emptied(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ev, _ = _events()
    gd.trace(_cut(_cut(ev, 100.0), 0.5), entries=4)
    assert re.search(r"getitem +4 -> 0 .*<-- removed every event$", capsys.readouterr().out, re.M)


def test_trace_reads_only_the_sample_it_was_asked_for(capsys: pytest.CaptureFixture[str]) -> None:
    ev, file = _events()
    gd.trace(ev.met, entries=2)
    assert re.search(r"field +2 -> 2", capsys.readouterr().out)
    assert file.reads == [2]


def test_trace_shows_a_scalar_as_a_dash(capsys: pytest.CaptureFixture[str]) -> None:
    ev, _ = _events()
    gd.trace(gak.sum(ev.met, axis=None), entries=4)
    assert re.search(r"ak\.sum +4 -> -", capsys.readouterr().out)


def test_trace_marks_nothing_when_rows_survive(capsys: pytest.CaptureFixture[str]) -> None:
    ev, _ = _events()
    gd.trace(ev[ev.met > 1.5], entries=4)
    assert "removed every event" not in capsys.readouterr().out


def test_trace_refuses_a_varied_array_and_a_bad_count() -> None:
    ev, _ = _events()
    with pytest.raises(graphed.GraphedError, match=r"graphed\.debug\.trace"):
        gd.trace(graphed.vary(ev.met, "sf", up=ev.met * 2, down=ev.met / 2), entries=4)
    with pytest.raises(ValueError, match=r"trace\(entries=\) is a positive int"):
        gd.trace(ev.met, entries=0)
    h = gh.histogram(gak.flatten(ev.pt), bins=4, range=(0, 8))
    with pytest.raises(TypeError, match="not Histogram"):
        gd.trace(h, entries=4)
