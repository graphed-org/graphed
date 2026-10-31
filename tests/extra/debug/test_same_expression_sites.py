"""#136: an expression written on several lines keeps them all, and the arrow is on a live one."""

from __future__ import annotations

import pickle
import sys
from collections.abc import Callable
from operator import add
from pathlib import Path
from typing import Any

import awkward as ak
import pytest

import graphed
import graphed.debug as gd
from graphed import Array
from graphed.awkward import AwkwardBackend, from_parquet
from graphed.core.execution import SequentialRunner
from graphed.debug import RunReport, StageError, format_traceback, lower
from graphed.errors import SourceFrame
from graphed.execute import compile_ir
from graphed.materialization import PlanPieces
from graphed.session import Session


@pytest.fixture
def events(tmp_path: Path) -> Callable[[], tuple[Session, Any]]:
    pytest.importorskip("pyarrow")
    path = tmp_path / "ev.parquet"
    ak.to_parquet(ak.Array({"m": [[1.0], [], [3.0]]}), path)

    def read() -> tuple[Session, Any]:
        session = Session(AwkwardBackend())
        return session, from_parquet("ev", str(path), session=session)

    return read


def here() -> int:
    return sys._getframe(1).f_lineno


def via_materialize(out: Array) -> Any:
    return graphed.materialize(out)


def via_plan(out: Array) -> Any:
    plan = graphed.aggregate_plan(
        out, reduce=lambda values: list(values), combine=lambda a, b: a + b, empty=list
    )
    return SequentialRunner().run(plan)


def via_debug_run(out: Array) -> Any:
    return gd.run(out.session, out, opt_level=0)


RUNS = [via_materialize, via_plan, via_debug_run]


def double(x: Any) -> Any:
    return x * 2


def first_of(x: Any) -> Any:
    return x[:, 0]


class Piece:
    def __init__(self, output: Array) -> None:
        self.output = output

    def __graphed_pieces__(self) -> PlanPieces:
        return PlanPieces(
            outputs=(self.output,),
            reduce=lambda values: [len(values)],
            combine=add,
            empty=list,
            finalize=list,
            on_compiled=lambda _: None,
        )


def failure(run: Callable[[Array], Any], out: Array) -> StageError:
    with pytest.raises(StageError) as caught:
        run(out)
    return caught.value


@pytest.mark.parametrize("run", RUNS)
def test_the_arrow_is_on_the_line_that_feeds_the_output(events: Any, run: Any) -> None:
    _, ev = events()
    peek, peek_line = ev.m[:, 0], here()
    lead, lead_line = ev.m[:, 0], here()
    assert peek.node_id == lead.node_id
    err = failure(run, lead * 2)
    assert err.user_frame.lineno == lead_line
    assert [site.lineno for site in err.user_frame.also] == [peek_line]
    text = format_traceback(err)
    assert f"line {lead_line}," in text
    assert f"also written at {__file__}:{peek_line}:  " in text


@pytest.mark.parametrize("run", RUNS)
def test_a_dead_line_written_after_the_live_one_is_not_named(events: Any, run: Any) -> None:
    _, ev = events()
    lead, lead_line = ev.m[:, 0], here()
    _peek = ev.m[:, 0]
    err = failure(run, lead * 2)
    assert err.user_frame.lineno == lead_line
    assert err.user_frame.also == ()


@pytest.mark.parametrize("run", RUNS)
def test_a_dead_consumer_does_not_make_its_line_live(events: Any, run: Any) -> None:
    _, ev = events()
    peek = ev.m[:, 0]
    peek_twice = peek * 2
    lead, lead_line = ev.m[:, 0], here()
    lead_twice = lead * 2
    assert peek_twice.node_id == lead_twice.node_id
    assert failure(run, lead_twice).user_frame.lineno == lead_line


@pytest.mark.parametrize("run", RUNS)
def test_removing_the_dead_line_does_not_move_the_error(events: Any, run: Any) -> None:
    _, ev = events()
    lead, lead_line = ev.m[:, 0], here()
    err = failure(run, lead * 2)
    assert err.user_frame.lineno == lead_line
    assert err.user_frame.also == ()
    assert "also written" not in format_traceback(err)


def test_every_line_feeding_the_output_counts_and_the_first_written_wins(events: Any) -> None:
    _, ev = events()
    first, first_line = ev.m[:, 0], here()
    second, second_line = ev.m[:, 0], here()
    err = failure(via_materialize, first * second)
    assert err.user_frame.lineno == first_line
    assert [site.lineno for site in err.user_frame.also] == [second_line]


def test_a_rerun_line_is_one_site(events: Any) -> None:
    session, ev = events()
    for _ in range(30):
        lead = ev.m[:, 0]
    assert len(session._sites[lead.node_id]) == 1
    assert failure(via_materialize, lead * 2).user_frame.also == ()


def test_provenance_answers_for_the_proxy_it_is_asked_about(events: Any) -> None:
    session, ev = events()
    peek, peek_line = ev.m[:, 0], here()
    lead, lead_line = ev.m[:, 0], here()
    assert session.provenance(peek).lineno == peek_line
    assert session.provenance(lead).lineno == lead_line


def test_a_proxy_rebuilt_from_a_node_id_keeps_every_site(events: Any) -> None:
    session, ev = events()
    _peek, peek_line = ev.m[:, 0], here()
    lead, lead_line = ev.m[:, 0], here()
    bare = Array(session, lead.node_id)
    assert session.provenance(bare).lineno == peek_line
    sites = session._lines(lead.node_id, session._live([bare]))
    assert [site.lineno for site in sites] == [peek_line, lead_line]
    assert "also written" in format_traceback(failure(via_materialize, bare))


def test_every_site_crosses_a_process_boundary_and_a_run_report(events: Any) -> None:
    _, ev = events()
    _peek, peek_line = ev.m[:, 0], here()
    lead = ev.m[:, 0]
    err = failure(via_plan, lead * 2)
    assert [site.lineno for site in err.user_frame.also] == [peek_line]
    assert pickle.loads(pickle.dumps(err)) == err
    assert format_traceback(pickle.loads(pickle.dumps(err))) == format_traceback(err)
    report = RunReport("failed", 1, 0.0, (), err, None, {}, "")
    assert RunReport.from_json(report.to_json()).failure == err


def test_a_report_of_one_site_keeps_its_four_field_frames(events: Any) -> None:
    _, ev = events()
    err = failure(via_plan, ev.m[:, 0] * 2)
    failed = RunReport("failed", 1, 0.0, (), err, None, {}, "").to_json()["failure"]
    assert failed is not None
    assert [len(frame) for frame in failed["frames"]] == [4]


def test_a_later_recording_leaves_the_frames_of_a_compile_alone(events: Any) -> None:
    session, ev = events()
    _ = ev.m[:, 0]
    out = ev.m[:, 0] * 2
    before = compile_ir(session, out).correspondence.frames
    assert any(len(frame) == 5 for _, frame in before)
    _ = ev.m[:, 0]
    assert compile_ir(session, out).correspondence.frames == before


def test_a_plan_without_a_repeated_line_ships_four_field_frames(events: Any) -> None:
    session, ev = events()
    frames = compile_ir(session, ev.m[:, 0] * 2).correspondence.frames
    assert {len(frame) for _, frame in frames} == {4}


def test_sites_change_neither_identity_nor_bytes(events: Any) -> None:
    def compiled(dead: bool) -> tuple[bytes, int, int]:
        session, ev = events()
        if dead:
            _ = ev.m[:, 0]
        out = ev.m[:, 0] * 2
        return bytes(compile_ir(session, out).ir), out.node_id, session.node_count()

    assert compiled(True) == compiled(False)


def test_lowering_prefers_the_live_line(events: Any) -> None:
    session, ev = events()
    _peek, peek_line = ev.m[:, 0], here()
    lead, lead_line = ev.m[:, 0], here()
    op = lower(session, lead * 2, opt_level=0).op_for(lead.node_id)
    assert op.provenance.lineno == lead_line
    assert [site.lineno for site in op.provenance.also] == [peek_line]


@pytest.mark.parametrize("run", RUNS)
def test_a_shared_helper_line_keeps_its_callers_apart(events: Any, run: Any) -> None:
    _, ev = events()
    peek, peek_line = ev.m[:, 0], here()
    lead, lead_line = ev.m[:, 0], here()
    dead, out = double(peek), double(lead)
    assert dead.node_id == out.node_id
    err = failure(run, out)
    assert err.user_frame.lineno == lead_line
    assert [site.lineno for site in err.user_frame.also] == [peek_line]


def test_a_later_call_of_a_shared_line_leaves_the_frames_of_a_compile_alone(events: Any) -> None:
    session, ev = events()
    out = double(ev.m[:, 0])
    before = compile_ir(session, out).correspondence.frames
    double(ev.m[:, 0])
    assert compile_ir(session, out).correspondence.frames == before


def test_a_helper_line_reached_by_two_callers_is_one_line(events: Any) -> None:
    _, ev = events()
    first_of(ev.m)
    err = failure(via_materialize, first_of(ev.m) * 2)
    assert err.user_frame.function == "first_of"
    assert err.user_frame.also == ()


def test_a_pass_beside_a_piece_arrows_the_live_line(events: Any) -> None:
    _, ev = events()
    peek, peek_line = ev.m[:, 0], here()
    lead, lead_line = ev.m[:, 0], here()
    assert peek.node_id == lead.node_id
    with pytest.raises(StageError) as caught:
        graphed.materialize([Piece(ev.m), lead])
    assert caught.value.user_frame.lineno == lead_line
    assert [site.lineno for site in caught.value.user_frame.also] == [peek_line]


def test_a_pass_beside_a_piece_keeps_every_line_feeding_it(events: Any) -> None:
    _, ev = events()
    one, one_line = ev.m[:, 0] * 2, here()
    two, two_line = ev.m[:, 0] * 2, here()
    assert one.node_id == two.node_id
    with pytest.raises(StageError) as caught:
        graphed.materialize([Piece(ev.m), one, two])
    assert caught.value.user_frame.lineno == one_line
    assert [site.lineno for site in caught.value.user_frame.also] == [two_line]


def test_a_site_without_source_text_prints_no_dangling_separator() -> None:
    frame = SourceFrame("a.py", 1, "f", "y", (SourceFrame("a.py", 2),))
    err = StageError(
        op="x",
        frames=(frame,),
        input_forms=(),
        partition="p",
        cause_type="T",
        cause_message="m",
        opt_level=0,
    )
    assert "(same expression also written at a.py:2)" in format_traceback(err)
