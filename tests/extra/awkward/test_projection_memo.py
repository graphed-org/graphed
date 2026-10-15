"""Projection replays a sub-DAG shared by several outputs once, not once per output (GW31)."""

from __future__ import annotations

from typing import Any

import awkward as ak
import pytest

from graphed import BufferNeed, Session
from graphed.awkward import AwkwardBackend, from_awkward, gak, project_buffers_many
from graphed.systematics import by_label
from graphed.systematics.by_label import cone

DATA = {"m": [1.0, 2.0], "n": [1, 2], "Jet": [[{"pt": 1.0, "eta": 0.1}], []]}


def _outputs(n_outputs: int) -> tuple[Session, Any, list[Any]]:
    session = Session(AwkwardBackend())
    ev = from_awkward(session, "ev", ak.Array(DATA))
    shared = gak.with_field(ev.Jet, ev.Jet.pt * 2, "pt2")
    return session, ev, [gak.sum(shared.pt2 + i, axis=1) for i in range(n_outputs)] + [ev.m]


def _counted_ops(session: Session, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    evaluate = session.backend.eval_stage

    def counting(name: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(name)
        return evaluate(name, *args, **kwargs)

    monkeypatch.setattr(session.backend, "eval_stage", counting)
    return calls


@pytest.mark.parametrize("n_outputs", [1, 4, 16])
def test_each_op_node_is_evaluated_once_across_the_outputs(n_outputs: int, monkeypatch: pytest.MonkeyPatch) -> None:
    session, ev, outputs = _outputs(n_outputs)
    union = {nid for out in outputs for nid in cone(session, out.node_id)}
    calls = _counted_ops(session, monkeypatch)
    project_buffers_many(outputs)
    assert len(calls) == len(union - {ev.node_id}) == 6 + 2 * n_outputs


def test_the_shared_memo_leaves_the_projection_unchanged() -> None:
    _, _, outputs = _outputs(4)
    assert project_buffers_many(outputs).read_buffers == {"ev": {"m": BufferNeed.DATA, "Jet.pt": BufferNeed.DATA}}


def test_reach_walks_a_shared_sub_dag_once(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _, outputs = _outputs(4)
    union = {nid for out in outputs for nid in cone(session, out.node_id)}
    visits: list[int] = []
    walk = Session.walk

    def counting_walk(self: Session, array: Any, **handlers: Any) -> object:
        def counted(handler: Any) -> Any:
            def note(nid: int, *rest: Any) -> Any:
                visits.append(nid)
                return handler(nid, *rest)

            return note

        names = ("source", "op", "external")
        return walk(self, array, **{k: counted(h) if k in names else h for k, h in handlers.items()})

    monkeypatch.setattr(Session, "walk", counting_walk)
    assert by_label._reach(outputs) == union
    assert sorted(visits) == sorted(union)
