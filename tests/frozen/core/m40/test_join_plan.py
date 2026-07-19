"""M40 — a two-producer join staged in the additive ``DurablePlanV2`` (plan §4.4, §7.2; contract E1).

A distributed join needs TWO map-write producer stages (build side, probe side) feeding ONE gather
stage that depends on BOTH — ``inputs=(0, 1)`` — unlike a shuffle's single-producer gather
(M39 ``gather_join`` had ``inputs=(0,)``). ``DurablePlanV2`` already carries arbitrary stage DAGs;
these tests pin that a join plan whose IR embeds a real ``Join`` node round-trips byte-identically,
keeps BOTH dependency edges, and is deterministic across runs — the M8 gate for the join path.

This exercises only the CORE seam (``graphed.core`` IR + plan). The frontend ``join_plan`` builder
(E1) is pinned in the frontend M40 suite. Pinned surface: ``GraphStore.add_join`` (two inputs) +
the existing ``DurablePlanV2``/``StageSpec`` API.
"""

from __future__ import annotations

from graphed.core import (
    DurablePlanV2,
    GraphStore,
    OpSpec,
    Partition,
    StageSpec,
    Task,
)

_SCHEME = {"how": "inner", "on": "event"}


def _join_ir() -> bytes:
    g = GraphStore()
    lsrc = g.add_source("left", {"uri": "l.root"})
    rsrc = g.add_source("right", {"uri": "r.root"})
    j = g.add_join([g.add_op("pt", [lsrc]), g.add_op("pt", [rsrc])], _SCHEME)
    out = g.add_reduction("sum", [j])
    return g.serialize(outputs=[out])


def _map_write_stage(side: str, backend_id: str = "graphed-awkward/0") -> StageSpec:
    return StageSpec(
        kind="map_write",
        inputs=(),
        process=OpSpec.from_ref("operator:add"),
        routing={"scheme": "hash", "key": "event", "parts": 4, "side": side, "backend_id": backend_id},
        tasks=(Task(0, Partition(f"{side}.root", "Events", 0, 100)),),
    )


def _gather_join_stage() -> StageSpec:
    return StageSpec(
        kind="gather_join",
        inputs=(0, 1),  # depends on BOTH producer stages (build side 0, probe side 1)
        process=OpSpec.from_ref("operator:add"),
        routing={"parts": 4, "backend_id": "graphed-awkward/0"},
        tasks=(Task(0, Partition("dest", "p", 0, 4)),),
    )


def _join_plan(backend_id: str = "graphed-awkward/0") -> DurablePlanV2:
    return DurablePlanV2(
        ir=_join_ir(),
        stages=(
            _map_write_stage("left", backend_id),
            _map_write_stage("right", backend_id),
            _gather_join_stage(),
        ),
    )


def test_join_plan_is_byte_identical_across_runs() -> None:
    # M8 determinism gate for the join path: identical plan -> identical bytes, twice.
    assert _join_plan().to_bytes() == _join_plan().to_bytes()


def test_join_plan_roundtrips_both_producer_edges() -> None:
    p = _join_plan()
    q = DurablePlanV2.from_bytes(p.to_bytes())
    assert q.to_bytes() == p.to_bytes()
    assert len(q.stages) == 3
    assert q.stages[2].kind == "gather_join"
    # THE two-producer witness: the gather stage depends on BOTH map-write stages, not one.
    assert q.stages[2].inputs == (0, 1), "a join gather must depend on both producer stages"


def test_join_ir_survives_the_plan_roundtrip() -> None:
    # the embedded IR (which carries the Join node) must round-trip byte-identically inside the plan.
    p = _join_plan()
    q = DurablePlanV2.from_bytes(p.to_bytes())
    assert q.ir == _join_ir()
    back = GraphStore.deserialize(q.ir)
    assert [n["kind"] for n in back.nodes()].count("join") == 1, "the plan's IR must keep its Join"


def test_join_plan_task_ids_are_deterministic_and_backend_folded() -> None:
    # per-task determinism + the cross-backend cache-poisoning guard (B-r5.2): two backends routing
    # the same key to different dests must not journal the same id for different content.
    p = _join_plan()
    t0 = p.stages[0].tasks[0]
    assert p.task_id(0, t0) == p.task_id(0, t0)
    assert p.task_id(0, t0) != _join_plan("graphed-numpy/0").task_id(0, t0), "backend_id must fold in"
