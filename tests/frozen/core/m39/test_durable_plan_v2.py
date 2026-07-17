"""M39 — the additive multi-stage plan schema ``DurablePlanV2`` (plan §4.4, §7.2).

A shuffle needs genuine intra-run staging (a gather stage depending on a map-write stage), which the
single map->reduce ``DurablePlan`` cannot express. ``DurablePlanV2`` is a SEPARATE, purely-additive
class with a DISTINCT ``format_version`` (``"graphed-plan/2"`` — a string, vs V1's int), so its bytes
can never collide with a V1 blob and a V1 reader rejects a V2 blob loudly. Its V2-only task ids fold
``backend_id`` (read from ``StageSpec.routing``, no schema change) so two conforming backends that
route a key to different dests never journal the same id for different content (B-r5.2).

Pinned surface: ``DurablePlanV2(ir, stages, ...)`` + ``StageSpec(kind, inputs, process, routing,
tasks)`` importable from ``graphed_core``; ``to_bytes``/``from_bytes`` deterministic; and
``DurablePlanV2.task_id(stage_index, task)`` folding the stage's ``routing["backend_id"]``.
"""

from __future__ import annotations

import pytest

from graphed_core import (
    DurablePlan,
    DurablePlanV2,
    GraphStore,
    OpSpec,
    Partition,
    StageSpec,
    Task,
)


def _ir() -> bytes:
    g = GraphStore()
    src = g.add_source("events", {"uri": "data.root"})
    xchg = g.add_exchange([g.add_op("pt", [src])], {"scheme": "hash", "key": "__joinkey__", "parts": 4})
    out = g.add_reduction("sum", [xchg])
    return g.serialize(outputs=[out])


def _map_write_stage(backend_id: str = "graphed-awkward/0") -> StageSpec:
    tasks = (
        Task(0, Partition("a.root", "Events", 0, 100)),
        Task(1, Partition("a.root", "Events", 100, 200)),
    )
    return StageSpec(
        kind="map_write",
        inputs=(),
        process=OpSpec.from_ref("operator:add"),
        routing={"scheme": "hash", "key": "__joinkey__", "parts": 4, "backend_id": backend_id},
        tasks=tasks,
    )


def _gather_stage() -> StageSpec:
    return StageSpec(
        kind="gather_join",
        inputs=(0,),  # depends on stage 0 (the map-write)
        process=OpSpec.from_ref("operator:add"),
        routing={"parts": 4, "backend_id": "graphed-awkward/0"},
        tasks=(Task(0, Partition("dest", "p", 0, 4)),),
    )


def _v2(backend_id: str = "graphed-awkward/0") -> DurablePlanV2:
    return DurablePlanV2(ir=_ir(), stages=(_map_write_stage(backend_id), _gather_stage()))


def _v1() -> DurablePlan:
    return DurablePlan(
        ir=_ir(),
        process=OpSpec.from_ref("operator:add"),
        combine=OpSpec.from_ref("operator:add"),
        empty=OpSpec.from_ref("builtins:float"),
        partitions=(Partition("a.root", "Events", 0, 100),),
    )


# ---- additive schema, distinct version ----------------------------------------------------------
def test_v2_declares_a_distinct_string_version() -> None:
    assert _v2().format_version == "graphed-plan/2"
    assert _v1().format_version == 1  # V1 stays an int -> the two byte spaces can never overlap


def test_v2_is_byte_identical_for_identical_plans() -> None:
    assert _v2().to_bytes() == _v2().to_bytes()


def test_v2_roundtrips_stages_and_deps() -> None:
    p = _v2()
    q = DurablePlanV2.from_bytes(p.to_bytes())
    assert q.to_bytes() == p.to_bytes()
    assert len(q.stages) == 2
    assert q.stages[0].kind == "map_write"
    assert q.stages[1].kind == "gather_join"
    assert q.stages[1].inputs == (0,), "the gather stage's dependency edge on the map stage must survive"


def test_a_v1_reader_rejects_a_v2_blob_and_vice_versa() -> None:
    # dispatch is on format_version: neither reader may silently mis-parse the other's bytes.
    with pytest.raises((ValueError, KeyError)):
        DurablePlan.from_bytes(_v2().to_bytes())
    with pytest.raises((ValueError, KeyError)):
        DurablePlanV2.from_bytes(_v1().to_bytes())


# ---- backend_id folded into V2-only task ids (B-r5.2) -------------------------------------------
def test_v2_task_id_is_deterministic_and_per_task() -> None:
    p = _v2()
    t0, t1 = p.stages[0].tasks
    assert p.task_id(0, t0) == p.task_id(0, t0)
    assert p.task_id(0, t0) != p.task_id(0, t1), "different producer-tasks -> different ids"
    assert len(p.task_id(0, t0)) == 64


def test_v2_task_id_folds_the_backend_id() -> None:
    # THE cross-backend cache-poisoning witness: two backends route the SAME key to DIFFERENT dests,
    # so the same structural id must NOT be reused for different content — folding backend_id closes it.
    awk = _v2("graphed-awkward/0")
    npy = _v2("graphed-numpy/0")
    t0 = awk.stages[0].tasks[0]
    assert awk.task_id(0, t0) != npy.task_id(0, t0), "backend_id must change the V2 map-write id"


def test_backend_id_rides_in_routing_not_a_schema_change() -> None:
    # §7.2 r6: backend_id is recorded in the EXISTING StageSpec.routing map, so §4.4's schema is
    # untouched. Witness it is present there and drives the id (previous test), not a new top field.
    assert _v2().stages[0].routing["backend_id"] == "graphed-awkward/0"
