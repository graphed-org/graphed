"""M40 — the multi-source ``graphed.join_plan`` builder (plan §3.2; contract E1, target 13).

A join has TWO independently-partitioned sources, so the single-source ``aggregate_plan`` /
``shuffle_plan`` guards cannot express it. ``join_plan`` is a NEW builder mirroring ``shuffle_plan``
that emits a **multi-stage** :class:`~graphed.core.DurablePlanV2`: one ``map_write`` stage per side
that a single ``kind="gather_join"`` stage (``inputs=(0, 1)``) depends on. This witnesses the plan
SHAPE (kinds + the two-input barrier edge) + its byte-identical serialization; cross-process
execution is witnessed in graphed-executors.

Backend-agnostic (toy ``ShuffleBackend`` + ``ListSource``), mirroring
``tests/frozen/frontend/m39/test_shuffle_plan_builder.py``. Pre-implementation ``graphed`` has no
``join``/``join_plan`` attribute -> right-reason ``AttributeError``.
"""

from __future__ import annotations

from shuffle_backends import ON, ToyJoinBackend, toy_join_sources

import graphed
from graphed import Session
from graphed.core import DurablePlanV2


def _plan(s: Session) -> DurablePlanV2:
    left, right = toy_join_sources(s)
    joined = graphed.join(left, right, on=ON, how="inner")
    return graphed.join_plan(joined, backend=ToyJoinBackend, steps_per_file=3)


def test_join_plan_emits_two_map_writes_and_a_gather_join() -> None:
    plan = _plan(Session(ToyJoinBackend()))
    assert isinstance(plan, DurablePlanV2), "a join must serialize as the multi-stage V2 plan"
    kinds = [st.kind for st in plan.stages]
    assert kinds.count("map_write") == 2, "one map-write stage per join side (co-partition)"
    assert "gather_join" in kinds, "the assemble stage is a gather_join (kind reserved for M40)"


def test_gather_join_depends_on_both_map_write_stages() -> None:
    plan = Session(ToyJoinBackend())
    p = _plan(plan)
    gather = next(st for st in p.stages if st.kind == "gather_join")
    map_idxs = [i for i, st in enumerate(p.stages) if st.kind == "map_write"]
    assert len(map_idxs) == 2
    assert set(map_idxs) <= set(gather.inputs), (
        "gather_join must depend on BOTH map-write stages (the two-input barrier edge)"
    )


def test_join_plan_to_bytes_is_deterministic_across_runs() -> None:
    # determinism: two independent builds of the same join produce a byte-identical durable plan.
    a = _plan(Session(ToyJoinBackend())).to_bytes()
    b = _plan(Session(ToyJoinBackend())).to_bytes()
    assert a == b, "join_plan serialization must be byte-identical across runs (M8 determinism)"
