"""M40 (E5 / F6) — ``join_plan`` RECORDS the broadcast-vs-shuffle choice as a durable plan field.

The broadcast decision must be computed at plan build (from typetracer/estimated form sizes) and FROZEN
into the :class:`~graphed.core.DurablePlanV2`, so an executor honours a reproducible plan choice instead
of racily recomputing it from the runtime worker pool (F6). The current ``join_plan`` records only
``{scheme, key, parts, backend_id, how}`` in the ``gather_join`` routing (``shuffle.py``) — there is NO
broadcast field, so there is nothing durable for the executor to honour.

This pins the field the fix must add: the ``gather_join`` stage carries a boolean ``broadcast`` choice
that is deterministic across independent builds and SURVIVES canonical serialization (a real durable
field in ``to_bytes``/``from_bytes``, not an ephemeral in-memory attribute). The value's agreement with
the pinned cost rule on real form sizes is witnessed executor-side (worker-count invariance) in
``graphed-executors`` ``tests/frozen/m40/test_join_broadcast_plan_choice.py``.
"""

from __future__ import annotations

from shuffle_backends import ON, ToyJoinBackend, toy_join_sources

import graphed
from graphed import Session
from graphed.core import DurablePlanV2, StageSpec


def _plan(s: Session) -> DurablePlanV2:
    left, right = toy_join_sources(s)
    joined = graphed.join(left, right, on=ON, how="inner")
    return graphed.join_plan(joined, backend=ToyJoinBackend, steps_per_file=3)


def _gather(plan: DurablePlanV2) -> StageSpec:
    return next(st for st in plan.stages if st.kind == "gather_join")


def test_gather_join_records_a_broadcast_boolean() -> None:
    gather = _gather(_plan(Session(ToyJoinBackend())))
    assert "broadcast" in gather.routing, "join_plan must record the broadcast-vs-shuffle choice (E5)"
    assert isinstance(gather.routing["broadcast"], bool), "the recorded choice is a boolean cost decision"


def test_broadcast_choice_is_deterministic_and_durable() -> None:
    a = _plan(Session(ToyJoinBackend()))
    b = _plan(Session(ToyJoinBackend()))
    assert _gather(a).routing["broadcast"] == _gather(b).routing["broadcast"], (
        "the recorded plan choice must not drift across independent builds"
    )
    # durable: the choice round-trips through the canonical serialization, not an ephemeral attribute.
    round_tripped = DurablePlanV2.from_bytes(a.to_bytes())
    assert _gather(round_tripped).routing["broadcast"] == _gather(a).routing["broadcast"], (
        "the broadcast choice must survive to_bytes/from_bytes (it is a durable plan field)"
    )
    assert a.to_bytes() == b.to_bytes(), "the durable plan (including the recorded choice) is byte-deterministic"
