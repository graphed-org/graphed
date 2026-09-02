"""§7.3 — interrupt/resume over a VARIED graph is byte-identical to an uninterrupted run.

The N-variation composite partial is the journal unit: one partition's stored blob carries every
universe's value. The fixture builds the `DurablePlan` BY VALUE over the plan `aggregate_plan`
returned (§7.3), because `run_resumable` takes a `DurablePlan` and `aggregate_plan` returns a plain
`Plan`.
"""

from __future__ import annotations

import m49_analyses as A
import pytest

from graphed.checkpoint import Store, run_resumable
from graphed.checkpoint.runner import _SimulatedInterrupt
from graphed.core import GraphStore


def _durable():  # type: ignore[no-untyped-def]
    plan, compiled, tags = A.build_plan()
    return A.durable(plan, compiled), tags


def test_the_journal_unit_is_the_whole_variation_composite(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The mechanism this suite rests on: ONE plan, ONE IR, one marked output per universe.

    A per-variation re-execution (§7.1) would show as anything other than `len(labels)` compiled
    outputs, and a nominal-only lowering would show as coincident per-universe values.
    """
    dp, tags = _durable()
    assert len(GraphStore.deserialize(dp.ir).outputs()) == len(tags)
    A.READS.clear()
    value = run_resumable(dp, Store(tmp_path)).value
    assert value.shape == (len(tags),)
    assert len(set(value.tolist())) == len(tags), "the universes must not coincide"
    assert len(A.READS) == len(dp.partitions), "one read per partition, not one per universe"


def test_kill_then_resume_is_byte_identical_and_does_less_work(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dp, _tags = _durable()
    reference = run_resumable(dp, Store(tmp_path / "clean")).value

    store = Store(tmp_path / "resumed")
    with pytest.raises(_SimulatedInterrupt):
        run_resumable(dp, store, _kill_after=4)
    assert len(store.completed()) == 4  # only committed composites survived the "kill"

    res = run_resumable(dp, store)
    assert res.report.skipped == 4 and res.report.executed == len(dp.partitions) - 4
    assert res.report.did_less_work
    assert res.value.dtype == reference.dtype and res.value.shape == reference.shape
    assert res.value.tobytes() == reference.tobytes()


def test_no_double_count_at_any_kill_boundary(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Every universe must contribute each partition exactly once, wherever the crash landed."""
    dp, _tags = _durable()
    reference = run_resumable(dp, Store(tmp_path / "clean")).value.tobytes()
    for k in range(1, len(dp.partitions)):
        store = Store(tmp_path / f"k{k}")
        with pytest.raises(_SimulatedInterrupt):
            run_resumable(dp, store, _kill_after=k)
        res = run_resumable(dp, store)
        assert res.report.skipped == k
        assert res.value.tobytes() == reference, f"kill after {k} changed the composite"


def test_resume_after_completion_recomputes_no_universe(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dp, _tags = _durable()
    store = Store(tmp_path)
    first = run_resumable(dp, store)
    A.READS.clear()
    again = run_resumable(dp, store)
    assert again.report.executed == 0 and again.report.skipped == len(dp.partitions)
    assert A.READS == [], "a skipped composite must not re-read its partition"
    assert again.value.tobytes() == first.value.tobytes()
