"""M8 acceptance — error harvesting, retry policies, and the error budget.

Plan M8 contract: "An injected per-partition failure lands in the dead-letter set with a
reproducible descriptor; `retry_smaller_chunk` succeeds where the original OOMed (simulated)."
Plus the policies `retry_n | retry_smaller_chunk | retry_elsewhere | quarantine` and an error budget
as a stopping condition.
"""

from __future__ import annotations

import analyses
import numpy as np

from graphed.checkpoint import (
    Quarantine,
    RetryElsewhere,
    RetryN,
    RetrySmallerChunk,
    Store,
    run_resumable,
)


def test_injected_failure_lands_in_dead_letter_with_reproducible_descriptor(tmp_path) -> None:  # type: ignore[no-untyped-def]
    plan = analyses.build_plan("analyses:poison_one", 6)
    res = run_resumable(plan, Store(tmp_path), retry=Quarantine())
    assert res.report.dead == 1
    (dl,) = res.report.dead_letters
    # the descriptor identifies the exact partition and carries M6 StageError provenance
    assert dl["entry_start"] == 1000
    assert dl["error_type"] == "StageError"
    assert dl["stage_error"]["op"] == "jet_pt"
    assert dl["stage_error"]["user_file"] == "analysis.py"
    assert dl["stage_error"]["user_line"] == 42
    # reproducible: the recorded task_id is exactly the plan's content-addressed id for that partition
    poison = next(p for p in plan.partitions if p.entry_start == 1000)
    assert dl["task_id"] == plan.task_id(poison)


def test_dead_lettered_partition_does_not_corrupt_the_rest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # the surviving partitions still reduce correctly (the poison chunk is simply absent)
    plan = analyses.build_plan("analyses:poison_one", 6)
    res = run_resumable(plan, Store(tmp_path), retry=Quarantine())
    poison = next(p for p in plan.partitions if p.entry_start == 1000)
    expected = analyses._hist(
        np.concatenate(
            [analyses._all_values()[p.entry_start : p.entry_stop] for p in plan.partitions if p is not poison]
        )
    )
    assert np.array_equal(res.value, expected)


def test_error_budget_stops_the_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # every partition poisons (entry_start==0 isn't poison, so use a plan where many fail): use a
    # process that always raises by pointing poison at all chunks via a single-chunk plan repeated
    plan = analyses.build_plan("analyses:always_fail", 6, error_budget=1)
    res = run_resumable(plan, Store(tmp_path), retry=Quarantine())
    assert res.report.stopped == "error_budget"
    assert res.report.dead == 2  # stopped once dead (2) exceeds the budget (1)


def test_retry_smaller_chunk_succeeds_where_whole_chunk_oomed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    plan = analyses.build_plan("analyses:oom_if_big", 2)  # 3000-entry chunks -> OOM whole
    # without recovery the big chunks are dead-lettered
    bare = run_resumable(plan, Store(tmp_path / "bare"), retry=Quarantine())
    assert bare.report.dead == 2 and bare.report.executed == 0
    # retry_smaller_chunk splits until each sub-chunk fits, then recombines -> single-pass result
    fixed = run_resumable(plan, Store(tmp_path / "fixed"), retry=RetrySmallerChunk(splits=2, min_size=1))
    assert fixed.report.dead == 0
    assert np.array_equal(fixed.value, analyses.reference())


def test_retry_n_recovers_a_transient_failure(tmp_path) -> None:  # type: ignore[no-untyped-def]
    plan = analyses.build_plan("analyses:flaky_twice", 4)
    res = run_resumable(plan, Store(tmp_path), resources={}, retry=RetryN(2))
    assert res.report.dead == 0
    assert np.array_equal(res.value, analyses.reference())


def test_exhausted_retries_still_dead_letter(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # a permanently-failing partition exhausts RetryN and is dead-lettered (not silently dropped)
    plan = analyses.build_plan("analyses:always_fail", 3)
    res = run_resumable(plan, Store(tmp_path), retry=RetryN(2))
    assert res.report.dead == 3 and res.report.executed == 0


def test_smaller_chunk_gives_up_on_a_non_size_failure(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # RetrySmallerChunk splits down to size 1; a failure that is not about size is still dead-lettered
    plan = analyses.build_plan("analyses:always_fail", 2)
    res = run_resumable(plan, Store(tmp_path), retry=RetrySmallerChunk(splits=2, min_size=1))
    assert res.report.dead == 2 and res.report.executed == 0


def test_retry_elsewhere_reruns_on_a_fresh_worker_context(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # the first worker context is "bad"; RetryElsewhere re-runs on a fresh context that succeeds
    plan = analyses.build_plan("analyses:fails_on_bad_worker", 4)
    res = run_resumable(
        plan,
        Store(tmp_path),
        resources={"bad": True},
        retry=RetryElsewhere(attempts=1, new_resources=dict),
    )
    assert res.report.dead == 0
    assert np.array_equal(res.value, analyses.reference())
