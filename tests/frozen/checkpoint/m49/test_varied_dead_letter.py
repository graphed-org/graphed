"""§7.4 — retry and dead-lettering stay PARTITION-atomic over a varied graph.

One poisoned variation dead-letters the partition's whole composite: the partition is the unit, so
every universe loses it, not only the guilty one. The LABEL that surface names rides the §8.2
`StageError` anchor in `graphed-executors`; what is bound here is the granularity.
"""

from __future__ import annotations

from collections import Counter

import m49_analyses as A

from graphed.checkpoint import Store, run_resumable
from graphed.checkpoint.retry import Quarantine, RetryN

DESCRIPTOR_KEYS = frozenset(
    {"task_id", "uri", "tree", "entry_start", "entry_stop", "error_type", "error_message"}
)


def _run(store, *, poison, partitions=None, retry=None):
    plan, compiled, tags = A.build_plan(poison=poison)
    dp = A.durable(plan, compiled)
    if partitions is not None:
        dp = dp.with_partitions(partitions)
    A.READS.clear()
    return dp, tags, run_resumable(dp, store, retry=retry)


def test_a_poison_free_twin_dead_letters_nothing(tmp_path) -> None:
    """Positive control: the fixture's shape, minus the poison, runs every partition clean."""
    dp, tags, res = _run(Store(tmp_path), poison=False)
    assert res.report.dead == 0 and res.report.dead_letters == []
    assert res.report.executed == len(dp.partitions)
    assert len(set(res.value.tolist())) == len(tags)


def test_one_poisoned_variation_dead_letters_the_whole_composite(tmp_path) -> None:
    dp, _tags, poisoned = _run(Store(tmp_path / "poisoned"), poison=True)
    first, rest = dp.partitions[0], dp.partitions[1:]

    assert poisoned.report.dead == 1
    assert poisoned.report.executed == len(dp.partitions) - 1
    (descriptor,) = poisoned.report.dead_letters
    assert descriptor["task_id"] == dp.task_id(first)
    assert (descriptor["entry_start"], descriptor["entry_stop"]) == (first.entry_start, first.entry_stop)

    # ORACLE: the same plan over the surviving partitions alone. Byte-equality says the composite
    # lost the poisoned partition WHOLE — every universe, not just the guilty one.
    _dp2, _tags2, short = _run(Store(tmp_path / "short"), poison=True, partitions=rest)
    assert short.report.dead == 0
    assert poisoned.value.tobytes() == short.value.tobytes()

    # DISCRIMINATOR: the nominal universe is the same expression in both builds, so a per-variation
    # dead-letter would have left its total intact.
    _dp3, tags3, clean = _run(Store(tmp_path / "clean"), poison=False)
    nominal = tags3.index("nominal")
    assert poisoned.value[nominal] != clean.value[nominal]


def test_retry_reruns_the_whole_composite_not_the_failing_universe(tmp_path) -> None:
    """`process` IS the composite, so an attempt reads the WHOLE partition once, `n+1` times over."""
    _dp, _tags, quarantined = _run(Store(tmp_path / "q"), poison=True, retry=Quarantine())
    first_reads = Counter(A.READS)
    dp, _tags2, retried = _run(Store(tmp_path / "r"), poison=True, retry=RetryN(n=2))
    retried_reads = Counter(A.READS)

    first = (dp.partitions[0].entry_start, dp.partitions[0].entry_stop)
    assert first_reads[first] == 1
    assert retried_reads[first] == 3, "RetryN(2) must re-run the composite twice more"
    for part in dp.partitions[1:]:
        key = (part.entry_start, part.entry_stop)
        assert retried_reads[key] == 1, "a partition that never failed must not be retried"

    # a deterministic poison is not recoverable: retry changes the attempt count, not the outcome
    assert quarantined.report.dead == retried.report.dead == 1
    assert quarantined.value.tobytes() == retried.value.tobytes()


def test_the_descriptor_keeps_its_fixed_key_list_and_gains_no_variation_key(tmp_path) -> None:
    _dp, _tags, res = _run(Store(tmp_path), poison=True)
    (descriptor,) = res.report.dead_letters
    assert set(descriptor) == DESCRIPTOR_KEYS
    assert descriptor["error_type"] == "ValueError"
