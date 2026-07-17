"""M20: the format-agnostic partitioned-write base (user-directed refactor of P3.6).

`graphed.write` is what every deferred writer specializes — parquet (the backends) and ROOT (the
reader integration) alike: the write plan (compute-disabled task graph whose later run IS the
enabled mode), the dependency-free key-ordered sequential runner, per-file base tables over
GENERIC keys (a plain uri, or a (uri, tree) pair for container formats), suffix-explicit part
naming, and part-index derivation from the partition alone (R15.9). `graphed.parquet` keeps its
M15 surface as the parquet specialization — pinned here as aliases, so the m15 suites and this
one can never diverge.
"""

from __future__ import annotations

import os

import pytest
from graphed_core import Partition, SequentialRunner
from graphed_core import execution as _gce
from graphed_core.execution import WorkerResources
from m15_toy import ToyForm, session  # noqa: F401  (the m15 toy backend serves this suite too)

from graphed import parquet as gpq
from graphed import write as gw


def _toy_writer(partition: Partition, resources: WorkerResources) -> list[str]:
    out = partition.uri + f".w{partition.entry_start}-{partition.entry_stop}.txt"
    with open(out, "w") as f:
        f.write("x")
    return [out]


def test_write_plan_and_sequential_runner_are_format_agnostic(tmp_path) -> None:  # type: ignore[no-untyped-def]
    parts = tuple(Partition(os.path.join(tmp_path, f"f{i}"), "tree", 0, i + 1) for i in range(3))
    plan = gw.write_plan(tuple(reversed(parts)), _toy_writer)
    assert plan.empty() == []
    result = SequentialRunner().run(plan)
    assert result.n_partitions == 3
    # key order, regardless of the order handed in; workers report their own paths
    assert result.value == [
        t.partition.uri + f".w{t.partition.entry_start}-{t.partition.entry_stop}.txt"
        for t in sorted(plan.tasks, key=lambda t: t.key)
    ]
    for p in result.value:
        assert os.path.exists(p)


def test_file_bases_accepts_generic_keys() -> None:
    uris = ["a.root", "b.root"]
    assert gw.file_bases(uris, 3) == {"a.root": 0, "b.root": 3}
    pairs = [("a.root", "events"), ("a.root", "other"), ("b.root", "events")]
    bases = gw.file_bases(pairs, 2)
    assert bases == {("a.root", "events"): 0, ("a.root", "other"): 2, ("b.root", "events"): 4}


def test_blind_part_index_derives_from_the_partition_alone() -> None:
    bases_uri = gw.file_bases(["a", "b"], 4)
    p = Partition.blind("b", "", 2, 4)
    assert gw.blind_part_index(p, bases_uri) == 6
    bases_pair = gw.file_bases([("a", "t1"), ("a", "t2")], 4)
    q = Partition.blind("a", "t2", 1, 4)
    assert gw.blind_part_index(q, bases_pair) == 5  # container formats key by (uri, tree)
    with pytest.raises(ValueError):
        gw.blind_part_index(Partition("a", "", 0, 5), bases_uri)  # not blind: no step to derive


def test_step_of_reconstructs_exactly() -> None:
    # entry_start alone is NOT invertible (n=5, steps=3 gives starts 0,1,3): exact reconstruction
    n, steps = 5, 3
    ranges = [((s * n) // steps, ((s + 1) * n) // steps) for s in range(steps)]
    for s, (start, stop) in enumerate(ranges):
        assert gw.step_of(start, stop, n, steps) == s
    with pytest.raises(ValueError):
        gw.step_of(0, 4, n, steps)  # no such step range


def test_part_path_is_suffix_explicit() -> None:
    assert gw.part_path("/out", 7, suffix=".root") == os.path.join("/out", "part-00007.root")
    assert gw.part_path("/out", 7, prefix="ana", suffix=".parquet") == os.path.join(
        "/out", "ana-00007.parquet"
    )


def test_parquet_remains_the_specialization_not_a_fork() -> None:
    # the M15 parquet surface IS the base (aliases) — the suites can never diverge
    assert gpq.write_plan is gw.write_plan
    assert gpq.file_bases is gw.file_bases
    # M32: SequentialRunner is general execution, not a write/parquet concept — it lives in the
    # execution contract and is NOT re-exported by either write-shaped module
    assert not hasattr(gw, "SequentialRunner")
    assert not hasattr(gpq, "SequentialRunner")
    assert SequentialRunner is _gce.SequentialRunner  # the one canonical reference runner
    assert gpq.part_path("/out", 7) == os.path.join("/out", "part-00007.parquet")  # m15 pin holds
