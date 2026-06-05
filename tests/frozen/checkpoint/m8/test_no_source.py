"""M8 acceptance — a serialized plan deserializes and runs on a machine with NO source files.

Plan M8 contract: "A serialized plan deserializes and runs on a machine with NO source files
present." Here the whole reduction spec is embedded by value (opaque) in the plan; a fresh
interpreter — with this test's directory neither on the path nor as the working dir — loads the
plan bytes and runs ``run_resumable`` to completion using only installed packages.
"""

from __future__ import annotations

import subprocess
import sys

from graphed_core import DurablePlan, GraphStore, OpSpec, Partition

from graphed_checkpoint import run_resumable

SEED, N, BINS, LO, HI = 99, 1200, 10, 0.0, 100.0


def _opaque_plan() -> DurablePlan:
    def proc(part, res):  # type: ignore[no-untyped-def]
        import numpy as np  # noqa: PLC0415

        vals = np.random.default_rng(SEED).uniform(LO, HI, N)[part.entry_start : part.entry_stop]
        return np.histogram(vals, bins=BINS, range=(LO, HI))[0].astype("int64")

    def add(a, b):  # type: ignore[no-untyped-def]
        return a + b

    def empty():  # type: ignore[no-untyped-def]
        import numpy as np  # noqa: PLC0415

        return np.zeros(BINS, dtype="int64")

    g = GraphStore()
    g.mark_output(g.add_op("hist", [g.add_source("events")]))
    edges = [0, N // 3, 2 * N // 3, N]
    parts = tuple(Partition("x", "", edges[i], edges[i + 1]) for i in range(3))
    return DurablePlan(
        ir=g.serialize(),
        process=OpSpec.from_callable(proc),
        combine=OpSpec.from_callable(add),
        empty=OpSpec.from_callable(empty),
        partitions=parts,
    )


def test_serialized_plan_runs_with_no_user_source(tmp_path) -> None:  # type: ignore[no-untyped-def]
    plan = _opaque_plan()
    assert plan.opaque  # the reduction spec is embedded by value
    blob = tmp_path / "plan.bin"
    blob.write_bytes(plan.to_bytes())
    store_dir = tmp_path / "store"

    child = (
        "import sys; from graphed_core import DurablePlan; from graphed_checkpoint import Store, run_resumable;"
        "p=DurablePlan.from_bytes(open(sys.argv[1],'rb').read());"
        "r=run_resumable(p, Store(sys.argv[2]));"
        "print(int(r.value.sum()), r.report.executed)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", child, str(blob), str(store_dir)],
        cwd=tmp_path,  # not this test's directory
        env={"PATH": "/usr/bin:/bin"},  # scrub PYTHONPATH; only installed packages are available
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    # all N uniform draws fall inside [LO,HI) -> the histogram sums to N; 3 partitions executed
    assert proc.stdout.split() == [str(N), "3"], proc.stdout


def test_resumed_subprocess_plan_skips_completed_work(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # run once locally to populate the store, then a fresh interpreter must SKIP all of it
    plan = _opaque_plan()
    store_dir = tmp_path / "store"
    run_resumable(plan, __import__("graphed_checkpoint").Store(store_dir))
    blob = tmp_path / "plan.bin"
    blob.write_bytes(plan.to_bytes())

    child = (
        "import sys; from graphed_core import DurablePlan; from graphed_checkpoint import Store, run_resumable;"
        "p=DurablePlan.from_bytes(open(sys.argv[1],'rb').read());"
        "r=run_resumable(p, Store(sys.argv[2]));"
        "print(r.report.executed, r.report.skipped)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", child, str(blob), str(store_dir)],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["0", "3"], proc.stdout  # nothing re-run; all 3 reused
