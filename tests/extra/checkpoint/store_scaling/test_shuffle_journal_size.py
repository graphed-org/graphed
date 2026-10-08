"""``run_shuffle_resumable`` journal records stay constant-size as the upstream block count grows."""

from __future__ import annotations

import json
from typing import Any

from graphed.checkpoint import Store, run_shuffle_resumable
from graphed.core import DurablePlanV2, GraphStore, OpSpec, Partition, StageSpec, Task

_G = 3


def map_write(task: Task, inputs: tuple[bytes, ...], resources: Any) -> bytes:
    return f"mw:{task.key}".encode()


def gather(task: Task, inputs: tuple[bytes, ...], resources: Any) -> bytes:
    return b"gj:" + b",".join(sorted(inputs))


def _plan(n_map: int) -> DurablePlanV2:
    g = GraphStore()
    src = g.add_source("events", {"uri": "corpus://values"})
    ir = g.serialize(outputs=[g.add_reduction("sum", [src])])
    mw = tuple(Task(i, Partition("src", "Events", i, i + 1)) for i in range(n_map))
    gj = tuple(Task(d, Partition("dest", "p", d, d + 1)) for d in range(_G))
    stages = (
        StageSpec(
            kind="map_write",
            inputs=(),
            process=OpSpec.from_ref(f"{__name__}:map_write"),
            routing={"parts": _G, "backend_id": "toy/0"},
            tasks=mw,
        ),
        StageSpec(
            kind="gather_join",
            inputs=(0,),
            process=OpSpec.from_ref(f"{__name__}:gather"),
            routing={"parts": _G, "backend_id": "toy/0"},
            tasks=gj,
        ),
    )
    return DurablePlanV2(ir=ir, stages=stages)


def _journal_line_lengths(store: Store) -> list[int]:
    return [len(line) for line in store.journal_path.read_text().splitlines()]


def test_record_size_does_not_grow_with_the_upstream_block_count(tmp_path: Any) -> None:
    small, large = Store(tmp_path / "small"), Store(tmp_path / "large")
    run_shuffle_resumable(_plan(4), small)
    run_shuffle_resumable(_plan(64), large)
    assert max(_journal_line_lengths(large)) == max(_journal_line_lengths(small))


def test_a_gather_record_still_names_its_upstream_blocks(tmp_path: Any) -> None:
    store = Store(tmp_path)
    run_shuffle_resumable(_plan(5), store)
    done = store.completed()
    maps = sorted(e.blob for e in done.values() if e.stage == "map_write")
    gathers = [e for e in done.values() if e.stage == "gather_join"]
    assert len(gathers) == _G
    for e in gathers:
        (listing,) = e.deps
        blob = store.get(listing)
        assert blob is not None
        assert sorted(json.loads(blob)) == maps


def test_a_journal_with_per_block_deps_still_resumes(tmp_path: Any) -> None:
    store = Store(tmp_path)
    plan = _plan(5)
    first = run_shuffle_resumable(plan, store)
    maps = [e.blob for e in store.completed().values() if e.stage == "map_write"]
    rewritten = []
    for line in store.journal_path.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("stage") == "gather_join":
            rec["deps"] = maps
        rewritten.append(json.dumps(rec))
    store.journal_path.write_text("\n".join(rewritten) + "\n")
    again = run_shuffle_resumable(plan, store)
    assert again.report.executed == 0
    assert again.value == first.value
