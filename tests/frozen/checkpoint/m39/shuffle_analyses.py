"""Self-contained two-phase (map-write -> gather) plan glue for the M39 shuffle-resume suite.

graphed-checkpoint owns the multi-stage journal + resume replay, NOT the shuffle executor, so this
suite drives an ABSTRACT two-phase ``DurablePlanV2`` (deterministic, module-level, importable-by-ref
processes) exactly as the M8 suite drives an abstract histogram plan. Each stage task produces a
content-addressed block; the gather stage's blocks depend on the map-write blocks (recorded as the
journal's ``deps``), so resume reuse and the deps machinery are exercised without real routing.

Pinned stage-process convention (test-author decision): a V2 stage process is
``fn(task, inputs, resources) -> bytes`` where ``inputs`` is the tuple of upstream dep block payloads
(empty for stage 0). ``run_shuffle_resumable`` journals each block with its ``stage`` and ``deps``.
"""

from __future__ import annotations

from typing import Any

from graphed.core import DurablePlanV2, GraphStore, OpSpec, Partition, StageSpec, Task


def map_write(task: Task, inputs: tuple[bytes, ...], resources: Any) -> bytes:
    """Stage-1 producer-task block: deterministic, one per producer-task (no upstream inputs)."""
    return f"mw:{task.key}".encode()


def gather(task: Task, inputs: tuple[bytes, ...], resources: Any) -> bytes:
    """Stage-2 gather block for one dest: a deterministic function of the (sorted) map-write blocks."""
    body = b",".join(sorted(inputs))
    return b"gj:" + str(task.key).encode() + b"|" + body


def _ir() -> bytes:
    g = GraphStore()
    src = g.add_source("events", {"uri": "corpus://values"})
    xchg = g.add_exchange([g.add_op("key", [src])], {"scheme": "hash", "key": "__joinkey__", "parts": 2})
    out = g.add_reduction("sum", [xchg])
    return g.serialize(outputs=[out])


def build_shuffle_plan_v2(n_producers: int = 3, n_dests: int = 2) -> DurablePlanV2:
    mw_tasks = tuple(Task(i, Partition("src", "Events", i, i + 1)) for i in range(n_producers))
    gj_tasks = tuple(Task(d, Partition("dest", "p", d, d + 1)) for d in range(n_dests))
    stages = (
        StageSpec(
            kind="map_write",
            inputs=(),
            process=OpSpec.from_ref("shuffle_analyses:map_write"),
            routing={"scheme": "hash", "key": "__joinkey__", "parts": n_dests, "backend_id": "toy/0"},
            tasks=mw_tasks,
        ),
        StageSpec(
            kind="gather_join",
            inputs=(0,),
            process=OpSpec.from_ref("shuffle_analyses:gather"),
            routing={"parts": n_dests, "backend_id": "toy/0"},
            tasks=gj_tasks,
        ),
    )
    return DurablePlanV2(ir=_ir(), stages=stages)


def n_blocks(n_producers: int = 3, n_dests: int = 2) -> int:
    """Total content-addressed blocks an uninterrupted run commits (map-write + gather)."""
    return n_producers + n_dests
