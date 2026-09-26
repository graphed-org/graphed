"""m72 (a)/(c): deferred part writes ride in `aggregate_plan` beside its reductions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from m72_multiout_fixtures import (
    CODEC_CALLS,
    EVALS,
    SEEN,
    add,
    by_range,
    by_step,
    by_step_in_dir,
    first_as_float,
    fixed_name,
    json_codec,
    no_paths,
    paths_only,
    read_part,
    record,
    reset,
    spy_reduce,
    zero,
)

import graphed
from graphed import GraphedError, compile_ir, refuse_chunk_partials
from graphed.core import GraphStore, Partition
from graphed.core.execution import SequentialRunner

FILES = {"fa": np.arange(0.0, 12.0), "fb": np.arange(20.0, 26.0)}
F32 = {"fa": np.array([0.1, 0.2, 0.7, 0.4], dtype=np.float32)}


@pytest.fixture(autouse=True)
def _clean() -> None:
    reset()


def _write(array: Any, dest: Path, name: Any = by_step, metadata: Any = None) -> Any:
    return graphed.write.PartWrite(array=array, destination=str(dest), name=name, codec=json_codec, metadata=metadata)


def test_write_and_reduce_share_one_read_and_one_evaluation(tmp_path: Path) -> None:
    _s, x, src = record(FILES)
    y = x * 2.0
    plan = graphed.aggregate_plan(
        y.sum(), reduce=first_as_float, combine=add, empty=lambda: 0.0, steps_per_file=3,
        writes=[_write(y, tmp_path)],
    )
    value = SequentialRunner().run(plan).value
    n = len(plan.tasks)
    assert n == 6
    assert value == float(2 * sum(v.sum() for v in FILES.values()))
    assert len(src.reads) == n
    assert EVALS["mul"] == n
    for task in plan.tasks:
        part = read_part(os.path.join(tmp_path, by_step(task.partition)))
        assert part["values"] == (2.0 * src.chunk(task.partition)).tolist()


def test_reduce_sees_todays_values_then_one_path_per_write(tmp_path: Path) -> None:
    _s, x, src = record(FILES)
    a, b, d = x * 2.0, x + 1.0, x * 3.0
    writes = [_write(a, tmp_path / "wa"), _write(b * 1.0, tmp_path / "wb"), _write(d, tmp_path / "wd")]
    plan = graphed.aggregate_plan(a, b, a, reduce=spy_reduce, combine=add, empty=zero, writes=writes)
    SequentialRunner().run(plan)
    assert len(SEEN) == len(plan.tasks) == 2
    for task, seen in zip(sorted(plan.tasks, key=lambda t: t.key), SEEN, strict=True):
        chunk = src.chunk(task.partition)
        assert len(seen) == 5
        np.testing.assert_array_equal(seen[0], 2.0 * chunk)
        np.testing.assert_array_equal(seen[1], chunk + 1.0)
        assert seen[2:] == [os.path.join(tmp_path / w, by_step(task.partition)) for w in ("wa", "wb", "wd")]
        assert read_part(seen[2])["values"] == (2.0 * chunk).tolist()
        assert read_part(seen[3])["values"] == (chunk + 1.0).tolist()
        assert read_part(seen[4])["values"] == (3.0 * chunk).tolist()

    SEEN.clear()
    SequentialRunner().run(graphed.aggregate_plan(a, b, a, reduce=spy_reduce, combine=add, empty=zero))
    assert [len(seen) for seen in SEEN] == [2, 2]
    for task, seen in zip(sorted(plan.tasks, key=lambda t: t.key), SEEN, strict=True):
        np.testing.assert_array_equal(seen[0], 2.0 * src.chunk(task.partition))
        np.testing.assert_array_equal(seen[1], src.chunk(task.partition) + 1.0)


class _Static:
    calls = 0

    def __str__(self) -> str:
        type(self).calls += 1
        return "static"


def test_part_metadata_holds_this_chunks_reduction(tmp_path: Path) -> None:
    _Static.calls = 0
    _s, x, src = record(F32)
    static = _Static()
    plan = graphed.aggregate_plan(
        reduce=paths_only, combine=add, empty=no_paths, steps_per_file=2,
        writes=[
            _write(x, tmp_path / "meta", metadata={"sumw": x.sum(), "kind": static}),
            _write(x, tmp_path / "none"),
        ],
    )
    assert _Static.calls == 1
    SequentialRunner().run(plan)
    assert _Static.calls == 1
    sums = []
    for task in sorted(plan.tasks, key=lambda t: t.key):
        chunk_sum = src.chunk(task.partition).sum()
        assert str(chunk_sum) != str(float(chunk_sum))
        kv = read_part(os.path.join(tmp_path / "meta", by_step(task.partition)))["kv"]
        assert kv == {"sumw": str(chunk_sum), "kind": "static"}
        sums.append(kv["sumw"])
        assert read_part(os.path.join(tmp_path / "none", by_step(task.partition)))["kv"] is None
    assert len(set(sums)) == 2
    assert [kv for path, kv in CODEC_CALLS if "none" in Path(path).parts] == [None, None]


def test_part_path_is_destination_joined_with_name_of_the_tasks_partition(tmp_path: Path) -> None:
    _s, x, src = record(FILES)
    explicit = [Partition("fa", "t", 0, 5), Partition("fa", "t", 5, 12), Partition("fb", "t", 0, 6)]
    dest = tmp_path / "not" / "yet"
    plan = graphed.aggregate_plan(
        reduce=paths_only, combine=add, empty=no_paths, partitions=explicit, writes=[_write(x, dest, by_range)]
    )
    paths = SequentialRunner().run(plan).value
    assert paths == [os.path.join(dest, by_range(p)) for p in explicit]
    for p, path in zip(explicit, paths, strict=True):
        assert read_part(path)["values"] == src.chunk(p).tolist()

    blind_dest = tmp_path / "blind"
    plan = graphed.aggregate_plan(
        reduce=paths_only, combine=add, empty=no_paths, steps_per_file=2,
        writes=[_write(x, blind_dest, by_step_in_dir)],
    )
    tasks = sorted(plan.tasks, key=lambda t: t.key)
    assert all(t.partition.is_blind for t in tasks)
    paths = SequentialRunner().run(plan).value
    assert paths == [os.path.join(blind_dest, by_step_in_dir(t.partition)) for t in tasks]
    for t, path in zip(tasks, paths, strict=True):
        assert read_part(path)["values"] == src.chunk(t.partition).tolist()


def test_colliding_part_names_are_refused_before_any_task_runs(tmp_path: Path) -> None:
    _s, x, src = record(FILES)
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="write the same part"):
        graphed.aggregate_plan(
            reduce=paths_only, combine=add, empty=no_paths, steps_per_file=2, writes=[_write(x, dest, by_range)]
        )
    with pytest.raises(ValueError, match="write the same part"):
        graphed.aggregate_plan(
            reduce=paths_only, combine=add, empty=no_paths,
            partitions=[Partition("fa", "t", 0, 12)],
            writes=[_write(x, dest, fixed_name), _write(x * 2.0, dest, fixed_name)],
        )
    assert src.reads == []
    assert not dest.exists()
    assert CODEC_CALLS == []
    explicit = [Partition("fa", "t", 0, 6), Partition("fa", "t", 6, 12)]
    plan = graphed.aggregate_plan(
        reduce=paths_only, combine=add, empty=no_paths, partitions=explicit, writes=[_write(x, dest, by_range)]
    )
    assert len(SequentialRunner().run(plan).value) == 2


def test_write_root_reduction_refused_metadata_reduction_allowed(tmp_path: Path) -> None:
    _s, x, src = record(FILES)
    with pytest.raises(GraphedError, match="a partitioned write has no combine step"):
        graphed.aggregate_plan(
            reduce=paths_only, combine=add, empty=no_paths, writes=[_write(x.sum(), tmp_path / "bad")]
        )
    assert not (tmp_path / "bad").exists()
    plan = graphed.aggregate_plan(
        x.sum(), reduce=first_as_float, combine=add, empty=lambda: 0.0, steps_per_file=2,
        writes=[_write(x * 2.0, tmp_path / "ok", metadata={"sumx": x.sum()})],
    )
    assert SequentialRunner().run(plan).value == float(sum(v.sum() for v in FILES.values()))
    for task in plan.tasks:
        kv = read_part(os.path.join(tmp_path / "ok", by_step(task.partition)))["kv"]
        assert kv == {"sumx": str(src.chunk(task.partition).sum())}


def test_refusals(tmp_path: Path) -> None:
    s, x, _src = record(FILES)
    _other, y, _ = record(FILES)
    with pytest.raises(TypeError, match="one session"):
        graphed.aggregate_plan(x.sum(), reduce=first_as_float, combine=add, empty=zero, writes=[_write(y, tmp_path)])
    with pytest.raises(TypeError, match="one session"):
        graphed.aggregate_plan(
            x.sum(), reduce=first_as_float, combine=add, empty=zero,
            writes=[_write(x, tmp_path, metadata={"s": y.sum()})],
        )
    with pytest.raises(TypeError, match="does not capture writes"):
        graphed.aggregate_plan(
            x.sum(), reduce=first_as_float, combine=add, empty=zero,
            store=str(tmp_path / "store"), writes=[_write(x, tmp_path / "w")],
        )
    with pytest.raises(ValueError, match="at least one output"):
        graphed.aggregate_plan(reduce=paths_only, combine=add, empty=no_paths, writes=())

    total, doubled = x.sum(), x * 2.0
    compiled = compile_ir(s, total, doubled)
    store = GraphStore.deserialize(compiled.ir)
    kinds = {nid: store.nodes()[nid]["kind"] for nid in store.outputs()}
    (reduction,) = [nid for nid, kind in kinds.items() if kind == "reduction"]
    (row_local,) = [nid for nid in kinds if nid != reduction]
    refuse_chunk_partials(compiled, as_outputs={row_local})
    refuse_chunk_partials(compiled, as_outputs=frozenset())
    with pytest.raises(GraphedError, match="a partitioned write has no combine step"):
        refuse_chunk_partials(compiled, as_outputs={reduction})

    varied = graphed.vary(x, "jes", up=x * 1.5, down=x * 0.5)
    with pytest.raises(GraphedError, match="does not accept a Varied"):
        graphed.aggregate_plan(reduce=paths_only, combine=add, empty=no_paths, writes=[_write(varied, tmp_path / "v")])
    for meta in ({"s": varied}, {"s": [varied]}):
        with pytest.raises(GraphedError, match="does not accept a Varied"):
            graphed.aggregate_plan(
                reduce=paths_only, combine=add, empty=no_paths, writes=[_write(x, tmp_path / "v", metadata=meta)]
            )
    assert not (tmp_path / "v").exists()
