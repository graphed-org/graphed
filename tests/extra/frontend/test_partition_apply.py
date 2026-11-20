"""``apply``/``map`` with ``partition=True`` hand the function its chunk's partition (#285)."""

from __future__ import annotations

import functools
import hashlib
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
import graphed.awkward as ga
import graphed.debug as gd
from graphed import Session
from graphed.aggregate import aggregate_plan
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward, gak
from graphed.core import Partition
from graphed.core.execution import LocalResources, Plan, SequentialRunner, WorkerResources
from graphed.errors import GraphedTypeError, StageError
from graphed.numpy import NumpyBackend
from graphed.numpy.io import from_parquet, to_parquet
from graphed.preserve.externals._base import ExternalPlugin, record_external
from graphed.systematics.varied import Varied, narrow

FILES = {
    f"mem://sample/kl_{k}/f.root": ak.Array({"x": [1.0] * n}) for k, n in ((1, 5), (2, 3), (5, 4))
}


@dataclass
class Files:
    """A multi-file source whose partitions are blind steps over each file."""

    def __call__(self) -> ak.Array:
        return ak.concatenate(list(FILES.values()))

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(
            Partition.blind(uri, "", s, steps_per_file)
            for uri in FILES
            for s in range(steps_per_file)
        )

    def num_entries(self, partition: Partition) -> int:
        return len(FILES[partition.uri])

    def read_partition(
        self, partition: Partition, columns: Any, resources: WorkerResources
    ) -> ak.Array:
        part = partition.resolve(len(FILES[partition.uri]))
        return FILES[partition.uri][part.entry_start : part.entry_stop]


def kl_and_start(x: Any, *, partition: Partition) -> Any:
    """Per row: the kl the file path names, and the chunk's first entry."""
    kl = float(re.search(r"kl_(\d+)", partition.uri).group(1))  # type: ignore[union-attr]
    return ak.zip({"kl": x * 0 + kl, "start": x * 0 + partition.entry_start})


def _source(session: Session) -> graphed.Array:
    form = AwkwardForm(ak.Array(Files()().layout.to_typetracer(forget_length=True)))
    return session.source("events", form=form, data=Files())


def _plan(steps: int) -> Plan[Any]:
    session = Session(AwkwardBackend())
    out = graphed.apply(
        kl_and_start,
        _source(session).x,
        name="kl",
        output_type="{kl: float64, start: float64}",
        partition=True,
    )
    plan = aggregate_plan(
        out,
        reduce=lambda values: ak.Array(values[0]),
        combine=lambda a, b: ak.concatenate([a, b]),
        empty=lambda: ak.Array({"kl": [], "start": []}),
        steps_per_file=steps,
    )
    assert isinstance(plan, Plan)
    return plan


def _eager(steps: int) -> list[dict[str, float]]:
    rows = []
    for uri, data in FILES.items():
        for s in range(steps):
            part = Partition.blind(uri, "", s, steps).resolve(len(data))
            rows += kl_and_start(
                data.x[part.entry_start : part.entry_stop], partition=part
            ).tolist()
    return rows


def _task(process: Any, partition: Partition) -> Any:
    return process(partition, LocalResources())


def test_each_partition_reads_its_own_file_constant_as_the_eager_loop_does() -> None:
    got = SequentialRunner().run(_plan(2)).value.tolist()
    assert got == _eager(2)
    assert {row["kl"] for row in got} == {1.0, 2.0, 5.0}
    # the resolved entry range reached the function: the second step of a 5-row file starts at 2
    assert {row["start"] for row in got} == {0.0, 1.0, 2.0}


def test_spawned_workers_hand_each_task_its_partition() -> None:
    plan = _plan(2)
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
        parts = list(
            pool.map(functools.partial(_task, plan.process), [t.partition for t in plan.tasks])
        )
    assert functools.reduce(plan.combine, parts, plan.empty()).tolist() == _eager(2)


def test_the_request_is_node_identity_and_the_ir_is_deterministic() -> None:
    def ir(partition: bool) -> bytes:
        session = Session(AwkwardBackend())
        x = _source(session).x
        fn = kl_and_start if partition else (lambda x: ak.zip({"kl": x, "start": x}))
        out = graphed.apply(
            fn, x, name="kl", output_type="{kl: float64, start: float64}", partition=partition
        )
        return bytes(graphed.compile_ir(session, out).ir)

    assert ir(True) == ir(True)
    assert ir(True) != ir(False)
    assert b"partition" in ir(True) and b"partition" not in ir(False)


def test_a_whole_source_evaluation_refuses_at_the_recording_line() -> None:
    session = Session(AwkwardBackend())
    out = _source(session).x.map(
        kl_and_start, output_type="{kl: float64, start: float64}", partition=True
    )
    with pytest.raises(StageError, match="reads its input partition"):
        session.materialize(out)


def _refused(make: Any, match: str) -> None:
    session = Session(AwkwardBackend())
    with pytest.raises(GraphedTypeError, match=match):
        graphed.apply(kl_and_start, make(session), output_type="float64", partition=True)


def test_no_declared_type_is_refused() -> None:
    session = Session(AwkwardBackend())
    with pytest.raises(GraphedTypeError, match="needs output_type="):
        graphed.apply(kl_and_start, _source(session).x, partition=True)


def test_a_partition_past_a_repartition_is_refused() -> None:
    _refused(lambda s: _source(s).x.repartition(n=2), "after a repartition")


def test_a_partition_past_a_join_is_refused() -> None:
    def joined(session: Session) -> Any:
        ev = _source(session)
        keyed = gak.zip({"x": ev.x})
        return graphed.join(keyed, keyed, on=["x"]).x

    _refused(joined, "after a repartition and a join")


def test_a_partition_past_a_whole_dataset_reduction_is_refused() -> None:
    def reduced(session: Session) -> Any:
        x = _source(session).x
        return x - gak.sum(x)

    _refused(reduced, "after a reduction")


def test_a_partition_over_two_sources_is_refused() -> None:
    session = Session(AwkwardBackend())
    form = AwkwardForm(ak.Array(Files()().layout.to_typetracer(forget_length=True)))
    a, b = _source(session).x, session.source("other", form=form, data=Files()).x
    with pytest.raises(
        GraphedTypeError,
        match=r"reads one source's partition; its inputs read \['events', 'other'\]",
    ):
        graphed.apply(
            lambda p, q, *, partition: p,
            gak.sum(a),
            gak.sum(b),
            output_type="float64",
            partition=True,
        )


def test_a_varied_operand_fans_out_one_partition_reader_per_universe() -> None:
    session = Session(AwkwardBackend())
    x = _source(session).x
    varied = graphed.vary(x, "s", points={"up": x * 2.0})
    out = graphed.apply(
        kl_and_start,
        varied,
        name="kl",
        output_type="{kl: float64, start: float64}",
        partition=True,
    )
    assert isinstance(out, Varied)
    nominal, up = narrow(out, "nominal"), narrow(out, "s_up")
    assert nominal.node_id != up.node_id
    assert all(b"partition" in bytes(graphed.compile_ir(session, m).ir) for m in (nominal, up))


def _mapped(session: Session) -> Any:
    return _source(session).x.map(
        kl_and_start, name="kl", output_type="{kl: float64, start: float64}", partition=True
    )


def test_one_partition_evaluates_as_its_task_does() -> None:
    plan = _plan(2)
    task = plan.tasks[1]
    want = plan.process(task.partition, LocalResources()).tolist()
    session = Session(AwkwardBackend())
    out = _mapped(session)
    assert want == [{"kl": 1.0, "start": 2.0}] * 3
    assert graphed.materialize(out, partition=task.partition).tolist() == want
    assert gd.run(session, out, partition=task.partition).tolist() == want  # type: ignore[attr-defined]
    with pytest.raises(StageError, match="reads its input partition"):
        gd.run(session, out)
    replayed = list(gd.replay(plan, 1).steps())
    assert replayed[-1].value.tolist() == want


def test_a_plugin_param_named_partition_is_no_request() -> None:
    plugin = ExternalPlugin(
        kind="doubler",
        content_hash=lambda b: hashlib.sha256(b).hexdigest(),
        evaluate=lambda _resource, _params, inputs: inputs[0] * 2,
        samples=lambda: [b"x"],
    )
    x = from_awkward("x", ak.Array([1.0, 2.0]))
    y = record_external(x.session, plugin, b"x", [x], params={"partition": "train"})
    assert graphed.materialize(y).tolist() == [2.0, 4.0]


def test_a_numpy_write_hands_the_resolved_blind_range(tmp_path: Any) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    src = tmp_path / "kl_3.parquet"
    pq.write_table(pa.table({"x": np.zeros(4)}), src)
    session = Session(NumpyBackend())
    x = from_parquet("events", str(src), steps_per_file=2, session=session)["x"]
    starts = x.map(
        lambda v, *, partition: v + partition.entry_start, output_type="float64", partition=True
    )
    parts = to_parquet(starts, str(tmp_path / "out"))
    assert isinstance(parts, list)
    written = [pq.read_table(p)["data"].to_pylist() for p in sorted(parts)]
    assert written == [[0.0, 0.0], [2.0, 2.0]]


class Uncounted(Files):
    """A source whose reader cannot count a file's entries."""

    num_entries = None  # type: ignore[assignment]


def test_a_reader_that_cannot_count_hands_the_blind_partition_as_it_is() -> None:
    def blind_step(x: Any, *, partition: Partition) -> Any:
        assert partition.is_blind
        return x * 0 + float(partition.blind_step or 0)

    session = Session(AwkwardBackend())
    form = AwkwardForm(ak.Array(Files()().layout.to_typetracer(forget_length=True)))
    ev = session.source("events", form=form, data=Uncounted())
    out = graphed.apply(blind_step, ev.x, name="step", output_type="float64", partition=True)
    plan = aggregate_plan(
        out,
        reduce=lambda values: ak.Array(values[0]),
        combine=lambda a, b: ak.concatenate([a, b]),
        empty=lambda: ak.Array([]),
        steps_per_file=2,
    )
    assert isinstance(plan, Plan)
    assert sorted(set(SequentialRunner().run(plan).value.tolist())) == [0.0, 1.0]


def kl_of(x: Any, *, partition: Partition) -> Any:
    """Per row: the kl the file path names."""
    return kl_and_start(x, partition=partition).kl


def _kls(steps: int) -> list[float]:
    return [row["kl"] for row in _eager(steps)]


def _written(parts: Any) -> list[Any]:
    assert isinstance(parts, list)
    pytest.importorskip("pyarrow")
    return [row for p in sorted(parts) for row in ak.from_parquet(p).tolist()]


def test_an_awkward_write_hands_each_part_its_partition(tmp_path: Any) -> None:
    pytest.importorskip("pyarrow")
    session = Session(AwkwardBackend())
    kl = _source(session).x.map(kl_of, output_type="float64", partition=True)
    parts = ga.to_parquet(kl, str(tmp_path / "out"), steps_per_file=2, column="kl")
    assert [row["kl"] for row in _written(parts)] == _kls(2)


def test_a_varied_awkward_write_hands_each_part_its_partition(tmp_path: Any) -> None:
    pytest.importorskip("pyarrow")
    session = Session(AwkwardBackend())
    x = _source(session).x
    record = gak.zip({"x": x, "kl": x.map(kl_of, output_type="float64", partition=True)})
    keep = graphed.vary(x > 0.0, "s", points={"up": x > -1.0})
    parts = ga.to_parquet(record, str(tmp_path / "out"), select=keep, steps_per_file=2)
    assert [row["kl"] for row in _written(parts)] == _kls(2)


def test_the_map_side_of_a_repartition_hands_each_task_its_partition() -> None:
    session = Session(AwkwardBackend())
    kl = _source(session).x.map(kl_of, output_type="float64", partition=True)
    plan = graphed.shuffle_plan(
        kl.repartition(n=2),
        reduce=lambda values: ak.Array(values[0]),
        combine=lambda a, b: ak.concatenate([a, b]),
        empty=lambda: ak.Array([]),
        steps_per_file=2,
    )
    assert sorted(SequentialRunner().run(plan).value.tolist()) == sorted(_kls(2))
