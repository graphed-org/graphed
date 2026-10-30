"""`graphed.materialize` evaluates every kind of deferred result, nested as it was passed."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.placeholder import PlaceholderArray
from awkward._nplikes.shape import unknown_length

pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

import graphed
import graphed.awkward as gak
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.awkward.functions import num
from graphed.awkward.functions import sum as gak_sum
from graphed.awkward.io import from_parquet, parquet_write, to_parquet
from graphed.core import Partition
from graphed.core.execution import SequentialRunner
from graphed.errors import GraphedError, StageError
from graphed.materialization import PlanPieces, _Joint
from graphed.session import Session


def _placeholder(dtype: Any) -> PlaceholderArray:
    return PlaceholderArray(Numpy.instance(), (unknown_length,), dtype)


def _projected(chunk: ak.Array, columns: Any) -> ak.Array:
    """``chunk`` as a projected read leaves it: every buffer ``columns`` skip is a placeholder."""
    form, length, buffers = ak.to_buffers(chunk)
    kept = set()
    for column in columns:
        kept |= set(form.select_columns([column.split(".")]).expected_from_buffers())
    return ak.from_buffers(
        form,
        length,
        {
            key: buffer if key in kept else _placeholder(buffer.dtype)
            for key, buffer in buffers.items()
        },
    )


@dataclass
class _Files:
    """A partitioned source that counts its reads and hands back projected chunks."""

    chunks: tuple[ak.Array, ...]
    reads: list[tuple[int, tuple[str, ...]]] = field(default_factory=list)
    #: the source's own read list, as `declared_columns` asks for it
    projected_columns: Any = None
    #: stands in for the projected read
    reader: Any = None

    def partitions(self, steps_per_file: int) -> tuple[Partition, ...]:
        return tuple(Partition.blind(f"f{i}", "t", 0, 1) for i in range(len(self.chunks)))

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> Any:
        if self.reader is not None:
            return self.reader(partition)
        i = int(partition.uri[1:])
        read = self.chunks[i].fields if columns is None else columns
        self.reads.append((i, tuple(read)))
        return _projected(self.chunks[i], read)

    def __call__(self) -> ak.Array:
        raise AssertionError("the whole-dataset loader must stay unused")


def _files(session: Session, *chunks: ak.Array, name: str = "files") -> tuple[Any, _Files]:
    files = _Files(chunks)
    form = gak.AwkwardForm(ak.Array(chunks[0].layout.to_typetracer(forget_length=True)))
    return session.source(name, form=form, data=files), files


def _flat() -> tuple[ak.Array, ak.Array]:
    one = ak.Array({"a": [1.0, 2.0], "b": [10.0, 20.0], "c": [[1], [2, 3]]})
    return one, ak.Array({"a": [3.0], "b": [30.0], "c": [[4]]})


class _Total:
    """A deferred sum over the partitions, built on the hook a deferred class implements."""

    def __init__(self, array: Any, *, scale: float = 1.0) -> None:
        self.array = array
        self.scale = scale
        self.pieces_built = 0

    def __graphed_pieces__(self) -> PlanPieces:
        self.pieces_built += 1
        slot: dict[str, int] = {}

        def on_compiled(compiled: Any) -> tuple[Any, ...]:
            target = compiled.correspondence.node_map[self.array.node_id][0]
            store = graphed.core.GraphStore.deserialize(bytes(compiled.ir))
            slot["at"] = list(store.outputs()).index(target)
            return (((target, None), (("nominal",), ("f.py", 1, "", ""))),)

        return PlanPieces(
            outputs=(self.array,),
            reduce=lambda values: float(ak.sum(values[slot["at"]])),
            combine=lambda a, b: a + b,
            empty=lambda: 0.0,
            finalize=lambda total: total * self.scale,
            on_compiled=on_compiled,
        )


def _labelled(labels: tuple[str, ...], _: object) -> tuple[Any, ...]:
    return (((0, None), (labels, "frame")),)


def _part_name(partition: Partition) -> str:
    return f"{partition.uri}.parquet"


def test_two_projected_chunks_concatenate_around_their_unread_buffers() -> None:
    session = Session(AwkwardBackend())
    source, files = _files(session, *_flat())
    assert ak.to_list(graphed.materialize(source.a)) == [1.0, 2.0, 3.0]
    assert files.reads == [(0, ("a",)), (1, ("a",))]


def test_a_list_keeps_its_offsets_when_every_record_field_under_it_is_unread() -> None:
    jets = ak.Array([{"jet": [{"pt": 1.0, "eta": 2.0}]}, {"jet": []}])
    session = Session(AwkwardBackend())
    source, files = _files(session, jets, jets[:1])
    form, _, buffers = ak.to_buffers(jets)
    offsets = next(k for k in buffers if k.endswith("-offsets"))

    def read(partition: Partition) -> ak.Array:
        chunk = jets if partition.uri == "f0" else jets[:1]
        _, length, held = ak.to_buffers(chunk)
        return ak.from_buffers(
            form,
            length,
            {
                k: v
                if k == offsets
                else PlaceholderArray(Numpy.instance(), (unknown_length,), v.dtype)
                for k, v in held.items()
            },
        )

    files.reader = read
    assert ak.to_list(graphed.materialize(num(source.jet, axis=1))) == [1, 0, 1]


def test_a_partly_read_record_under_a_list_keeps_what_was_read() -> None:
    jets = ak.Array([{"jet": [{"pt": 1.0, "eta": 2.0}]}, {"jet": []}])
    session = Session(AwkwardBackend())
    source, files = _files(session, jets, jets[:1])
    files.projected_columns = lambda outputs: ("jet.pt",)
    assert ak.to_list(graphed.materialize(source.jet.pt)) == [[1.0], [], [1.0]]
    assert {columns for _, columns in files.reads} == {("jet.pt",)}


def test_an_all_read_chunk_beside_a_projected_one_concatenates_on_the_same_fields() -> None:
    session = Session(AwkwardBackend())
    source, files = _files(session, *_flat())
    one, two = files.chunks
    files.reader = lambda p: two[:0] if p.uri == "f1" else _projected(one, ["a"])
    assert ak.to_list(graphed.materialize(source.a)) == [1.0, 2.0]


def test_concat_keeps_unread_fields_under_their_behavior() -> None:
    jets = ak.with_name(ak.Array([{"pt": 1.0, "eta": 2.0}]), "Jet")
    blocks = [_projected(ak.Array({"jet": ak.unflatten(jets, [1])}), ["jet.pt"])] * 2
    joined = AwkwardBackend().concat(blocks)
    assert joined.jet.fields == ["pt", "eta"]
    assert ak.to_list(joined.jet.pt) == [[1.0], [1.0]]
    assert joined.jet.layout.content.parameter("__record__") == "Jet"


def test_concat_shifts_offsets_of_a_block_sliced_past_its_first_list() -> None:
    chunk = ak.Array({"c": [[1], [2, 3], [4]], "u": [1.0, 2.0, 3.0]})
    sliced = _projected(chunk, ["c"])[1:]
    joined = AwkwardBackend().concat([sliced, _projected(chunk, ["c"])])
    assert ak.to_list(joined.c) == [[2, 3], [4], [1], [2, 3], [4]]


def test_concat_of_tuples_keeps_them_tuples() -> None:
    chunk = ak.Array([(1.0, 2.0)])
    joined = AwkwardBackend().concat([_projected(chunk, ["0"])] * 2)
    assert joined.layout.is_tuple
    assert ak.to_list(joined["0"]) == [1.0, 1.0]


def test_an_unread_regular_list_stays_unread() -> None:
    chunk = ak.Array({"r": ak.to_regular(ak.Array([[1.0, 2.0]])), "x": [1.0]})
    joined = AwkwardBackend().concat([_projected(chunk, ["x"])] * 2)
    assert ak.to_list(joined.x) == [1.0, 1.0]
    assert joined.fields == ["r", "x"]


def test_a_failing_op_raises_a_stage_error_at_the_users_line() -> None:
    session = Session(AwkwardBackend())
    source = from_awkward("m", ak.Array({"x": [[1.0, 2.0]], "y": [[1.0]]}), session=session)
    bad = source.x + source.y
    with pytest.raises(StageError) as caught:
        graphed.materialize(gak_sum(bad, axis=1))
    assert caught.value.user_frame.filename == __file__
    assert caught.value.user_frame.lineno == bad.session.provenance(bad).lineno
    assert caught.value.cause_type == "ValueError"
    assert caught.value.__cause__ is not None


def test_a_failing_external_raises_a_stage_error() -> None:
    session = Session(AwkwardBackend())
    source = from_awkward("m", ak.Array([1.0]), session=session)

    def boom(*_: object) -> object:
        raise ValueError("no")

    out = graphed.apply(boom, source)
    with pytest.raises(StageError) as caught:
        graphed.materialize(out)
    assert caught.value.op == "external"
    assert caught.value.user_frame.lineno == out.session.provenance(out).lineno


def test_a_graphed_error_is_not_rewrapped() -> None:
    class Refusing(AwkwardBackend):
        def eval_stage(self, op: str, inputs: Any, params: Any) -> Any:
            raise GraphedError("already attributed")

    session = Session(Refusing())
    source = from_awkward("m", ak.Array([1.0]), session=session)
    with pytest.raises(GraphedError, match="already attributed"):
        graphed.materialize(source + 1.0)


def test_a_plan_runs_and_gives_a_runners_value(tmp_path: Any) -> None:
    session = Session(AwkwardBackend())
    source, _ = _files(session, *_flat())
    plan = to_parquet(source.a, str(tmp_path / "out"), compute=False)
    paths = graphed.materialize({"written": plan})["written"]
    assert [Path(p).name for p in paths] == ["part-00000.parquet", "part-00001.parquet"]
    assert ak.to_list(ak.concatenate([ak.from_parquet(p) for p in paths]).data) == [1.0, 2.0, 3.0]


def test_a_durable_plan_runs() -> None:
    session = Session(AwkwardBackend())
    source, _ = _files(session, *_flat())
    plan = graphed.shuffle_plan(
        graphed.repartition(source, by="a"),
        reduce=lambda values: float(ak.sum(values[0].a)),
        combine=lambda a, b: a + b,
        empty=lambda: 0.0,
    )
    assert graphed.materialize([plan]) == [SequentialRunner().run(plan).value] == [6.0]


def test_a_part_write_returns_its_part_paths(tmp_path: Any) -> None:
    session = Session(AwkwardBackend())
    source, files = _files(session, *_flat())
    write = parquet_write(
        source.a, str(tmp_path), name=_part_name, metadata={"n": gak_sum(source.a, axis=None)}
    )
    assert graphed.materialize((write,)) == (
        [str(tmp_path / "f0.parquet"), str(tmp_path / "f1.parquet")],
    )
    assert [r[0] for r in files.reads] == [0, 1]
    assert pq.read_schema(tmp_path / "f0.parquet").metadata[b"n"] == b"3.0"


def test_everything_in_one_call_shares_one_read_of_each_partition(tmp_path: Any) -> None:
    session = Session(AwkwardBackend())
    source, files = _files(session, *_flat())
    total = _Total(source.a)
    write = parquet_write(source.b, str(tmp_path), name=_part_name)
    done = graphed.materialize(
        {"array": [source.c], "total": total, "write": write, "n": 3, "total again": total}
    )
    assert ak.to_list(done["array"][0]) == [[1], [2, 3], [4]]
    assert done["total"] == done["total again"] == 6.0
    assert done["write"] == [str(tmp_path / "f0.parquet"), str(tmp_path / "f1.parquet")]
    assert done["n"] == 3
    assert total.pieces_built == 1
    assert [r[0] for r in files.reads] == [0, 1]
    assert {r[1] for r in files.reads} == {("a", "b", "c")}


def test_an_array_reduced_over_the_dataset_sees_every_partition_beside_a_deferred_sum() -> None:
    session = Session(AwkwardBackend())
    source, files = _files(session, *_flat())
    arrays = graphed.materialize([gak_sum(source.a, axis=None), _Total(source.a)])
    assert arrays == [6.0, 6.0]
    assert [r[0] for r in files.reads] == [0, 1]


def test_two_sources_run_a_pass_each() -> None:
    session = Session(AwkwardBackend())
    one, one_files = _files(session, *_flat(), name="one")
    two, two_files = _files(session, *_flat(), name="two")
    assert graphed.materialize([_Total(one.a), _Total(two.b)]) == [6.0, 60.0]
    assert len(one_files.reads) == len(two_files.reads) == 2


def test_sessions_are_independent() -> None:
    first, second = Session(AwkwardBackend()), Session(AwkwardBackend())
    one, _ = _files(first, *_flat())
    two, _ = _files(second, *_flat())
    assert graphed.materialize([_Total(one.a), _Total(two.a, scale=2.0)]) == [6.0, 12.0]


def test_an_in_memory_source_evaluates_a_deferred_result_once_over_the_whole_source() -> None:
    session = Session(AwkwardBackend())
    source = from_awkward("m", ak.Array({"x": [1.0, 2.0, 3.0]}), session=session)
    total = _Total(source.x)
    arrays = graphed.materialize({"t": total, "x": source.x * 2.0})
    assert arrays["t"] == 6.0
    assert ak.to_list(arrays["x"]) == [2.0, 4.0, 6.0]


def test_a_failure_over_a_whole_source_is_attributed() -> None:
    session = Session(AwkwardBackend())
    source = from_awkward("m", ak.Array({"x": [[1.0, 2.0]], "y": [[1.0]]}), session=session)
    bad = source.x + source.y
    with pytest.raises(StageError) as caught:
        graphed.materialize(_Total(bad))
    assert caught.value.user_frame.lineno == bad.session.provenance(bad).lineno


def test_a_write_needs_a_partitioned_source(tmp_path: Any) -> None:
    session = Session(AwkwardBackend())
    source = from_awkward("m", ak.Array({"x": [1.0]}), session=session)
    with pytest.raises(TypeError, match="exactly one partitioned source"):
        graphed.materialize(parquet_write(source.x, str(tmp_path), name=_part_name))


def test_a_varied_array_comes_back_per_label() -> None:
    session = Session(AwkwardBackend())
    source, files = _files(session, *_flat())
    varied = graphed.vary(source.a, "s", up=source.a * 2.0, down=source.a * 0.5)
    done = graphed.materialize({"v": varied})["v"]
    assert {k: ak.to_list(v) for k, v in done.items()} == {
        "nominal": [1.0, 2.0, 3.0],
        "s_up": [2.0, 4.0, 6.0],
        "s_down": [0.5, 1.0, 1.5],
    }
    assert len(files.reads) == 2


def test_pieces_with_no_outputs_give_their_empty_value_and_read_nothing() -> None:
    class Unfilled:
        def __graphed_pieces__(self) -> PlanPieces:
            return PlanPieces(
                outputs=(),
                reduce=list,
                combine=lambda a, _: a,
                empty=lambda: 0.0,
                finalize=lambda value: value + 1.0,
                on_compiled=lambda _: None,
            )

    assert graphed.materialize({"u": Unfilled()}) == {"u": 1.0}


def test_a_class_with_the_hook_passes_through() -> None:
    assert graphed.materialize([_Total]) == [_Total]


def test_real_parquet_files_share_a_pass(tmp_path: Any) -> None:
    for i, chunk in enumerate(_flat()):
        ak.to_parquet(chunk, tmp_path / f"{i}.parquet")
    session = Session(AwkwardBackend())
    source = from_parquet(
        "p", [str(tmp_path / "0.parquet"), str(tmp_path / "1.parquet")], session=session
    )
    done = graphed.materialize({"a": source.a, "t": _Total(source.b)})
    assert ak.to_list(done["a"]) == [1.0, 2.0, 3.0]
    assert done["t"] == 60.0


def test_pieces_labelling_one_key_carry_the_union_of_their_labels() -> None:
    session = Session(AwkwardBackend())
    source, _ = _files(session, *_flat())
    joint = _Joint()
    compiled: Any = None
    for i, labels in enumerate((("up",), ("down",))):
        joint.pieces[i] = PlanPieces(
            outputs=(source.a,),
            reduce=list,
            combine=lambda a, _: a,
            empty=list,
            finalize=list,
            on_compiled=functools.partial(_labelled, labels),
        )
    assert joint.on_compiled(compiled) == (((0, None), (("down", "up"), "frame")),)


def test_blocks_read_whole_concatenate_as_awkward_does() -> None:
    chunks = [ak.Array(np.arange(2.0)), ak.Array(np.arange(1.0))]
    assert ak.to_list(AwkwardBackend().concat(chunks)) == [0.0, 1.0, 0.0]


def test_a_runner_and_materialize_agree_on_a_part_write(tmp_path: Any) -> None:
    session = Session(AwkwardBackend())
    source, _ = _files(session, *_flat())
    plan = graphed.aggregate_plan(
        writes=[parquet_write(source.a, str(tmp_path), name=_part_name)],
        reduce=lambda values: list(values),
        combine=lambda a, b: [*a, *b],
        empty=list,
    )
    expected = SequentialRunner().run(plan).value
    assert graphed.materialize(parquet_write(source.a, str(tmp_path), name=_part_name)) == expected
