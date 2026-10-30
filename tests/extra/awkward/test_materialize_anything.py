"""`graphed.materialize` evaluates every kind of deferred result, nested as it was passed."""

from __future__ import annotations

import functools
from collections import Counter
from collections.abc import Iterator, Mapping
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
from graphed.awkward.functions import firsts, local_index, mask, num, pad_none, where, with_name
from graphed.awkward.functions import sum as gak_sum
from graphed.awkward.io import from_parquet, parquet_write, to_parquet
from graphed.core import DurablePlan, OpSpec, Partition
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
            frame = dict(compiled.correspondence.frames).get((target, None), ("f.py", 1, "", ""))
            return (((target, None), (("nominal",), frame)),)

        return PlanPieces(
            outputs=(self.array,),
            reduce=lambda values: float(ak.sum(values[slot["at"]])),
            combine=lambda a, b: a + b,
            empty=lambda: 0.0,
            finalize=lambda total: total * self.scale,
            on_compiled=on_compiled,
        )


class _Counting(AwkwardBackend):
    """Counts each op's evaluations, and what it was asked to concatenate."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.evaluated: Counter[str] = Counter()
        self.concatenated: list[ak.Array] = []

    def eval_stage(self, op: str, inputs: Any, params: Any) -> Any:
        self.evaluated[op] += 1
        return super().eval_stage(op, inputs, params)

    def concat(self, blocks: Any) -> Any:
        self.concatenated.extend(blocks)
        return super().concat(blocks)


def _counted(*chunks: ak.Array) -> tuple[_Counting, Any]:
    backend = _Counting()
    return backend, _files(Session(backend), *chunks)[0]


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
            {k: v if k == offsets else _placeholder(v.dtype) for k, v in held.items()},
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
    assert caught.value.op.startswith("external")
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


def test_a_v1_durable_plan_runs_without_a_store() -> None:
    plan = DurablePlan(
        ir=b"v1",
        process=OpSpec("ref", "m:p", live=lambda p, r: float(p.entry_start)),
        combine=OpSpec("ref", "operator:add"),
        empty=OpSpec("ref", "builtins:float"),
        partitions=tuple(Partition("mem://v1", "Events", i, i + 1) for i in range(3)),
        finalize=OpSpec("ref", "builtins:str"),
    )
    assert graphed.materialize({"run": plan}) == {"run": "3.0"}


def test_a_source_read_list_is_asked_for_once_with_every_array_sharing_the_read() -> None:
    session = Session(AwkwardBackend())
    source, files = _files(session, *_flat())
    asked: list[int] = []

    def declare(arrays: tuple[Any, ...]) -> tuple[str, ...]:
        asked.append(len(arrays))
        return ("a", "b")

    files.projected_columns = declare
    done = graphed.materialize({"total": _Total(source.a), "array": source.b})
    assert done["total"] == 6.0
    assert ak.to_list(done["array"]) == [10.0, 20.0, 30.0]
    assert asked == [2]


def test_a_loaded_source_stays_loaded_for_the_arrays_beside_pieces() -> None:
    session = Session(AwkwardBackend())
    loads: list[int] = []

    def load() -> ak.Array:
        loads.append(1)
        return ak.Array({"x": [1.0, 2.0]})

    source = session.source("loaded", form=gak.AwkwardForm(load()), data=load)
    loads.clear()
    assert graphed.materialize([_Total(source.x), source.x + 1.0])[0] == 3.0
    assert loads == [1]


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


def test_pieces_over_whole_sources_share_the_read_of_a_pass_beside_them() -> None:
    session = Session(AwkwardBackend())
    source, files = _files(session, *_flat())
    memory = from_awkward("m", ak.Array({"x": [1.0, 2.0, 3.0]}), session=session)
    done = graphed.materialize(
        {"pass": _Total(source.a), "whole": _Total(source.b + memory.x), "array": source.c}
    )
    assert (done["pass"], done["whole"]) == (6.0, 66.0)
    assert ak.to_list(done["array"]) == [[1], [2, 3], [4]]
    assert [r[0] for r in files.reads] == [0, 1]
    assert {r[1] for r in files.reads} == {("a", "b", "c")}


def test_pieces_alone_beside_a_pass_share_its_read() -> None:
    session = Session(AwkwardBackend())
    source, files = _files(session, *_flat())
    memory = from_awkward("m", ak.Array({"x": [1.0, 2.0, 3.0]}), session=session)
    assert graphed.materialize([_Total(source.a), _Total(source.b + memory.x)]) == [6.0, 66.0]
    assert [r[0] for r in files.reads] == [0, 1]


def _held(buffer: Any) -> Any:
    return buffer


def _virtual(chunk: ak.Array, unread: str | None = None) -> ak.Array:
    """``chunk`` with every buffer loaded on first use, but field ``unread``'s a placeholder."""
    form, length, buffers = ak.to_buffers(chunk)
    skipped = None if unread is None else f"{form.content(unread).form_key}-data"
    return ak.from_buffers(
        form,
        length,
        {
            key: _placeholder(buffer.dtype) if key == skipped else functools.partial(_held, buffer)
            for key, buffer in buffers.items()
        },
    )


def test_concat_reads_virtual_buffers_rather_than_dropping_them() -> None:
    chunk = _virtual(ak.Array({"c": [[1], [2, 3]], "x": [1.0, 2.0]}))
    joined = AwkwardBackend().concat([chunk, chunk])
    assert ak.to_list(joined.x) == [1.0, 2.0, 1.0, 2.0]
    assert ak.to_list(joined.c) == [[1], [2, 3], [1], [2, 3]]


def test_concat_joins_virtual_offsets_around_an_unread_field() -> None:
    chunk = _virtual(ak.Array({"c": [[1], [2, 3]], "u": [1.0, 2.0]}), unread="u")
    joined = AwkwardBackend().concat([chunk, chunk])
    assert ak.to_list(joined.c) == [[1], [2, 3], [1], [2, 3]]
    assert joined.fields == ["c", "u"]


def test_concat_of_routed_blocks_keeps_what_was_read() -> None:
    chunk = ak.Array({"c": [[1], [2, 3], [4]], "x": [1.0, 2.0, 3.0], "u": [0.0, 0.0, 0.0]})
    block = _projected(chunk, ["c", "x"])
    keep, drop = np.array([True, False, True]), np.array([False, True, False])
    joined = AwkwardBackend().concat([block[keep], block[drop]])
    assert ak.to_list(joined.x) == [1.0, 3.0, 2.0]
    assert ak.to_list(joined.c) == [[1], [4], [2, 3]]
    assert joined.fields == ["c", "x", "u"]


def test_concat_keeps_a_union_around_its_unread_fields() -> None:
    left, right = (
        ak.zip({"p": [1.0, 2.0], "r": [4.0, 5.0]}),
        ak.zip({"p": [7.0, 8.0], "s": [1, 2]}),
    )
    chunk = ak.Array({"u": ak.where(np.array([True, False]), left, right), "x": [1.0, 2.0]})
    assert isinstance(chunk.layout.content("u"), ak.contents.UnionArray)
    block = _projected(chunk, ["u.p"])
    assert ak.to_list(AwkwardBackend().concat([block, block]).u.p) == [1.0, 8.0, 1.0, 8.0]


def test_concat_keeps_the_missing_rows_of_an_indexed_option() -> None:
    chunk = ak.Array(
        {"o": [{"p": 1.0, "u": 5.0}, None, {"p": 2.0, "u": 6.0}], "x": [1.0, 2.0, 3.0]}
    )
    assert isinstance(chunk.layout.content("o"), ak.contents.IndexedOptionArray)
    block = _projected(chunk, ["o.p"])
    joined = AwkwardBackend().concat([block, block])
    assert ak.to_list(joined.o.p) == [1.0, None, 2.0, 1.0, None, 2.0]


def test_pieces_over_whole_sources_fold_as_a_plan_does() -> None:
    class Folded(_Total):
        def __graphed_pieces__(self) -> PlanPieces:
            base = super().__graphed_pieces__()
            return PlanPieces(
                outputs=base.outputs,
                reduce=lambda values: [base.reduce(values)],
                combine=lambda total, partial: {"sum": total["sum"] + sum(partial)},
                empty=lambda: {"sum": 0.0},
                finalize=lambda total: total["sum"],
                on_compiled=base.on_compiled,
            )

    session = Session(AwkwardBackend())
    source, _ = _files(session, *_flat())
    memory = from_awkward("m", ak.Array({"x": [1.0, 2.0, 3.0]}), session=session)
    assert graphed.materialize([Folded(source.a), Folded(memory.x)]) == [6.0, 6.0]


def test_pieces_evaluate_on_the_sessions_own_backend() -> None:
    class Held(AwkwardBackend):
        def __init__(self, needed: int) -> None:
            super().__init__()
            self.needed = needed

    session = Session(Held(1))
    source, _ = _files(session, *_flat())
    assert graphed.materialize(_Total(source.a)) == 6.0


def test_a_refused_write_runs_nothing_beside_it(tmp_path: Any) -> None:
    session = Session(AwkwardBackend())
    source, _ = _files(session, *_flat())
    memory = from_awkward("m", ak.Array({"x": [1.0]}), session=session)
    written = tmp_path / "written"
    plan = to_parquet(source.b, str(tmp_path / "plan"), compute=False)
    with pytest.raises(TypeError, match="exactly one partitioned source"):
        graphed.materialize(
            [
                plan,
                parquet_write(source.a, str(written), name=_part_name),
                parquet_write(memory.x, str(tmp_path / "refused"), name=_part_name),
            ]
        )
    assert list(tmp_path.iterdir()) == []


def test_a_mapping_building_its_values_on_each_read() -> None:
    class Built(Mapping[str, Any]):
        def __init__(self, array: Any) -> None:
            self.array = array

        def __getitem__(self, key: str) -> Any:
            return self.array.a

        def __iter__(self) -> Iterator[str]:
            return iter(["a"])

        def __len__(self) -> int:
            return 1

    session = Session(AwkwardBackend())
    source, _ = _files(session, *_flat())
    assert ak.to_list(graphed.materialize(Built(source))["a"]) == [1.0, 2.0, 3.0]


def test_a_node_an_array_and_a_piece_share_runs_once_per_partition() -> None:
    backend, source = _counted(*_flat())
    shared = (source.a + 1.0) * 2.0
    total, normalized = graphed.materialize([_Total(shared), shared / gak_sum(shared, axis=None)])
    assert backend.evaluated["add"] == backend.evaluated["mul"] == 2

    _, alone = _counted(*_flat())
    alone_shared = (alone.a + 1.0) * 2.0
    assert total == graphed.materialize(_Total(alone_shared)) == 18.0
    assert ak.to_list(normalized) == ak.to_list(
        graphed.materialize(alone_shared / gak_sum(alone_shared, axis=None))
    )
    assert float(ak.sum(normalized)) == pytest.approx(1.0)


def test_a_piece_past_a_reduction_evaluates_whole_with_the_array_beside_it() -> None:
    backend, source = _counted(*_flat())
    index = local_index(source.a, axis=0)
    total, whole, doubled = graphed.materialize([_Total(index), index, _Total(index * 2)])
    assert backend.evaluated["ak.local_index"] == 1
    assert ak.to_list(whole) == [0, 1, 2]
    assert (total, doubled) == (3.0, 6.0)


def test_an_array_past_a_reduction_takes_the_node_under_it() -> None:
    backend, source = _counted(*_flat())
    shared = source.a + 1.0
    total, summed = graphed.materialize([_Total(shared), gak_sum(shared, axis=None)])
    assert total == summed == 9.0
    assert backend.evaluated["add"] == 2


def test_a_write_past_a_reduction_stays_refused(tmp_path: Any) -> None:
    session = Session(AwkwardBackend())
    source, _ = _files(session, *_flat())
    summed = source.a + gak_sum(source.a, axis=None)
    write = parquet_write(summed, str(tmp_path / "w"), name=_part_name)
    with pytest.raises(GraphedError, match="feeds another node"):
        graphed.materialize([write, source.b])


def test_an_array_of_one_column_keeps_only_that_column_of_the_pass() -> None:
    backend, source = _counted(*_flat())
    total, b = graphed.materialize([_Total(source.a), source.b])
    assert total == 6.0
    assert ak.to_list(b) == [10.0, 20.0, 30.0]
    assert backend.concatenated
    assert all(not block.fields for block in backend.concatenated)


def test_an_external_is_shared_as_row_wise() -> None:
    calls: list[int] = []

    def double(x: Any) -> Any:
        calls.append(len(x))
        return x * 2.0

    session = Session(AwkwardBackend())
    source, _ = _files(session, *_flat())
    shared = source.a.map(double, output_type="float64")
    total, whole = graphed.materialize([_Total(shared), shared])
    assert total == 12.0
    assert ak.to_list(whole) == [2.0, 4.0, 6.0]
    assert calls == [2, 1]


def test_a_source_with_no_partitions_leaves_arrays_to_its_loader() -> None:
    session = Session(AwkwardBackend())
    source, files = _files(session, *_flat())
    files.chunks = ()
    files.__class__ = type("Loaded", (_Files,), {"__call__": lambda self: _flat()[0]})
    total, a = graphed.materialize([_Total(source.a), source.a])
    assert total == 0.0
    assert ak.to_list(a) == [1.0, 2.0]


class _Doubling(ak.Array):  # type: ignore[misc]  # awkward is untyped here
    @property
    def doubled(self) -> ak.Array:
        return self.a * 2.0


def test_a_behavior_property_is_shared_as_row_wise() -> None:
    def chunks() -> list[ak.Array]:
        return [ak.Array({"p": ak.with_name(ak.zip({"a": x}), "P")}) for x in ([1.0, 2.0], [3.0])]

    backend = _Counting(behavior={("*", "P"): _Doubling})
    source, _ = _files(Session(backend), *chunks())
    doubled = source.p.doubled
    total, whole = graphed.materialize([_Total(doubled), doubled])
    assert backend.evaluated["field"] == 4  # `p` then `doubled`, once per partition
    assert total == 12.0
    assert ak.to_list(whole) == [2.0, 4.0, 6.0]


def test_a_length_one_row_set_refuses_in_a_pass_and_broadcasts_alone() -> None:
    session = Session(AwkwardBackend())
    source, _ = _files(session, *_flat())
    only = source.a - source.a[source.a == 3.0]
    assert ak.to_list(graphed.materialize(only)) == [-2.0, -1.0, 0.0]
    with pytest.raises(StageError, match="rows in one chunk"):
        graphed.materialize([_Total(source.a), only])


def test_lists_filtered_past_a_gap_keep_their_unread_content_unread() -> None:
    def counted(chunk: ak.Array) -> ak.Array:
        form, length, buffers = ak.to_buffers(chunk)
        unread = form.content("p").content.form_key
        return ak.from_buffers(
            form,
            length,
            {k: _placeholder(v.dtype) if unread in k else v for k, v in buffers.items()},
        )

    chunks = (ak.Array({"p": [[1.0], [2.0, 3.0], [4.0]], "a": [1.0, 2.0, 3.0]}),) * 2
    source, files = _files(Session(AwkwardBackend()), *chunks)
    files.reader = lambda partition: counted(chunks[int(partition.uri[1:])])
    kept = source.p[source.a != 2.0]
    total, index = graphed.materialize([_Total(source.a), local_index(kept, axis=0)])
    assert total == 12.0
    assert ak.to_list(index) == [0, 1, 2, 3]


def test_columns_of_one_selection_share_their_pass() -> None:
    backend, source = _counted(*_flat())
    kept = source[source.a > 1.0]
    together = kept.a + kept.b
    _, whole = graphed.materialize([_Total(together), gak_sum(together, axis=None)])
    assert float(whole) == 55.0
    assert backend.evaluated["add"] == 2


def _jets(pts: list[list[float]], keep: list[bool]) -> ak.Array:
    jets = ak.zip({"pt": pts, "eta": pts})
    return ak.Array({"jet": jets, "lep": ak.zip({"pt": pts, "q": pts}), "keep": keep, "a": keep})


_PARTLY_READ = {
    "ListArray": lambda e: e.jet[e.keep],
    "ByteMaskedArray": lambda e: mask(e.jet, e.keep),
    "RegularArray": lambda e: pad_none(e.jet, 1, axis=1, clip=True),
    "IndexedOptionArray": lambda e: firsts(e.jet),
    "UnionArray": lambda e: where(e.keep, e.jet, e.lep),
}


@pytest.mark.parametrize("build", _PARTLY_READ.values(), ids=_PARTLY_READ.keys())
def test_a_partly_read_value_keeps_what_was_read_across_chunks(build: Any) -> None:
    def named(source: Any) -> Any:
        return with_name(build(source), "Named").pt

    chunks = (_jets([[1.0, 2.0], [3.0]], [True, False]), _jets([[4.0], [5.0, 6.0]], [False, True]))
    source, files = _files(Session(AwkwardBackend()), *chunks)
    files.projected_columns = lambda outputs: ("jet.pt", "lep.pt", "keep", "a")
    alone, others = _files(Session(AwkwardBackend()), *chunks)
    others.projected_columns = files.projected_columns

    together = graphed.materialize([_Total(source.a), named(source)])[1]
    assert ak.to_list(together) == ak.to_list(graphed.materialize(named(alone)))


def test_a_bit_mask_keeps_what_was_read_across_chunks() -> None:
    layout = ak.contents.BitMaskedArray(
        ak.index.IndexU8(np.array([0b101], dtype=np.uint8)),
        ak.Array({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}).layout,
        valid_when=True,
        length=3,
        lsb_order=True,
    )
    block = _projected(ak.Array(layout), ["a"])
    assert ak.to_list(AwkwardBackend().concat([block, block]).a) == [1.0, None, 3.0] * 2


def test_an_unmasked_array_keeps_what_was_read_across_chunks() -> None:
    layout = ak.contents.UnmaskedArray(ak.Array({"a": [1.0], "b": [2.0]}).layout)
    block = _projected(ak.Array(layout), ["a"])
    assert ak.to_list(AwkwardBackend().concat([block, block]).a) == [1.0, 1.0]
