"""M39/M40 — ``graphed.shuffle``'s input-validation raises (plan guards in ``repartition``,
``join``, ``shuffle_plan``, ``join_plan``), plus the stage-1/stage-2 kernels called direct.

The frozen M39/M40 shape suites build only VALID plans, so these guard branches — wrong scheme
kwargs, a cross-session join, a graph missing its Exchange/Join boundary, the wrong partitioned-
source count — are never taken. Toy backend/source inlined rather than imported from
``tests/frozen/frontend/m39|m40/shuffle_backends.py``: those two modules share the top-level
basename ``shuffle_backends`` and collide when both sit on ``sys.path`` in one process, and
``tests/extra/frontend`` runs as a single pytest process (``scripts/run-tests.sh``).
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

import graphed
from graphed import Session
from graphed.backend import Form
from graphed.core import Partition, PayloadDescriptor
from graphed.errors import GraphedError
from graphed.shuffle import _GatherReduce, partition_block

Row = dict[str, int]
Block = list[Row]


def _route(key: int, parts: int) -> int:
    return int.from_bytes(hashlib.sha256(int(key).to_bytes(8, "big")).digest()[:8], "big") % parts


@dataclass(frozen=True)
class _ToyForm:
    """A minimal ``Form`` (just needs ``describe()``); the op name it was inferred from."""

    tag: str

    def describe(self) -> str:
        return self.tag


class ToyBackend:
    """Minimal ``Backend`` + ``ShuffleBackend`` (mirrors ``tests/frozen/frontend/m39/
    shuffle_backends.py``'s ``ToyBackend``, extended with the join-family ``op_form`` arms so a
    self-join records over it too)."""

    identity = "toy/0"

    def op_form(self, op: str, inputs: Sequence[Form], params: Mapping[str, object]) -> Form:
        if op in {"exchange", "join", "pack_key"}:
            return inputs[0] if inputs else _ToyForm(op)
        return _ToyForm(op)

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        return inputs[0]

    def boundary_ops(self) -> frozenset[str]:
        return frozenset({"source", "exchange", "join"})

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        return used

    def external_payload(self, op: str, params: Mapping[str, object]) -> PayloadDescriptor | None:
        return None

    def partition(
        self, block: Block, key_field: str, parts: int, *, salt: int = 0, boundaries: object = None
    ) -> tuple[Block, ...]:
        out: tuple[Block, ...] = tuple([] for _ in range(parts))
        for row in block:
            out[_route(row[key_field], parts)].append(row)
        return out

    def concat(self, blocks: Sequence[Block]) -> Block:
        merged: Block = []
        for b in blocks:
            merged.extend(b)
        return merged

    def slice_rows(self, block: Block, start: int, stop: int) -> Block:
        return block[start:stop]

    def estimated_bytes(self, block_or_form: object) -> int:
        return 16 * len(block_or_form) if isinstance(block_or_form, list) else 0

    def to_wire(self, block: Block) -> bytes:
        return pickle.dumps(block)

    def from_wire(self, data: bytes) -> Block:
        return cast(Block, pickle.loads(data))


@dataclass
class ListSource:
    """A ``PartitionedSource`` over an in-memory list of rows (copied from the m39 frozen toy)."""

    data: Block
    reads: list[tuple[int, int]] = field(default_factory=list)

    def __call__(self) -> Block:
        raise AssertionError("the whole-dataset loader must never run during a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("toy://list", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition: Any, columns: Any, resources: Any) -> Block:
        part = partition.resolve(len(self.data))
        self.reads.append((part.entry_start, part.entry_stop))
        return list(self.data[part.entry_start : part.entry_stop])


def _rows(n: int) -> Block:
    return [{"__joinkey__": i, "v": i} for i in range(n)]


# ---- _scheme_params / repartition: none of by=/n=/target_bytes= given -------------------------


def test_repartition_with_no_scheme_arg_raises_type_error() -> None:
    s = Session(ToyBackend())
    arr = s.source("x", form=_ToyForm("f"), data=ListSource(_rows(4)))
    with pytest.raises(TypeError, match=r"needs one of by=, n=, or target_bytes="):
        graphed.repartition(arr)


# ---- join: the two sides must belong to the same Session --------------------------------------


def test_join_across_two_sessions_raises_graphed_error() -> None:
    left = Session(ToyBackend()).source("left", form=_ToyForm("f"), data=ListSource(_rows(3)))
    right = Session(ToyBackend()).source("right", form=_ToyForm("f"), data=ListSource(_rows(3)))
    with pytest.raises(GraphedError, match="same Session"):
        graphed.join(left, right, on=["__joinkey__"])


# ---- shuffle_plan: needs a repartition Exchange in the graph -----------------------------------


def test_shuffle_plan_without_an_exchange_raises_type_error() -> None:
    s = Session(ToyBackend())
    arr = s.source("x", form=_ToyForm("f"), data=ListSource(_rows(4))).reduce("sum")
    with pytest.raises(TypeError, match="needs a repartition Exchange"):
        graphed.shuffle_plan(
            arr, reduce=lambda vs: vs[0], combine=lambda a, b: a, empty=lambda: None, backend=ToyBackend
        )


# ---- shuffle_plan: needs exactly one partitioned source ----------------------------------------


def test_shuffle_plan_with_two_partitioned_sources_raises_type_error() -> None:
    s = Session(ToyBackend())
    s.source("unused", form=_ToyForm("f"), data=ListSource(_rows(2)))  # 2nd, unreferenced source
    output = graphed.repartition(
        s.source("x", form=_ToyForm("f"), data=ListSource(_rows(4))), by="__joinkey__"
    ).reduce("sum")
    with pytest.raises(TypeError, match=r"exactly one partitioned source; this session has 2"):
        graphed.shuffle_plan(
            output, reduce=lambda vs: vs[0], combine=lambda a, b: a, empty=lambda: None, backend=ToyBackend
        )


# ---- join_plan: needs a Join boundary in the graph ----------------------------------------------


def test_join_plan_without_a_join_boundary_raises_type_error() -> None:
    s = Session(ToyBackend())
    output = graphed.repartition(
        s.source("x", form=_ToyForm("f"), data=ListSource(_rows(4))), by="__joinkey__"
    ).reduce("sum")
    with pytest.raises(TypeError, match="needs a Join boundary"):
        graphed.join_plan(output, backend=ToyBackend)


# ---- join_plan: needs exactly two partitioned sources (a self-join has only one) -----------------


def test_join_plan_self_join_has_one_partitioned_source_raises_type_error() -> None:
    s = Session(ToyBackend())
    a = s.source("a", form=_ToyForm("f"), data=ListSource(_rows(4)))
    joined = graphed.join(a, a, on=["__joinkey__"])
    with pytest.raises(TypeError, match=r"exactly two partitioned sources; this session has 1"):
        graphed.join_plan(joined, backend=ToyBackend)


# ---- _GatherReduce / partition_block: the stage-2/stage-1 kernels, never executed by graphed's
# own frozen suite (execution lives in graphed-exec-local) but unit-callable directly ------------


def test_gather_reduce_folds_reduce_over_combine_from_the_empty_identity() -> None:
    # __call__ wraps each gathered block as a singleton list before handing it to `reduce`
    # (the real stage-2 per-block-list reduce contract), so `reduce` here sums the numbers
    # inside that one-block list.
    gr = _GatherReduce(
        reduce=lambda blocks: sum(x for blk in blocks for x in blk),
        combine=lambda a, b: a + b,
        empty=lambda: 0,
    )
    assert gr([[1, 2], [3], [4, 5, 6]]) == 21, "each block reduces, then combines onto the empty identity"


def test_partition_block_delegates_to_the_backend_partition_primitive() -> None:
    block = _rows(6)
    out = partition_block(ToyBackend(), block, parts=3)
    assert len(out) == 3
    assert sum(len(p) for p in out) == 6, "every row lands in exactly one output partition"
