"""Toy backend + shuffle backend + source for the M39 frontend/engine suite (backend-agnostic).

Mirrors the M2 discipline (``tests/frozen/m2/backends.py``): graphed itself MUST NOT import
numpy/awkward (§A.4), so the frontend-recording and generic-engine tests run over a TOY
``ShuffleBackend`` whose blocks are plain lists of ``{"__joinkey__": int, ...}`` rows. The toy routes
by the SAME pinned §4 rule the real backends must, so it is a legitimate conformance reference; the
REAL awkward/numpy golden-vector + backend-independence witnesses live in graphed-awkward /
graphed-numpy / graphed-exec-local.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from graphed_core import Partition

Row = dict
Block = list  # a toy partition == a list of Row


def route(key: int, parts: int, *, salt: int = 0) -> int:
    """The pinned §4/§3.0 routing rule (salt=0 == no bytes appended)."""
    key_bytes = int(key).to_bytes(8, "big") + (salt.to_bytes(8, "big") if salt else b"")
    return int.from_bytes(hashlib.sha256(key_bytes).digest()[:8], "big") % parts


class ToyBackend:
    """A Session ``Backend`` AND a ``ShuffleBackend`` in one object (the M39 pattern — a backend is
    both). ``op_form(exchange)`` is identity; the exchange primitives operate on list-of-row blocks."""

    identity = "toy/0"

    # ---- Backend (record/fusion + reference eval) ----
    def op_form(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        if op == "exchange":
            return inputs[0]  # identity on the payload form (§3.3a)
        return op

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        return inputs[0]

    def boundary_ops(self) -> frozenset[str]:
        return frozenset({"source", "exchange"})

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        return used

    def external_payload(self, op: str, params: Mapping[str, object]) -> None:
        return None

    # ---- ShuffleBackend (exchange half, §3.0) ----
    def partition(
        self, block: Block, key_field: str, parts: int, *, salt: int = 0, boundaries=None
    ) -> tuple[Block, ...]:
        out: tuple[Block, ...] = tuple([] for _ in range(parts))
        for row in block:
            out[route(row[key_field], parts, salt=salt)].append(row)
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
        return pickle.loads(data)


@dataclass
class ListSource:
    """A PartitionedSource over an in-memory list of rows (the plan-builder toy)."""

    data: list
    reads: list = field(default_factory=list)

    def __call__(self) -> list:
        raise AssertionError("the whole-dataset loader must never run during a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("toy://list", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition, columns, resources) -> list:  # type: ignore[no-untyped-def]
        part = partition.resolve(len(self.data))
        self.reads.append((part.entry_start, part.entry_stop))
        return list(self.data[part.entry_start : part.entry_stop])
