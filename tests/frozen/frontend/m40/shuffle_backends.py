"""Fixtures for the M40 join frontend suite (plan §3.1-§3.3, theme (a)/(a2)/determinism).

Two kinds of fixture, mirroring the M39 discipline:

* ``REAL_BACKENDS`` — the two shipping ``ShuffleBackend`` implementations (``AwkwardBackend`` and
  ``NumpyBackend``) wrapped as :class:`BackendCase`s that know how to (i) build a flat record *source*
  from a column dict and (ii) normalize a *materialized* join block back to a canonical, comparable
  set of relational rows. The SAME join suite runs over BOTH cases (the M2 two-backend discipline,
  ``tests/frozen/frontend/m2/backends.py``): a primitive that leaked awkward-only semantics (jagged
  list-of-matches, native option) would fail on numpy, so green-on-both witnesses that the join half
  of the ``ShuffleBackend`` seam is real, not an awkward-shaped hole.
* ``ToyJoinBackend`` + ``ListSource`` — a backend-agnostic ``Backend``/``ShuffleBackend`` and a
  ``PartitionedSource`` (copied from ``tests/frozen/frontend/m39/shuffle_backends.py``, extended with
  the join-family ``op_form`` arms) for the ``join_plan`` **plan-shape** test, which asserts stage
  topology only and never executes blocks.

The relational payload columns (``njet``/``nmu``) are derived from a FIXED corpus skim
(``graphed_corpus.make_events(seed=1234)``, the M3/M5 fixture); the ``(run, lumi, event)`` keys are
synthesized with controlled multiplicity (the corpus records carry no event id) so the duplicating
semantics of a SQL/pandas inner join are actually exercised (a key with k left * m right matches
=> k*m output rows). See ``README.md``.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import awkward as ak
import numpy as np
import pandas as pd
from graphed_corpus import make_events

from graphed import Array, Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.core import Partition
from graphed.numpy import NumpyBackend, from_record

# The join key fields and the full relational output column set (union of both sides' fields minus the
# duplicated key), in a fixed order the normalizers project onto so internal column order is irrelevant.
ON: tuple[str, ...] = ("run", "lumi", "event")
COLUMNS: tuple[str, ...] = ("run", "lumi", "event", "njet", "nmu")


def route(key: int, parts: int, *, salt: int = 0) -> int:
    """The pinned §4/§3.0 routing rule (salt=0 == no bytes appended) — the reference the golden
    vectors pin; reused here only to keep the toy backend conformant."""
    key_bytes = int(key).to_bytes(8, "big") + (salt.to_bytes(8, "big") if salt else b"")
    return int.from_bytes(hashlib.sha256(key_bytes).digest()[:8], "big") % parts


# --------------------------------------------------------------------------------------------------
# The fixed corpus skim -> two flat relational tables with deliberate key multiplicity + orphans.
# --------------------------------------------------------------------------------------------------
def skim_tables() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Two flat record tables keyed on ``(run, lumi, event)``.

    Left keys (event): 10,10,10,20,20,30,88  -> key 10 x3, 20 x2, 30 x1, 88 orphan (no right match).
    Right keys (event): 10,10,20,30,30,99     -> key 10 x2, 20 x1, 30 x2, 99 orphan (no left match).
    Inner join => 10:3*2 + 20:2*1 + 30:1*2 = 6+2+2 = 10 rows across 3 distinct keys (duplication),
    with the 88/99 orphans dropped (the inner-join witness). Payloads (njet/nmu) come from the skim.
    """
    ev = make_events(n_events=24, seed=1234)
    njet = np.asarray(ak.num(ev.Jet.pt, axis=1)).astype(np.int64)
    nmu = np.asarray(ak.num(ev.Muon.pt, axis=1)).astype(np.int64)

    left_event = np.array([10, 10, 10, 20, 20, 30, 88], dtype=np.int64)
    right_event = np.array([10, 10, 20, 30, 30, 99], dtype=np.int64)
    left = {
        "run": np.ones(left_event.size, dtype=np.int64),
        "lumi": np.ones(left_event.size, dtype=np.int64),
        "event": left_event,
        "njet": njet[: left_event.size],
    }
    right = {
        "run": np.ones(right_event.size, dtype=np.int64),
        "lumi": np.ones(right_event.size, dtype=np.int64),
        "event": right_event,
        "nmu": nmu[: right_event.size],
    }
    return left, right


def pandas_reference(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray], *, how: str = "inner"
) -> list[tuple[int, ...]]:
    """The DUPLICATING baseline: a SQL/pandas ``merge`` (k*m rows per key), NOT a list-of-matches
    grouping. Returned as a sorted list of integer tuples over ``COLUMNS`` (a multiset of rows)."""
    ref = pd.merge(pd.DataFrame(dict(left)), pd.DataFrame(dict(right)), on=list(ON), how=how)
    return sorted(tuple(int(ref[c].iloc[i]) for c in COLUMNS) for i in range(len(ref)))


def _rows_from_records(records: Sequence[Mapping[str, Any]]) -> list[tuple[int, ...]]:
    return sorted(tuple(int(r[c]) for c in COLUMNS) for r in records)


# --------------------------------------------------------------------------------------------------
# Real-backend cases (the two-backend discipline).
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BackendCase:
    name: str
    make_backend: Callable[[], Any]
    make_source: Callable[[Session, str, Mapping[str, np.ndarray]], Array]
    to_rows: Callable[[Any], list[tuple[int, ...]]]


def _awkward_source(session: Session, name: str, cols: Mapping[str, np.ndarray]) -> Array:
    return from_awkward(session, name, ak.Array({k: np.asarray(v) for k, v in cols.items()}))


def _awkward_rows(block: Any) -> list[tuple[int, ...]]:
    return _rows_from_records(ak.to_list(block))


def _numpy_source(session: Session, name: str, cols: Mapping[str, np.ndarray]) -> Array:
    return from_record(session, name, **{k: np.asarray(v) for k, v in cols.items()})


def _numpy_rows(block: Any) -> list[tuple[int, ...]]:
    if isinstance(block, Mapping):
        n = len(block[COLUMNS[0]])
        return sorted(tuple(int(block[c][i]) for c in COLUMNS) for i in range(n))
    arr = np.asarray(block)
    if arr.dtype.names is not None:  # structured record array
        return sorted(tuple(int(arr[c][i]) for c in COLUMNS) for i in range(len(arr)))
    raise AssertionError(f"unrecognized numpy join block: {type(block)!r} dtype={arr.dtype!r}")


REAL_BACKENDS: tuple[BackendCase, ...] = (
    BackendCase("awkward", AwkwardBackend, _awkward_source, _awkward_rows),
    BackendCase("numpy", NumpyBackend, _numpy_source, _numpy_rows),
)


# --------------------------------------------------------------------------------------------------
# Toy backend + ListSource for the backend-agnostic join_plan shape test (mirror of m39).
# --------------------------------------------------------------------------------------------------
Row = dict
Block = list


class ToyJoinBackend:
    """A ``Backend`` AND ``ShuffleBackend`` in one object (the M39 pattern), extended with the M40
    join-family ``op_form`` arms so ``graphed.join`` records over it. Structure only — the join_plan
    shape test never evaluates blocks."""

    identity = "toy/0"

    def op_form(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        if op in {"exchange", "join", "pack_key"}:
            return inputs[0] if inputs else op  # identity on the payload form (§3.3a)
        return op

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        return inputs[0]

    def boundary_ops(self) -> frozenset[str]:
        return frozenset({"source", "exchange", "join"})

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        return used

    def external_payload(self, op: str, params: Mapping[str, object]) -> None:
        return None

    def partition(
        self, block: Block, key_field: str, parts: int, *, salt: int = 0, boundaries: object = None
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
    """A ``PartitionedSource`` over an in-memory list of rows (the plan-builder toy, from m39)."""

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


def toy_join_sources(session: Session) -> tuple[Array, Array]:
    """Two independently-partitioned toy sources carrying ``(run, lumi, event)`` keys."""
    left = session.source(
        "left", form="f", data=ListSource([{"run": 1, "lumi": 1, "event": i % 3} for i in range(9)])
    )
    right = session.source(
        "right", form="f", data=ListSource([{"run": 1, "lumi": 1, "event": i % 3} for i in range(6)])
    )
    return left, right
