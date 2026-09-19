"""Fixtures for the m59 frozen suite — awkward-idiom parity at the frontend.

The `m59_` prefix is load-bearing: the frozen awkward tree runs one process per milestone dir, and
under prepend import mode a bare helper name binds to whichever sibling dir imported it first.

Every value is a dyadic rational with a small denominator, so every asserted product, sum and
widened float32 below is exact in binary floating point and comparable bit-for-bit. Eager awkward
is the ORACLE throughout: `eager_form` runs the same key/operator on a typetracer of the same data
(what a recorded form must equal) and the real array answers the values.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np

import graphed.core
from graphed import Array, Session
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward
from graphed.core import Partition

#: four events of `{x: var * float64, y: var * float64}`, NO empty list: `[:, 0]` and `[:, -1]`
#: are defined on every row, so an I1 failure is about the key and never about raggedness.
DATA = ak.Array(
    [
        {"x": [1.0, 2.0, 4.0], "y": [8.0, 0.5]},
        {"x": [16.0, 32.0], "y": [0.25]},
        {"x": [0.125], "y": [64.0, 2.0, 1.0]},
        {"x": [0.5, 0.25], "y": [128.0]},
    ]
)

#: the empty row I5's runtime index error needs — the typetracer cannot see it, so `[:, 0]` records
#: and fails only when real data reaches the op
RAGGED = ak.Array([[1.0, 2.0], [], [4.0]])

#: the tuple keys that leave the partitioned axis whole (I1), by their spelling
ACCEPTED: dict[str, Any] = {
    "[:, :2]": (slice(None), slice(None, 2)),
    "[:, 0]": (slice(None), 0),
    "[:, -1]": (slice(None), -1),
    "[:, ::2]": (slice(None), slice(None, None, 2)),
    "[:, :, None]": (slice(None), slice(None), None),
    "[:, None, :]": (slice(None), None, slice(None)),
    "[..., 0]": (Ellipsis, 0),
}

#: tuple keys whose FIRST member consumes or restructures the partitioned axis (I3)
CONSUMES_AXIS0: dict[str, Any] = {
    "[1:3, 0]": (slice(1, 3), 0),
    "[0, :]": (0, slice(None)),
    "[None, :]": (None, slice(None)),
}

#: tuple keys refused for what a member IS, whatever axis it sits on (I3)
MALFORMED: dict[str, Any] = {
    "[:, True]": (slice(None), True),
    "[:, 1.5]": (slice(None), 1.5),
    "[:, 0:2.5]": (slice(None), slice(0, 2.5)),
    "[..., ..., 0]": (Ellipsis, Ellipsis, 0),
}

BOOLS = ak.Array([[True, False], [True], [False, True, True]])
INTS = ak.Array([[1, 2], [4], [8, 16, 32]])
FLOATS = ak.Array([[1.0, 2.0], [4.0], [8.0, 16.0, 32.0]])

#: S1's array side, by dtype name
ARRAYS: dict[str, ak.Array] = {"bool": BOOLS, "int64": INTS, "float64": FLOATS}

#: S1's scalar side. `np.uint64(1 << 3)` is the shape `PackedSelection` packs bits with.
SCALARS: dict[str, Any] = {
    "uint64": np.uint64(8),
    "int32": np.int32(2),
    "float32": np.float32(0.5),
    "bool_": np.bool_(True),
}

#: S3's side: Python operands, which must keep recording exactly as they do today
PY_SCALARS: dict[str, Any] = {"int": 2, "float": 0.5, "bool": True}


def tracer(array: ak.Array) -> ak.Array:
    """The metadata-only view of `array` — what a recorded form is inferred on."""
    return ak.Array(array.layout.to_typetracer(forget_length=True))


def eager_form(value: Any) -> str:
    """The form description eager awkward gives for a typetracer expression."""
    return str(ak.Array(value).type)


def dtype_of(value: Any) -> str:
    """A value's leaf dtype name — the half of the type that a scalar operand decides."""
    return str(ak.Array(value).type).rsplit("* ", 1)[-1]


def recorded(session: Session, array: Array) -> dict[str, Any]:
    """The (kind, name, params) of the node `array` denotes, read back from the serialized IR: the
    house route to the boundary flag (`kind`), as m13 uses it."""
    graph = graphed.core.GraphStore.deserialize(session.serialized_ir(array, optimize=False))
    return next(node for node in graph.nodes() if node["id"] == array.node_id)


def session_over(data: ak.Array = DATA) -> tuple[Session, Array]:
    session = Session(AwkwardBackend())
    return session, from_awkward(session, "events", data)


# ---- the partitioned run (I2) -------------------------------------------------------------------
class PlainSource:
    """A `graphed.write.PartitionedSource` over `DATA` that records the `columns` of every read."""

    def __init__(self, data: ak.Array = DATA) -> None:
        self.data = data
        self.seen: list[Any] = []

    def __call__(self) -> ak.Array:
        raise AssertionError("the whole-dataset loader must never run inside a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(
            Partition.blind("mem://m59", "", step, steps_per_file) for step in range(steps_per_file)
        )

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> ak.Array:
        self.seen.append(columns)
        part = partition.resolve(len(self.data))
        chunk = self.data[part.entry_start : part.entry_stop]
        if columns is None:
            return chunk
        return chunk[list(dict.fromkeys(columns))]


def partitioned(source: PlainSource) -> tuple[Session, Array]:
    """A session whose single source is `source`, recorded with the dataset's metadata-only form."""
    session = Session(AwkwardBackend())
    root = session.source("events", form=AwkwardForm(tracer(DATA)), data=source, uri="mem://m59")
    return session, root


# the plan's reduce/combine/empty (module level, so the closure ships through pickle): one entry per
# partition carrying that partition's own type AND rows, so a partitioned run is compared to an
# unpartitioned one on structure as well as values.
def rows(values: list[Any]) -> list[Any]:
    value = ak.Array(values[0])
    return [(str(value.type).split("* ", 1)[-1], ak.to_list(value))]


def merge(left: list[Any], right: list[Any]) -> list[Any]:
    return left + right


def nothing() -> list[Any]:
    return []


def concatenated(value: list[Any]) -> list[Any]:
    return [row for _type, part in value for row in part]


# ---- behavior methods (M) -----------------------------------------------------------------------
class PairArray(ak.Array):
    """A behavior class known to `BEHAVIOR` alone, never to global `ak.behavior`."""

    def nested(self) -> Any:
        """M1's motivating shape: `(metric, (a, b))` — a tuple with a tuple inside it."""
        return self.a * 2.0, (self.b + 1.0, self.a + self.b)

    def flat(self) -> Any:
        return self.a * 2.0, self.b + 1.0

    def one(self) -> Any:
        return self.a + self.b

    def leafy(self) -> Any:
        """M3: the same nesting as `nested`, with a non-array leaf in the inner tuple."""
        return self.a * 2.0, (self.b + 1.0, 42)


BEHAVIOR: dict[Any, Any] = {("*", "m59pair"): PairArray}

PAIRS = ak.with_name(ak.Array([{"a": 1.0, "b": 2.0}, {"a": 4.0, "b": 8.0}]), "m59pair")


def pair_session() -> tuple[Session, Array]:
    session = Session(AwkwardBackend(behavior=BEHAVIOR))
    return session, from_awkward(session, "pairs", PAIRS)


def eager_pairs() -> ak.Array:
    """The eager, behavior-carrying counterpart of the `pair_session` source."""
    return ak.Array(PAIRS.layout, behavior=BEHAVIOR)


def tracer_pairs() -> ak.Array:
    """Its metadata-only counterpart — the oracle for a recorded leaf form."""
    return ak.Array(PAIRS.layout.to_typetracer(forget_length=True), behavior=BEHAVIOR)
