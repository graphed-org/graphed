"""Fixtures for the m58 frozen suite — projection honours what sources and Externals declare.

The `m58_` prefix is load-bearing: the frozen awkward tree runs one process per milestone dir, and
under prepend import mode a bare helper name binds to whichever sibling dir imported it first.

One dataset serves every driver. All values are dyadic rationals with small denominators, so every
sum, product and XOR delta below is exact in binary floating point and comparable bit-for-bit.
"""

from __future__ import annotations

from typing import Any

import awkward as ak

import graphed
import graphed.awkward as ga
from graphed import Array, Session
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward, gak
from graphed.core import Partition, PayloadDescriptor

DATA = ak.Array(
    [
        {"x": [1.0, 2.0], "y": [4.0], "z": 8.0},
        {"x": [3.0], "y": [2.0, 1.0], "z": 16.0},
        {"x": [0.5, 0.25], "y": [], "z": 32.0},
        {"x": [], "y": [8.0], "z": 64.0},
    ]
)

#: what a declaring source answers: UNSORTED, with a DUPLICATE, and naming a column ("z") no
#: driver's own computation reaches for the aggregate/parquet programs — so sorting, deduping,
#: reordering or ignoring the answer is visible at `read_partition`. A superset of every program's
#: needs, so the plans still evaluate.
DECLARED = ("z", "y", "x", "x")

#: the drivers' own answers on this dataset, with no hook in play (the H2/H5 regression pins)
AGGREGATE_COLUMNS = ("x", "y")
PARQUET_COLUMNS = ("x",)
VARIED_COLUMNS = ("y", "z")


class PlainSource:
    """A `graphed.write.PartitionedSource` over an in-memory awkward array that RECORDS the
    `columns` argument of every `read_partition` call — the witness for what the plan shipped."""

    def __init__(self, data: ak.Array = DATA) -> None:
        self.data = data
        self.seen: list[Any] = []

    def __call__(self) -> ak.Array:
        raise AssertionError("the whole-dataset loader must never run inside a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(
            Partition.blind("mem://m58", "", step, steps_per_file) for step in range(steps_per_file)
        )

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> ak.Array:
        self.seen.append(columns)
        part = partition.resolve(len(self.data))
        chunk = self.data[part.entry_start : part.entry_stop]
        if columns is None:
            return chunk
        return chunk[list(dict.fromkeys(columns))]


class DeclaringSource(PlainSource):
    """A source that DECLARES its own read list. Counts the calls and keeps the exact `outputs` it
    was handed; `max_calls` makes any further call loud, which is the worker-side witness."""

    def __init__(self, data: ak.Array = DATA, answer: Any = DECLARED, max_calls: int | None = None) -> None:
        super().__init__(data)
        self.answer = answer
        self.calls = 0
        self.outputs: tuple[Any, ...] = ()
        self.max_calls = max_calls

    def projected_columns(self, outputs: Any) -> list[str]:
        self.calls += 1
        if self.max_calls is not None and self.calls > self.max_calls:
            raise AssertionError(
                f"projected_columns called {self.calls}x: the built plan must carry the answer"
            )
        self.outputs = tuple(outputs)
        return list(self.answer)  # a LIST: the driver is the one that makes it a tuple


def partitioned(source: PlainSource) -> tuple[Session, Array]:
    """A session whose single source is `source`, recorded with the dataset's metadata-only form."""
    session = Session(AwkwardBackend())
    tracer = ak.Array(DATA.layout.to_typetracer(forget_length=True))
    root = session.source("events", form=AwkwardForm(tracer), data=source, uri="mem://m58")
    return session, root


# ---- the three drivers' programs (module level, so a process pool could import them) ------------
def counts(values: list[Any]) -> tuple[float, float]:
    return (float(ak.sum(values[0])), float(ak.sum(values[1])))


def add_counts(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
    return (left[0] + right[0], left[1] + right[1])


def no_counts() -> tuple[float, float]:
    return (0.0, 0.0)


#: `counts` over the whole dataset: 2+1+2+0 jagged x entries, 1+2+0+1 y entries
AGGREGATE_VALUE = (5.0, 4.0)
#: `root.x * 2.0`, the parquet program's written payload
PARQUET_VALUE = [[2.0, 4.0], [6.0], [1.0, 0.5], []]
#: the varied write's universes on the superset rows (`z > 8`)
VARIED_LABELS = ("murf_1", "murf_5em1", "nominal")
VARIED_UNIVERSES = {
    "nominal": [{"zz": 16.0, "w": 3.0}, {"zz": 32.0, "w": 0.0}, {"zz": 64.0, "w": 8.0}],
    "murf_1": [{"zz": 16.0, "w": 3.0}, {"zz": 32.0, "w": 0.0}, {"zz": 64.0, "w": 8.0}],
    "murf_5em1": [{"zz": 16.0, "w": 1.5}, {"zz": 32.0, "w": 0.0}, {"zz": 64.0, "w": 4.0}],
}


def aggregate_outputs(root: Array) -> tuple[Array, Array]:
    """Two outputs, so the hook's argument pins ORDER as well as identity."""
    return gak.num(root.x, axis=1), gak.num(root.y, axis=1)


def aggregate_over(source: PlainSource, *, steps: int = 2) -> tuple[Any, tuple[Array, Array]]:
    _session, root = partitioned(source)
    outputs = aggregate_outputs(root)
    plan = graphed.aggregate_plan(
        *outputs, reduce=counts, combine=add_counts, empty=no_counts, steps_per_file=steps
    )
    return plan, outputs


def varied_record(root: Array) -> tuple[Any, Any]:
    """(record, level-0 mask) for the varied write: a μR/μF weight family over a `{zz, w}` skim."""
    events = ga.gnano.events(root)
    weight = gak.sum(events.y, axis=1)
    ctx = graphed.vary(
        events, "murf", weight, is_weight=True, points={"1": weight, "0.5": weight * 0.5}
    )
    record = gak.zip({"zz": events.z, "w": graphed.weight(ctx)}, depth_limit=1)
    return record, events.z > 8.0


# ---- declared Externals (E) ---------------------------------------------------------------------
class PairArray(ak.Array):
    """A behavior class known to the backend's dict alone, never to global `ak.behavior`."""

    @property
    def total(self) -> Any:
        return self.a + self.b


BEHAVIOR: dict[Any, Any] = {("*", "m58pair"): PairArray}

#: one list level deeper than `DATA.x` — the op `gak.num(..., axis=2)` is valid on THIS form only
NESTED = ak.Array([[[1.0, 2.0], [3.0]], [[4.0]], [], [[5.0], []]])
#: a behavior-named record, one level SHALLOWER than `DATA.x`, carrying the `total` property
PAIRS = ak.with_name(ak.Array([{"a": 1.0, "b": 2.0}, {"a": 4.0, "b": 8.0}]), "m58pair")


def form_of(array: ak.Array) -> AwkwardForm:
    return AwkwardForm(ak.Array(array.layout.to_typetracer(forget_length=True)))


def never_runs(*inputs: Any) -> Any:
    raise AssertionError("projection is metadata-only: an External's evaluator must never run")


def declared_external(session: Session, inputs: list[Array], form: AwkwardForm, tag: str) -> Array:
    """The M23 seam: a package recording its OWN External family supplies descriptor AND form, so
    the node's recorded form is the caller's declaration, not the backend's input-shaped guess."""
    return session.record_external(
        "external",
        never_runs,
        inputs,
        {"kind": "m58"},
        descriptor=PayloadDescriptor(
            kind="m58",
            content_hash=f"sha256:{tag}",
            framework="m58",
            version="1",
            io_schema="array",
            preprocessing_ref=None,
        ),
        form=form,
    )


def external_session(*, behavior: Any = None) -> tuple[Session, Array]:
    session = Session(AwkwardBackend(behavior=behavior))
    return session, from_awkward(session, "events", DATA)
