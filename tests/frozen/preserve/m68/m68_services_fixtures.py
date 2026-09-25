"""Builders shared by the frozen m68 suite: one partitioned in-memory awkward source (no pyarrow)
the shipped Triton External recorded by ``url=`` or ``service=`` over the copied fake transport, and a
non-Triton ``GENERIC`` External (the service surface is kind-agnostic).
Only 0.0.6 API is imported here, so the no-services pins collect on the base."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import awkward as ak
import numpy as np

from graphed import Session, aggregate_plan
from graphed.awkward import AwkwardBackend, AwkwardForm
from graphed.core import Partition, Plan, WorkerResources
from graphed.preserve import TRITON_PLUGIN, ExternalPlugin, record_external, register_plugin, sha256_bytes

X = np.array([0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0, 13.0])
TRANSPORT = "fake_triton_services:transport"
HIST = {"name": "x", "bins": 4, "lo": 0.0, "hi": 16.0}


def descriptor(tag: str, w: float = 0.45, b: float = -0.1) -> bytes:
    """A served-model descriptor; a distinct ``tag`` per scenario keeps connection caches apart."""
    return json.dumps(
        {"model": "scorer", "version": tag, "weights": {"w": w, "b": b}}, sort_keys=True
    ).encode()


def expected(w: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(w * X + b)))


class Chunks:
    """A ``PartitionedSource`` over ``X``; identity equality keeps plan processes comparable."""

    def __init__(self) -> None:
        self.data = ak.Array({"x": X})

    def partitions(self, steps_per_file: int) -> tuple[Partition, ...]:
        return tuple(Partition.blind("mem://events", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition: Partition, columns: Any, resources: WorkerResources) -> Any:
        part = partition.resolve(len(self.data))
        return self.data[part.entry_start : part.entry_stop]


def events(s: Session) -> Any:
    source = Chunks()
    form = AwkwardForm(ak.Array(source.data.layout.to_typetracer(forget_length=True)))
    return s.source("events", form=form, data=source)


def new_events() -> Any:
    register_plugin(TRITON_PLUGIN, validate=False)  # preserve/m9 re-registers the kind with its own plugin
    return events(Session(AwkwardBackend()))


def score(ev: Any, payload: bytes, **where: str) -> Any:
    params = {"model": "scorer", "transport": TRANSPORT, "input_name": "x", "output_name": "y", **where}
    return record_external(ev.session, TRITON_PLUGIN, payload, [ev.x], params=params)


def _first(resource: Any, params: Any, inputs: list[Any]) -> Any:
    return inputs[0]


GENERIC = ExternalPlugin(
    kind="m68_generic", content_hash=sha256_bytes, evaluate=_first, samples=lambda: [b"g"]
)


def mark(ev: Any, tag: str, service: str) -> Any:
    """A ``GENERIC`` External naming ``service``."""
    return record_external(ev.session, GENERIC, tag.encode(), [ev.x], params={"service": service})


def _rows(values: list[Any]) -> list[list[float]]:
    return [np.asarray(ak.to_numpy(ak.Array(v)), dtype="float64").tolist() for v in values]


def _cat(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [x + y for x, y in zip(a, b, strict=True)] if a else b


def values_plan(*outputs: Any, reduce: Any = _rows, services: Sequence[str] = ()) -> Plan[Any]:
    """One output row per output Array, over two partitions; ``services=`` only when given."""
    extra: dict[str, Any] = {"services": tuple(services)} if services else {}
    return aggregate_plan(*outputs, reduce=reduce, combine=_cat, empty=list, steps_per_file=2, **extra)
