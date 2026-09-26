"""Fixtures for awkward/m72. Module level so a built plan pickles; `m72_` prefixed so the bare
name is unique under prepend import."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from graphed import Array, Session
from graphed.awkward import AwkwardBackend, AwkwardForm, from_parquet
from graphed.core import Partition

#: float32 weights: no chunk sum below prints the same through `str(float(x))`
W = np.array([0.1, 0.2, 0.3, 0.7, 1.3, 0.05, 2.2, 0.4, 0.9, 0.15], dtype=np.float32)
EVENTS = ak.Array(
    {
        "w": W,
        "n": np.array([0, 2, 3, 1, 5, 2, 0, 4, 2, 3], dtype=np.int64),
        "pt": [[1.5, 2.0], [], [3.25], [4.0, 0.5, 1.0], [2.0], [], [7.0], [1.0, 1.0], [0.25], [9.0]],
    }
)


def write_input(path: Path, events: ak.Array = EVENTS) -> str:
    ak.to_parquet(events, str(path))
    return str(path)


def source(paths: str | list[str], steps: int = 1) -> Any:
    return from_parquet(Session(AwkwardBackend()), "ev", paths, steps_per_file=steps)


def by_step(p: Partition) -> str:
    return f"{Path(p.uri).stem}_{p.blind_step}.parquet"


def resolved_chunk(events: ak.Array, p: Partition) -> ak.Array:
    part = p.resolve(len(events))
    return events[part.entry_start : part.entry_stop]


def option_record(ev: Any, zip_: Any, mask: Any) -> Any:
    """?{num, jag, nest} with fields given unsorted; `zip_`/`mask` are gak's or ak's."""
    rec = zip_({"num": ev.w, "jag": ev.pt, "nest": zip_({"b": ev.n, "a": ev.w * 2})}, depth_limit=1)
    return mask(rec, ev.n > 1)


def dump_to_parquet(chunk: ak.Array, path: str, metadata: dict[str, str]) -> None:
    """The original's writer (HiggsDNA `dump_ak_array` / inclusive_processor.dump_to_parquet)."""
    table = ak.to_arrow_table(chunk, extensionarray=False)
    names = sorted(table.schema.names)
    table = pa.table([table.column(n) for n in names], names=names)
    if metadata:
        table = table.replace_schema_metadata({**metadata, **(table.schema.metadata or {})})
    pq.write_table(table, path)


# ---- reduce / combine / empty ------------------------------------------------------------------
def paths_only(values: list[Any]) -> list[str]:
    return [v for v in values if isinstance(v, str)]


def no_paths() -> list[str]:
    return []


def add(a: Any, b: Any) -> Any:
    return a + b


def zero() -> int:
    return 0


def first_int(values: list[Any]) -> int:
    return int(values[0])


def mc_counts(values: list[Any]) -> tuple[float, int]:
    return float(values[0]), int(values[1])


def data_counts(values: list[Any]) -> tuple[int]:
    return (int(values[0]),)


def add_tuples(a: tuple[Any, ...], b: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def mc_zero() -> tuple[float, int]:
    return 0.0, 0


def data_zero() -> tuple[int]:
    return (0,)


def json_codec(value: object, path: str, kv: Any) -> None:
    with open(path, "w") as handle:
        json.dump({"kv": None if kv is None else dict(kv), "values": ak.to_list(value)}, handle, sort_keys=True)


def part_bytes(root: Path) -> dict[str, bytes]:
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            full = Path(dirpath) / name
            out[str(full.relative_to(root))] = full.read_bytes()
    return out


class RecordingSource:
    """A `PartitionedSource` over an in-memory record array that records every `columns` argument."""

    def __init__(self, data: ak.Array) -> None:
        self.data = data
        self.seen: list[Any] = []

    def __call__(self) -> ak.Array:
        raise AssertionError("the whole-dataset loader must never run inside a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("mem://m72", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> ak.Array:
        self.seen.append(columns)
        chunk = resolved_chunk(self.data, partition)
        return chunk if columns is None else chunk[list(dict.fromkeys(columns))]


def recorded(src: RecordingSource) -> Array:
    session = Session(AwkwardBackend())
    tracer = ak.Array(src.data.layout.to_typetracer(forget_length=True))
    return session.source("events", form=AwkwardForm(tracer), data=src, uri="mem://m72")

