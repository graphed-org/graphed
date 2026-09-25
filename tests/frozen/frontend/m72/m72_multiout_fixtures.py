"""Awkward-free fixtures for the m72 frontend suite (the free-threaded job collects
`tests/frozen/frontend` with only numpy installed). Everything a plan ships is module level, so a
spawned worker can import it."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import graphed
from graphed import Array, Session
from graphed.core import Partition
from graphed.numpy import NumpyBackend, NumpyForm

#: op name -> eval_stage calls in THIS process
EVALS: Counter[str] = Counter()
#: (path, kv) per codec call in THIS process
CODEC_CALLS: list[tuple[str, Any]] = []
#: every list `spy_reduce` received in THIS process
SEEN: list[list[Any]] = []


def reset() -> None:
    EVALS.clear()
    CODEC_CALLS.clear()
    SEEN.clear()


class CountingBackend(NumpyBackend):
    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        EVALS[op] += 1
        return super().eval_stage(op, inputs, params)


@dataclass
class FileSource:
    """A `PartitionedSource` over in-memory "files" (uri -> 1-D array), recording every read."""

    files: dict[str, np.ndarray]
    tree: str = "t"
    reads: list[tuple[str, str, int, int]] = field(default_factory=list)

    def __call__(self) -> np.ndarray:
        raise AssertionError("the whole-dataset loader must never run inside a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(
            Partition.blind(uri, self.tree, step, steps_per_file)
            for uri in self.files
            for step in range(steps_per_file)
        )

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> np.ndarray:
        if partition.tree != self.tree or partition.uri not in self.files:
            raise AssertionError(f"routed to the wrong source: {partition}")
        data = self.files[partition.uri]
        part = partition.resolve(len(data))
        self.reads.append((part.uri, part.tree, part.entry_start, part.entry_stop))
        return data[part.entry_start : part.entry_stop]

    def chunk(self, partition: Partition) -> np.ndarray:
        data = self.files[partition.uri]
        part = partition.resolve(len(data))
        return data[part.entry_start : part.entry_stop]


def record(files: dict[str, np.ndarray], *, tree: str = "t") -> tuple[Session, Array, FileSource]:
    src = FileSource(files, tree)
    dtype = next(iter(files.values())).dtype
    session = Session(CountingBackend())
    return session, session.source("x", form=NumpyForm(dtype, shape=(None,)), data=src), src


def json_codec(value: object, path: str, kv: Mapping[str, str] | None) -> None:
    """The toy codec: one JSON document per part; byte-deterministic for equal input."""
    CODEC_CALLS.append((path, None if kv is None else dict(kv)))
    doc = {"kv": None if kv is None else dict(kv), "values": np.asarray(value).tolist()}
    with open(path, "w") as handle:
        json.dump(doc, handle, sort_keys=True)


def read_part(path: str) -> dict[str, Any]:
    with open(path) as handle:
        doc: dict[str, Any] = json.load(handle)
    return doc


def by_step(p: Partition) -> str:
    return f"{p.uri}-{p.tree}-{p.blind_step}.json"


def by_step_in_dir(p: Partition) -> str:
    return os.path.join(p.uri, f"step{p.blind_step}.json")


def by_range(p: Partition) -> str:
    return f"{p.uri}-{p.tree}-{p.entry_start}-{p.entry_stop}.json"


def fixed_name(p: Partition) -> str:
    return "same.json"


# ---- reduce / combine / empty (module level: they ship) -------------------------------------
def spy_reduce(values: list[Any]) -> int:
    SEEN.append(list(values))
    return 0


def add(a: Any, b: Any) -> Any:
    return a + b


def zero() -> int:
    return 0


def paths_only(values: list[Any]) -> list[str]:
    return [v for v in values if isinstance(v, str)]


def no_paths() -> list[str]:
    return []


def first_as_float(values: list[Any]) -> float:
    return float(values[0])


def mc_reduce(values: list[Any]) -> tuple[int, list[int], list[str]]:
    """(sum x, [sum 2x] per chunk, paths): an int sum and two list concats — exact monoids."""
    return int(values[0]), [int(values[1])], [v for v in values[2:] if isinstance(v, str)]


def mc_combine(a: tuple[int, list[int], list[str]], b: tuple[int, list[int], list[str]]) -> tuple[int, list[int], list[str]]:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def mc_empty() -> tuple[int, list[int], list[str]]:
    return 0, [], []


def data_reduce(values: list[Any]) -> tuple[int, list[str]]:
    return int(values[0]), [v for v in values[1:] if isinstance(v, str)]


def data_combine(a: tuple[int, list[str]], b: tuple[int, list[str]]) -> tuple[int, list[str]]:
    return a[0] + b[0], a[1] + b[1]


def data_empty() -> tuple[int, list[str]]:
    return 0, []


MC_FILES = {"mc-a": np.arange(0, 12, dtype=np.int64), "mc-b": np.arange(100, 108, dtype=np.int64)}
#: `mc-a` again, under another tree: routing must key on (uri, tree), not uri
DATA_FILES = {"mc-a": np.arange(1000, 1010, dtype=np.int64), "data-b": np.arange(50, 56, dtype=np.int64)}


def mc_plan(steps: int, destination: str | None = None) -> tuple[Any, FileSource]:
    """The "MC" graph: a weight-like reduction (`2x`) beside the plain sum, optionally writing `2x`."""
    _s, x, src = record(MC_FILES, tree="Events")
    w = x * 2
    writes: tuple[Any, ...] = ()
    if destination is not None:
        writes = (graphed.write.PartWrite(array=w, destination=destination, name=by_step, codec=json_codec, metadata={"sumw": w.sum()}),)
    plan = graphed.aggregate_plan(
        x.sum(), w.sum(), reduce=mc_reduce, combine=mc_combine, empty=mc_empty, steps_per_file=steps, writes=writes
    )
    return plan, src


def data_plan(steps: int, destination: str | None = None) -> tuple[Any, FileSource]:
    """The "data" graph: no weight, a static KV."""
    _s, x, src = record(DATA_FILES, tree="Runs")
    writes: tuple[Any, ...] = ()
    if destination is not None:
        writes = (graphed.write.PartWrite(array=x, destination=destination, name=by_step, codec=json_codec, metadata={"kind": "Data"}),)
    plan = graphed.aggregate_plan(
        x.sum(), reduce=data_reduce, combine=data_combine, empty=data_empty, steps_per_file=steps, writes=writes
    )
    return plan, src


def tree_fold(combine: Callable[[Any, Any], Any], partials: list[Any]) -> Any:
    """Unseeded pairwise fold of key-ordered partials (never touches `empty()`)."""
    level = list(partials)
    while len(level) > 1:
        paired = [combine(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        level = paired + ([level[-1]] if len(level) % 2 else [])
    return level[0]


def part_bytes(root: str) -> dict[str, bytes]:
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            full = os.path.join(dirpath, name)
            with open(full, "rb") as handle:
                out[os.path.relpath(full, root)] = handle.read()
    return out


# ---- hand-built plans (collate takes any fixed-task Plan) -------------------------------------
def echo_uri(partition: Partition, resources: Any) -> list[str]:
    return [f"{partition.uri}:{partition.entry_start}"]


def concat(a: list[str], b: list[str]) -> list[str]:
    return a + b


def nothing() -> list[str]:
    return []
