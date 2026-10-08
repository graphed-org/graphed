"""Importable pieces of the m74 suite: processes, codecs and builders that the tests and the ``python``
scripts beside them share. Spies are module state, so the objects that read them pickle by name."""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
import subprocess
import sys
import threading
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from graphed.core import DurablePlan, DurablePlanV2, GraphStore, OpSpec, Partition, Plan, StageSpec, Task

HERE = Path(__file__).resolve().parent
T = 6
#: process calls in this interpreter; a call numbered ``interrupt_at`` raises :class:`Interrupt`
STATE: dict[str, int] = {"calls": 0, "interrupt_at": 0}


class Interrupt(BaseException):
    """A driver death: escapes every ``except Exception``."""


def reset(interrupt_at: int = 0) -> None:
    STATE.update(calls=0, interrupt_at=interrupt_at)


def tick() -> None:
    STATE["calls"] += 1
    if STATE["calls"] == STATE["interrupt_at"]:
        raise Interrupt(f"interrupted at call {STATE['calls']}")


def partitions(n: int = T, uri: str = "mem://m74/events") -> tuple[Partition, ...]:
    return tuple(Partition(uri, "Events", 10 * i, 10 * i + 10) for i in range(n))


def plan_of(process: Any, parts: Sequence[Partition] | None = None) -> Plan[Any]:
    parts = partitions() if parts is None else parts
    return Plan(process=process, combine=add, empty=zero, tasks=tuple(Task(i, p) for i, p in enumerate(parts)))


def plain(p: Partition) -> Partition:
    return Partition(**{f.name: getattr(p, f.name) for f in dataclasses.fields(Partition)})


def add(a: Any, b: Any) -> Any:
    return a + b


def zero() -> float:
    return 0.0


def marked(p: Partition, r: Any) -> float:
    tick()
    return 1.5 * p.entry_start + 1.0


@dataclasses.dataclass(frozen=True)
class Marker:
    """Appends one line per call to ``path``."""

    path: str

    def __call__(self, p: Partition, r: Any) -> float:
        tick()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{p.entry_start}\n")
        return 1.5 * p.entry_start + 1.0


def marks(path: str | os.PathLike[str]) -> list[str]:
    return Path(path).read_text(encoding="utf-8").split() if Path(path).exists() else []


@dataclasses.dataclass(frozen=True)
class Binned:
    bins: int

    def __call__(self, p: Partition, r: Any) -> float:
        tick()
        return float(p.entry_start % self.bins)


class LockedProc:
    """Importable, with a public class-level lock and a class-level cache each call fills."""

    LOCK = threading.Lock()
    CACHE: ClassVar[dict[int, int]] = {}

    def __call__(self, p: Partition, r: Any) -> float:
        with LockedProc.LOCK:
            LockedProc.CACHE[p.entry_start] = p.entry_stop
        return float(p.entry_start)


def raising(p: Partition, r: Any) -> float:
    raise ValueError(f"bad partition {p.entry_start}")


# ---- wrapper fidelity (T5) ------------------------------------------------------------------------
#: every call a Recorder saw, as ``(type(partition), partition, endpoint)``
RECEIVED: list[tuple[type, Partition, str]] = []


@dataclasses.dataclass(frozen=True)
class Recorder:
    endpoint: str = ""

    def __call__(self, p: Partition, r: Any) -> float:
        RECEIVED.append((type(p), p, self.endpoint))
        return float(p.entry_start)

    def bind_services(self, endpoints: dict[str, str]) -> Recorder:
        return dataclasses.replace(self, endpoint=endpoints["svc"])

    def resolve_services(self, value: Any) -> Any:
        return ("resolved", value)


# ---- threads (T6) ---------------------------------------------------------------------------------
#: the barrier the current T6 leg's calls hold on
BARRIER: list[threading.Barrier] = []


class Barriered:
    def __call__(self, p: Partition, r: Any) -> float:
        BARRIER[0].wait()
        return float(p.entry_start)


# ---- codecs (T9) ----------------------------------------------------------------------------------
#: ``(pid, thread ident)`` of every CountingCodec decode
DECODES: list[tuple[int, int]] = []


class CountingCodec:
    def encode(self, value: Any) -> bytes:
        return pickle.dumps(value, protocol=5)

    def decode(self, data: bytes) -> Any:
        DECODES.append((os.getpid(), threading.get_ident()))
        return pickle.loads(data)


# ---- live partials (T10) --------------------------------------------------------------------------
LIVE: dict[str, int] = {"now": 0, "peak": 0}


class Partial:
    """A partial that counts its live instances."""

    def __init__(self, v: float) -> None:
        self.v = v
        LIVE["now"] += 1
        LIVE["peak"] = max(LIVE["peak"], LIVE["now"])

    def __del__(self) -> None:
        LIVE["now"] -= 1

    def __reduce__(self) -> tuple[Any, ...]:
        return (Partial, (self.v,))


def partial_of(p: Partition, r: Any) -> Partial:
    return Partial(float(p.entry_start))


def add_partials(a: Partial, b: Partial) -> Partial:
    return Partial(a.v + b.v)


def no_partial() -> Partial:
    return Partial(0.0)


def leaf_tuple(p: Partition, r: Any) -> tuple[int, ...]:
    return (p.entry_start // 10,)


def concat(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    return a + b


def no_tuple() -> tuple[int, ...]:
    return ()


def leaf_float(p: Partition, r: Any) -> float:
    return 1e16 if p.entry_start == 0 else 1.0


# ---- durable plans --------------------------------------------------------------------------------
def ir_bytes() -> bytes:
    g = GraphStore()
    src = g.add_source("events", {"uri": "mem://m74/events"})
    xchg = g.add_exchange([g.add_op("key", [src])], {"scheme": "hash", "key": "__joinkey__", "parts": 2})
    return g.serialize(outputs=[g.add_reduction("sum", [xchg])])


def durable(process: str, combine: str = "m74_helpers:add", empty: str = "m74_helpers:zero", n: int = T) -> DurablePlan:
    return DurablePlan(
        ir=ir_bytes(),
        process=OpSpec.from_ref(process),
        combine=OpSpec.from_ref(combine),
        empty=OpSpec.from_ref(empty),
        partitions=partitions(n),
    )


def stage_map(task: Task, inputs: tuple[bytes, ...], resources: Any) -> bytes:
    tick()
    return f"mw:{task.key}".encode()


def stage_gather(task: Task, inputs: tuple[bytes, ...], resources: Any) -> bytes:
    tick()
    return b"gj:" + str(task.key).encode() + b"|" + b",".join(sorted(inputs))


def staged(n_map: int = 4, n_dest: int = 2) -> DurablePlanV2:
    return DurablePlanV2(
        ir=ir_bytes(),
        stages=(
            StageSpec(
                kind="map_write",
                process=OpSpec.from_ref("m74_helpers:stage_map"),
                routing={"scheme": "hash", "key": "__joinkey__", "parts": n_dest, "backend_id": "toy/0"},
                tasks=tuple(Task(i, Partition("src", "Events", i, i + 1)) for i in range(n_map)),
            ),
            StageSpec(
                kind="gather_join",
                inputs=(0,),
                process=OpSpec.from_ref("m74_helpers:stage_gather"),
                routing={"parts": n_dest, "backend_id": "toy/0"},
                tasks=tuple(Task(d, Partition("dest", "p", d, d + 1)) for d in range(n_dest)),
            ),
        ),
    )


def stage_reused(plan: DurablePlanV2) -> list[int]:
    return [stage.process.resolve().reused for stage in plan.stages]


# ---- refusals (T4) --------------------------------------------------------------------------------
#: one entry per pickling of a Served
PICKLED: list[str] = []


class Served:
    """Stands in for a histserv ``_Served``: tasks return receipts for server state, and pickling one
    would create that state."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __call__(self, p: Partition, r: Any) -> Any:
        return self.inner(p, r)

    def __reduce__(self) -> tuple[Any, ...]:
        PICKLED.append("served")
        return (Served, (self.inner,))

    def bind_services(self, endpoints: dict[str, str]) -> Served:
        return self


# ---- an importable object that resolves by name and whose reduce carries a uuid (T2 xiv) ----------
def _kernel(tag: str) -> Kernel:
    return KERNEL


class Kernel:
    def __init__(self) -> None:
        self.__qualname__ = "KERNEL"

    def __call__(self, x: float) -> float:
        return x * 2.0

    def __reduce__(self) -> tuple[Any, ...]:
        return (_kernel, (uuid.uuid4().hex,))


KERNEL = Kernel()


# ---- awkward ---------------------------------------------------------------------------------------
#: a behavior dict a backend import ref names; ``__main__`` mixins register into it
behavior: dict[Any, Any] = {}


def make_backend() -> Any:
    from graphed.awkward import AwkwardBackend  # noqa: PLC0415

    return AwkwardBackend(behavior=behavior)


def events(directory: str | os.PathLike[str]) -> list[str]:
    """Two 50-row parquet files with float ``x`` and int ``y`` (written once per directory)."""
    import awkward as ak  # noqa: PLC0415

    paths = []
    for i in range(2):
        path = os.path.join(directory, f"events-{i}.parquet")
        if not os.path.exists(path):
            ak.to_parquet(ak.Array({"x": [3.0 * k + i for k in range(50)], "y": [k % 5 for k in range(50)]}), path)
        paths.append(path)
    return paths


def agg_reduce(values: list[Any]) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def agg_combine(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def agg_empty() -> tuple[float, ...]:
    return (0.0,)


class StopReduce:
    """A gather ``reduce`` that counts its calls (and so can be interrupted)."""

    def __call__(self, values: list[Any]) -> tuple[float, ...]:
        tick()
        return tuple(float(v) for v in values)


def agg_plan(paths: Sequence[str]) -> Plan[Any]:
    from graphed import Session, aggregate_plan  # noqa: PLC0415
    from graphed.awkward import AwkwardBackend, gak  # noqa: PLC0415
    from graphed.awkward.io import from_parquet  # noqa: PLC0415

    ev = from_parquet(Session(AwkwardBackend()), "events", list(paths), open_files=False, steps_per_file=2)
    return aggregate_plan(gak.sum((ev.x + ev.y) * 2), reduce=agg_reduce, combine=agg_combine, empty=agg_empty)


def _sf_load(payload: bytes, params: Any) -> str:
    return "sf"


def _sf_scale(handle: str, params: Any, inputs: list[Any]) -> Any:
    return inputs[0] * 1.25


def _sf_samples() -> list[bytes]:
    return [b"sf-v1"]


def service_events(paths: Sequence[str]) -> Any:
    """``x`` scaled by an External that names the service ``sf``."""
    from graphed import Session  # noqa: PLC0415
    from graphed.awkward import AwkwardBackend  # noqa: PLC0415
    from graphed.awkward.io import from_parquet  # noqa: PLC0415
    from graphed.preserve import ExternalPlugin, record_external, sha256_bytes  # noqa: PLC0415
    from graphed.services import ServiceSpec  # noqa: PLC0415

    plugin = ExternalPlugin(
        kind="m74_sf", content_hash=sha256_bytes, load=_sf_load, evaluate=_sf_scale, samples=_sf_samples
    )
    s = Session(AwkwardBackend())
    s.declare_service(ServiceSpec("sf", "http"))
    ev = from_parquet(s, "events", list(paths), open_files=False, steps_per_file=2)
    return record_external(s, plugin, b"sf-v1", [ev.x], params={"service": "sf"})


# ---- subprocesses ---------------------------------------------------------------------------------
def run_script(
    script: str, *args: str, seed: str = "1", path: Sequence[str | os.PathLike[str]] = (), timeout: float = 180.0
) -> dict[str, Any]:
    """Run ``python <script> <args>`` from this directory and return its last stdout line as JSON; ``path``
    goes ahead of this directory on ``PYTHONPATH``."""
    full = {**os.environ, "PYTHONHASHSEED": seed}
    inherited = [p for p in full.get("PYTHONPATH", "").split(os.pathsep) if p]
    full["PYTHONPATH"] = os.pathsep.join([*map(str, path), str(HERE), *inherited])
    done = subprocess.run(
        [sys.executable, str(HERE / script), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=full,
        cwd=HERE,
        check=False,
    )
    lines = done.stdout.strip().splitlines()
    if done.returncode != 0 or not lines:
        raise AssertionError(f"{script} {args} exited {done.returncode}:\n{done.stdout}\n{done.stderr}")
    report: dict[str, Any] = json.loads(lines[-1])
    return report


def fake_dist(root: str | os.PathLike[str], name: str, version: str) -> Path:
    """A directory holding ``<name>-<version>.dist-info`` with a ``METADATA`` naming it."""
    where = Path(root) / f"{name}-{version}"
    info = where / f"{name}-{version}.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")
    return where


def snapshot(root: str | os.PathLike[str]) -> dict[str, bytes]:
    """Every file under a directory store, by relative path."""
    base = Path(root)
    return {str(p.relative_to(base)): p.read_bytes() for p in sorted(base.rglob("*")) if p.is_file()}


def journal_records(root: str) -> dict[str, list[dict[str, Any]]]:
    """``{journal name: records}`` for a directory ``Store`` or an ``FsspecStore`` URL."""
    from fsspec.core import url_to_fs  # noqa: PLC0415

    fs, base = url_to_fs(root)
    out: dict[str, list[dict[str, Any]]] = {}
    for journal in fs.glob(f"{base}/journal*.log"):
        if fs.isdir(journal):
            raw = [fs.cat_file(obj).decode() for obj in sorted(fs.find(journal))]
        else:
            raw = fs.cat_file(journal).decode().splitlines()
        out[journal.rsplit("/", 1)[-1]] = [json.loads(line) for line in raw if line.strip()]
    return out
