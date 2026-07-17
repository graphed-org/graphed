"""The serializable, versioned, content-addressed durable **Plan** (plan M8).

This is the "serializable physical artifact for an executor" the glossary calls *Plan* — forecast by
``execution.Plan``'s docstring ("the real serializable Plan is M8"). ``execution.Plan`` is the
in-memory runtime job (live callables); ``DurablePlan`` is its durable form: it can be written
to bytes, content-addressed, shipped, and re-executed **on a machine with no user source files**.

Canonical durable form (plan A.3.1). The computation is carried as the **serialized IR** (from
``graphed_core.GraphStore.serialize``), never cloudpickle — *except* for genuinely opaque user
callables that cannot be expressed in the IR, which are embedded **by value** (cloudpickle) and
flagged ``opaque=True`` as a preservation risk so M9 can surface them. Everything else in the plan
(partitions, read columns, reduction/stopping/locality/resource metadata) is plain data.

Determinism (plan M8 gate). ``DurablePlan.to_bytes`` is a single canonical JSON document
(sorted keys, no insignificant whitespace) with binary blobs base64-encoded, so **identical plans
serialize byte-identically** and a round trip reproduces the same bytes.

Content addressing (plan M8 review focus: "is ``task_id`` actually content-addressed,
cache-poisoning-safe?"). ``DurablePlan.task_id`` is a SHA-256 over the IR identity, the
``process`` spec, and the partition — so the id changes iff the computation or its input changes,
and two different computations cannot collide onto one id. The same hash function is the basis of
the M8 content-addressed Store in ``graphed-checkpoint``.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .execution import Partition, Task
from .graphed_core import GraphStore

FORMAT_VERSION = 1
FORMAT_VERSION_V2 = "graphed-plan/2"  # a STRING (vs V1's int) -> V1 and V2 byte spaces never overlap
_TASK_DOMAIN = b"graphed-task-v1"
_TASK_DOMAIN_V2 = b"graphed-task-v2"


def _sha256_hex(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        # length-prefix each part so concatenation is injective (no boundary ambiguity)
        h.update(len(p).to_bytes(8, "big"))
        h.update(p)
    return h.hexdigest()


@dataclass(frozen=True)
class OpSpec:
    """How to recover a callable on a source-free machine.

    - ``kind="ref"``: import ``module:qualname`` from an **installed package** (no user source
      files on disk are needed — the package is on ``sys.path``). The durable, preferred form.
    - ``kind="opaque"``: the callable is embedded **by value** via cloudpickle (base64). Used only
      when the callable is not importable (a closure/lambda/``__main__`` function). ``opaque`` is
      then ``True`` — a preservation risk M9 surfaces.
    """

    kind: Literal["ref", "opaque"]
    ref: str = ""
    blob_b64: str = ""

    @property
    def opaque(self) -> bool:
        return self.kind == "opaque"

    def identity(self) -> bytes:
        """Bytes that identify the callable, for content addressing."""
        if self.kind == "ref":
            return b"ref\0" + self.ref.encode()
        return b"opaque\0" + base64.b64decode(self.blob_b64)

    def resolve(self) -> Callable[..., Any]:
        if self.kind == "ref":
            mod_name, _, qual = self.ref.partition(":")
            if not qual:
                raise ValueError(f"OpSpec ref must be 'module:qualname', got {self.ref!r}")
            obj: Any = importlib.import_module(mod_name)
            for part in qual.split("."):
                obj = getattr(obj, part)
            return obj  # type: ignore[no-any-return]
        import cloudpickle  # noqa: PLC0415  (lazy: only opaque plans need it)

        fn = cloudpickle.loads(base64.b64decode(self.blob_b64))
        return fn  # type: ignore[no-any-return]

    @classmethod
    def from_ref(cls, ref: str) -> OpSpec:
        return cls(kind="ref", ref=ref)

    @classmethod
    def from_callable(cls, fn: Callable[..., Any], *, prefer_ref: bool = True) -> OpSpec:
        """Reference ``fn`` by import path if it is importable, else embed it by value (opaque)."""
        mod = getattr(fn, "__module__", None)
        qual = getattr(fn, "__qualname__", None)
        if prefer_ref and mod and qual and "<locals>" not in qual and mod != "__main__":
            try:
                if cls.from_ref(f"{mod}:{qual}").resolve() is fn:
                    return cls.from_ref(f"{mod}:{qual}")
            except (ImportError, AttributeError, ValueError):
                pass
        import cloudpickle  # noqa: PLC0415

        return cls(kind="opaque", blob_b64=base64.b64encode(cloudpickle.dumps(fn)).decode())

    def _to_json(self) -> dict[str, str]:
        return {"kind": self.kind, "ref": self.ref, "blob_b64": self.blob_b64}

    @classmethod
    def _from_json(cls, d: Mapping[str, str]) -> OpSpec:
        kind = d["kind"]
        ref, blob = d.get("ref", ""), d.get("blob_b64", "")
        if kind == "ref":
            return cls(kind="ref", ref=ref, blob_b64=blob)
        if kind == "opaque":
            return cls(kind="opaque", ref=ref, blob_b64=blob)
        raise ValueError(f"unknown OpSpec kind {kind!r}")


@dataclass(frozen=True)
class DurablePlan:
    """A versioned, deterministic, content-addressed serializable plan (plan M8).

    Holds the canonical serialized IR plus the execution metadata an executor needs: the partition
    set, the columns to read, the reduction spec (``process``/``combine``/``empty``), stopping
    conditions, file-locality, and resource hints.
    """

    ir: bytes
    process: OpSpec
    combine: OpSpec
    empty: OpSpec
    partitions: Sequence[Partition] = ()
    read_columns: Sequence[str] = ()
    stopping: Mapping[str, float] = field(default_factory=dict)
    file_locality: Mapping[str, str] = field(default_factory=dict)
    resource_hints: Mapping[str, float] = field(default_factory=dict)
    format_version: int = FORMAT_VERSION

    # ---- the live graph -------------------------------------------------------------------------
    def graph(self) -> GraphStore:
        """Rebuild the interned IR (no user source files required)."""
        return GraphStore.deserialize(self.ir)

    @property
    def opaque(self) -> bool:
        """True if any callable is embedded by value (a preservation risk; plan A.3.1)."""
        return self.process.opaque or self.combine.opaque or self.empty.opaque

    # ---- content addressing ---------------------------------------------------------------------
    def ir_fingerprint(self) -> str:
        """SHA-256 of the canonical IR bytes — the identity of the *computation*."""
        return _sha256_hex(self.ir)

    def fingerprint(self) -> str:
        """SHA-256 of the whole serialized plan (computation + all execution metadata)."""
        return _sha256_hex(self.to_bytes())

    def task_id(self, partition: Partition) -> str:
        """Content-addressed id for running ``process`` over ``partition`` under this plan.

        Cache-poisoning-safe: the id is a SHA-256 over the IR identity, the ``process`` spec, and
        the partition, so it changes iff the computation or the input changes, and distinct
        computations cannot collide onto one id.
        """
        return _sha256_hex(
            _TASK_DOMAIN,
            self.ir,
            self.process.identity(),
            _partition_bytes(partition),
        )

    # ---- "compile once, run on N datasets" ------------------------------------------------------
    def with_partitions(self, partitions: Iterable[Partition]) -> DurablePlan:
        """A sibling plan with the SAME computation (``ir`` + reduction spec + column/stopping/
        resource metadata) but a new partition set. The ``ir`` is shared unchanged, so the analysis
        is **not** recorded/optimized/serialized again — this is the basis of the *compile once, run
        on N datasets* deployment pattern."""
        return replace(self, partitions=tuple(partitions))

    def for_dataset(self, dataset: Dataset, *, chunk_size: int) -> DurablePlan:
        """This analysis, partitioned over a single dataset (chunked into entry ranges)."""
        return self.with_partitions(partition_dataset(dataset, chunk_size=chunk_size))

    def for_datasets(self, datasets: Iterable[Dataset], *, chunk_size: int) -> DurablePlan:
        """This analysis, partitioned over several datasets at once (partitions concatenated)."""
        return self.with_partitions(partition_datasets(datasets, chunk_size=chunk_size))

    # ---- (de)serialization ----------------------------------------------------------------------
    def to_bytes(self) -> bytes:
        """Canonical, byte-identical serialization (sorted-key JSON; binary blobs base64'd)."""
        doc = {
            "format_version": self.format_version,
            "ir_b64": base64.b64encode(self.ir).decode(),
            "process": self.process._to_json(),
            "combine": self.combine._to_json(),
            "empty": self.empty._to_json(),
            "partitions": [_partition_json(p) for p in self.partitions],
            "read_columns": list(self.read_columns),
            "stopping": dict(self.stopping),
            "file_locality": dict(self.file_locality),
            "resource_hints": dict(self.resource_hints),
        }
        return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> DurablePlan:
        doc = json.loads(data)
        return cls(
            ir=base64.b64decode(doc["ir_b64"]),
            process=OpSpec._from_json(doc["process"]),
            combine=OpSpec._from_json(doc["combine"]),
            empty=OpSpec._from_json(doc["empty"]),
            partitions=tuple(_partition_from_json(p) for p in doc["partitions"]),
            read_columns=tuple(doc["read_columns"]),
            stopping=dict(doc["stopping"]),
            file_locality=dict(doc["file_locality"]),
            resource_hints=dict(doc["resource_hints"]),
            format_version=int(doc["format_version"]),
        )


@dataclass(frozen=True)
class StageSpec:
    """One stage of a multi-stage :class:`DurablePlanV2` (plan M39 §4.4). A shuffle is a stage-1
    map-write (``kind="map_write"``) that a stage-2 gather (``kind="gather"|"gather_join"``) depends
    on via ``inputs`` (the indices of the stages it reads).

    ``process`` is the per-task callable (an :class:`OpSpec`, source-free like V1). ``routing`` is
    the free-form exchange scheme the stage carries (``scheme``/``key``/``parts`` + the backend's
    ``backend_id`` — recorded here, in the EXISTING routing map, so folding it into the task id is
    NOT a schema change, §7.2). ``tasks`` are the stage's producer-/gather-tasks.
    """

    kind: str
    inputs: tuple[int, ...] = ()
    process: OpSpec = field(default_factory=lambda: OpSpec.from_ref("builtins:list"))
    routing: Mapping[str, Any] = field(default_factory=dict)
    tasks: tuple[Task, ...] = ()

    def _to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "inputs": list(self.inputs),
            "process": self.process._to_json(),
            "routing": dict(self.routing),
            "tasks": [{"key": t.key, "partition": _partition_json(t.partition)} for t in self.tasks],
        }

    @classmethod
    def _from_json(cls, d: Mapping[str, Any]) -> StageSpec:
        return cls(
            kind=str(d["kind"]),
            inputs=tuple(int(i) for i in d["inputs"]),
            process=OpSpec._from_json(d["process"]),
            routing=dict(d["routing"]),
            tasks=tuple(Task(int(t["key"]), _partition_from_json(t["partition"])) for t in d["tasks"]),
        )


@dataclass(frozen=True)
class DurablePlanV2:
    """The additive multi-stage durable plan (plan M39 §4.4). A shuffle needs genuine intra-run
    staging (a gather stage depending on a map-write stage), which the single map->reduce
    :class:`DurablePlan` cannot express. This is a SEPARATE, purely-additive class with a DISTINCT
    string ``format_version`` (``"graphed-plan/2"`` vs V1's int), so a V1 blob and a V2 blob can
    never collide and each reader rejects the other's bytes loudly. V1 is untouched.
    """

    ir: bytes
    stages: tuple[StageSpec, ...] = ()
    format_version: str = FORMAT_VERSION_V2

    def graph(self) -> GraphStore:
        """Rebuild the interned IR (no user source files required)."""
        return GraphStore.deserialize(self.ir)

    def ir_fingerprint(self) -> str:
        return _sha256_hex(self.ir)

    def task_id(self, stage_index: int, task: Task) -> str:
        """Content-addressed id for one task of one stage. Folds the stage's ``routing`` (hence its
        ``backend_id``, §7.2) so two conforming backends that route a key to different dests never
        journal the same id for different content (B-r5.2); cache-poisoning-safe like V1's."""
        stage = self.stages[stage_index]
        routing_bytes = json.dumps(dict(stage.routing), sort_keys=True, separators=(",", ":")).encode()
        return _sha256_hex(
            _TASK_DOMAIN_V2,
            self.ir,
            str(stage_index).encode(),
            stage.kind.encode(),
            stage.process.identity(),
            routing_bytes,
            _partition_bytes(task.partition),
            str(task.key).encode(),
        )

    def to_bytes(self) -> bytes:
        """Canonical, byte-identical serialization (sorted-key JSON; the IR base64'd)."""
        doc = {
            "format_version": self.format_version,
            "ir_b64": base64.b64encode(self.ir).decode(),
            "stages": [s._to_json() for s in self.stages],
        }
        return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> DurablePlanV2:
        doc = json.loads(data)
        if doc.get("format_version") != FORMAT_VERSION_V2:
            raise ValueError(
                f"not a {FORMAT_VERSION_V2!r} blob (format_version={doc.get('format_version')!r}); "
                "a V1 DurablePlan blob must be read by DurablePlan.from_bytes"
            )
        return cls(
            ir=base64.b64decode(doc["ir_b64"]),
            stages=tuple(StageSpec._from_json(s) for s in doc["stages"]),
        )


@dataclass(frozen=True)
class Dataset:
    """A named input dataset to be split into work units (plan glossary 'Partition').

    ``uri`` identifies the input (a file path, a content-addressed dataset id, …) and is what an
    executor's source op / ``open_once`` reads at run time; ``n_events`` is its length; ``tree`` is
    the object path within it. A ``Dataset`` carries no data — only the reference + length.
    """

    uri: str
    n_events: int
    tree: str = "Events"
    name: str = ""


def partition_dataset(dataset: Dataset, *, chunk_size: int) -> tuple[Partition, ...]:
    """Split a dataset's ``[0, n_events)`` into contiguous half-open chunks of at most ``chunk_size``."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    out: list[Partition] = []
    start = 0
    while start < dataset.n_events:
        stop = min(start + chunk_size, dataset.n_events)
        out.append(Partition(dataset.uri, dataset.tree, start, stop))
        start = stop
    return tuple(out)


def partition_datasets(datasets: Iterable[Dataset], *, chunk_size: int) -> tuple[Partition, ...]:
    """Partition several datasets and concatenate the work units (one combined run over all of them)."""
    out: list[Partition] = []
    for ds in datasets:
        out.extend(partition_dataset(ds, chunk_size=chunk_size))
    return tuple(out)


def _partition_json(p: Partition) -> dict[str, str | int]:
    d: dict[str, str | int] = {
        "uri": p.uri,
        "tree": p.tree,
        "entry_start": p.entry_start,
        "entry_stop": p.entry_stop,
    }
    # blind fields only when set, so non-blind plans' bytes/task_ids are unchanged (M8 pins)
    if p.blind_step is not None and p.blind_n_steps is not None:
        d["blind_step"] = p.blind_step
        d["blind_n_steps"] = p.blind_n_steps
    return d


def _partition_from_json(d: Mapping[str, Any]) -> Partition:
    if "blind_step" in d:
        return Partition.blind(d["uri"], d.get("tree", ""), int(d["blind_step"]), int(d["blind_n_steps"]))
    return Partition(
        uri=d["uri"],
        tree=d.get("tree", ""),
        entry_start=int(d.get("entry_start", 0)),
        entry_stop=int(d.get("entry_stop", 0)),
    )


def _partition_bytes(p: Partition) -> bytes:
    return json.dumps(_partition_json(p), sort_keys=True, separators=(",", ":")).encode()
