"""Resume any fixed-task plan from a checkpoint store, on any runner.

:func:`resumable` is a plan transform. On the driver it keys every task, reads the store's journal
once, and returns the same plan with two changes: its ``process`` is a wrapper, and each task's
partition is a :class:`Partition` subclass carrying the task's key and, for a task the store already
holds, its blob's hash. A runner ships and calls that process as it would any other; the wrapper
either decodes the stored partial or runs the inner process and records the result. No runner reads
the extra fields, so resume is a property of the plan, not of the executor that runs it.

A task's key is :func:`_key_digest`, a hash of cloudpickle's own by-value pickling of what a worker
runs, with only value-free bytes blanked, so it changes exactly when what the worker computes can
change. The environment (installed distributions and the interpreter's cache tag) is kept beside the keys as a
store record a changed environment refuses on, rather than in them.
"""

from __future__ import annotations

import collections
import copyreg
import dataclasses
import functools
import hashlib
import importlib.metadata
import io
import json
import logging
import operator
import os
import pickle
import re
import reprlib
import sys
import threading
import types
import typing
import uuid
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, NoReturn, TypeVar, cast, overload

import cloudpickle

from graphed.core import DurablePlan, DurablePlanV2, OpSpec, Partition, Plan, Task
from graphed.core.plan import _partition_bytes, _sha256_hex
from graphed.services import Bindable, Resolvable
from graphed.write import is_url

from .codec import Codec, PickleCodec
from .fsspec_store import FsspecStore
from .store import CheckpointStore, JournalEntry, Store

R = TypeVar("R")
_T = TypeVar("_T")

_LOG = logging.getLogger("graphed.checkpoint")
_TASK_DOMAIN = b"graphed-exec-task-v1"
_ENV_DOMAIN = "graphed-env"


class StoreUnavailable(RuntimeError):
    """A worker could not open, read or write the checkpoint store."""


class EnvironmentChanged(ValueError):
    """The store's environment record for this salt differs from the running environment."""


# ---- the task key -----------------------------------------------------------------------------
_IMMUTABLE = (str, bytes, int, float, complex, tuple, frozenset, type(None), bool)
#: code attributes that say where the code is, not what it does
_LOCATIONS = frozenset(
    {"co_filename", "co_firstlineno", "co_linetable", "co_lnotab", "co_positions", "co_lines", "co_branches"}
)
#: class-dict entries the interpreter fills, and source locations
_CLASS_BLANKS = (
    "__slotnames__",
    "__non_callable_proto_members__",
    "__callable_proto_members_only__",
    "__firstlineno__",
    "__doc__",
)
_TRACKED = (type, typing.TypeVar, typing.ParamSpec, typing.TypeVarTuple)
#: the pure-Python pickler, whose hooks typeshed does not declare
_PURE: Any = pickle


def _key_only(*args: object) -> NoReturn:
    """The callable of a reduction that exists only in key bytes."""
    raise AssertionError("key bytes are hashed, never loaded")


def _code_reduce(code: types.CodeType) -> tuple[Any, ...]:
    attrs = [
        (name, getattr(code, name))
        for name in sorted(dir(code))
        if name.startswith("co_") and name not in _LOCATIONS and not callable(getattr(code, name))
    ]
    return _key_only, ("code", attrs)


_CLOUDPICKLE_PRIVATE = {
    "reducer_override": "Pickler.reducer_override",
    "function_reduce": "Pickler._function_reduce",
    "dynamic_function_reduce": "Pickler._dynamic_function_reduce",
    "function_getnewargs": "Pickler._function_getnewargs",
    "dispatch_table": "Pickler._dispatch_table",
    "trackers": "_DYNAMIC_CLASS_TRACKER_BY_CLASS",
    "module_reduce": "_module_reduce",
    "dynamic_subimport": "dynamic_subimport",
}


@functools.cache
def _cloudpickle() -> types.SimpleNamespace:
    """cloudpickle's private API the key pickler reuses (named in ``_CLOUDPICKLE_PRIVATE``), read here
    alone and only when a key is taken."""
    try:
        cp = cloudpickle.cloudpickle
        return types.SimpleNamespace(
            **{name: operator.attrgetter(path)(cp) for name, path in _CLOUDPICKLE_PRIVATE.items()}
        )
    except AttributeError as exc:
        raise TypeError(
            f"resumable cannot key tasks with cloudpickle {cloudpickle.__version__}: {exc}"
        ) from exc


class _HashSink:
    """A pickler's file that hashes what it is given instead of keeping it."""

    def __init__(self) -> None:
        self.hash = hashlib.sha256()

    def write(self, data: bytes) -> int:
        self.hash.update(data)
        return len(data)


def _resolves(module: str, qualname: str, obj: object) -> bool:
    found: Any = sys.modules.get(module)
    for part in qualname.split("."):
        found = getattr(found, part, None)
    return found is obj


class _KeyPickler(pickle._Pickler):
    """cloudpickle's by-value/by-name partition and reducers on the pure-Python pickler (the C one
    skips ``reducer_override`` for exact sets), with the key rules of :func:`_key_digest`.

    ``seen`` holds the classes pickled by value so far; a set element's sort key, pickled apart from
    the whole, names such a class instead of recursing into it."""

    def __init__(self, file: io.BytesIO | _HashSink, seen: Mapping[int, type]) -> None:
        super().__init__(file, protocol=5)
        self._cp = _cloudpickle()
        self.globals_ref: dict[int, dict[str, Any]] = {}
        self.proto = 5
        self._seen = dict(seen)
        own: dict[type, Callable[[Any], Any]] = {
            weakref.WeakSet: lambda ws: (weakref.WeakSet, (self._sorted(ws),)),
            types.ModuleType: self._module_reduce,
            types.CodeType: _code_reduce,
        }
        # built per call, so a reducer registered after import is honoured
        live = cast("dict[type, Callable[[Any], Any]]", copyreg.dispatch_table)
        self.dispatch_table = collections.ChainMap(own, self._cp.dispatch_table, live)

    def _module_reduce(self, module: types.ModuleType) -> tuple[Any, ...]:
        if module.__name__ == "__main__":
            raise TypeError("the __main__ module cannot be keyed: a worker's __main__ is not the driver's")
        if getattr(module, "__spec__", None) is None:  # made at run time, so no import finds it by name
            return self._cp.dynamic_subimport, (
                module.__name__,
                {k: v for k, v in vars(module).items() if k != "__builtins__"},
            )
        reduced: tuple[Any, ...] = self._cp.module_reduce(module)
        return reduced

    def _sorted(self, items: Iterable[Any]) -> list[Any]:
        return sorted(items, key=lambda item: _dumps(item, self._seen))

    def memoize(self, obj: Any) -> None:
        # interning would decide where memo references fall, and a loaded closure never shares a cell
        if not isinstance(obj, (*_IMMUTABLE, types.CellType)):
            _PURE._Pickler.memoize(self, obj)

    def save_global(self, obj: Any, name: str | None = None) -> None:
        qualname = name or getattr(obj, "__qualname__", None) or obj.__name__
        module = _PURE.whichmodule(obj, qualname)
        if module == "__main__":
            raise TypeError(
                f"{obj!r} is referenced by the name __main__.{name or obj.__qualname__}, whose content"
                " cannot be keyed: define it in an importable module"
            )
        _PURE._Pickler.save_global(self, obj, name)

    def reducer_override(self, obj: object) -> Any:
        kind = type(obj)
        if kind is set or kind is frozenset:
            return kind, (self._sorted(cast("Iterable[Any]", obj)),)
        if isinstance(obj, type) and id(obj) in self._seen:
            return _key_only, (obj.__module__, obj.__qualname__)
        ignore: Sequence[str] = getattr(kind, "checkpoint_ignore", ())
        resolve: Mapping[str, Callable[[Any], Any]] = getattr(kind, "checkpoint_resolve", {})
        if ignore or resolve:
            fields: dict[str, Any] = dict.fromkeys(ignore)
            fields.update(
                {name: (getattr(obj, name), fn(getattr(obj, name))) for name, fn in resolve.items()}
            )
            return dataclasses.replace(cast("Any", obj), **fields).__reduce_ex__(5)
        if not isinstance(obj, (type, types.FunctionType, types.ModuleType)):
            qualname, module = getattr(obj, "__qualname__", None), getattr(obj, "__module__", None)
            named = isinstance(qualname, str) and isinstance(module, str) and module != "__main__"
            if named and _resolves(cast("str", module), cast("str", qualname), obj):
                return qualname  # by name, before a library reduce that carries per-object ids
        reduced = self._cp.reducer_override(self, obj)
        if reduced is NotImplemented or not isinstance(obj, type) or len(reduced) < 3:
            return reduced
        self._seen[id(obj)] = obj
        func, args, (state, slotstate), *rest = reduced
        # from 3.14, a class cloudpickle rebuilt by setattr holds its annotations under this name
        state = {
            "__annotations__" if k == "__annotations_cache__" else k: v
            for k, v in state.items()
            if k not in _CLASS_BLANKS
        }
        if isinstance(state.get("_abc_impl"), list):  # cloudpickle lists an ABC's registry from a set
            state["_abc_impl"] = self._sorted(state["_abc_impl"])
        return (func, args, (state, slotstate), *rest)

    def _function_reduce(self, obj: object) -> Any:
        return self._cp.function_reduce(self, obj)

    def _dynamic_function_reduce(self, func: types.FunctionType) -> tuple[Any, ...]:
        make, args, (state, slotstate), *rest = self._cp.dynamic_function_reduce(self, func)
        return (make, args, (state, {**slotstate, "__doc__": None}), *rest)

    def _function_getnewargs(self, func: types.FunctionType) -> tuple[Any, ...]:
        code, base_globals, *rest = self._cp.function_getnewargs(self, func)
        return (code, {k: v for k, v in base_globals.items() if k != "__file__"}, *rest)

    def save_reduce(self, func: Any, args: Any, *rest: Any, obj: Any = None) -> None:
        if isinstance(obj, _TRACKED):  # cloudpickle's per-process tracker id of a by-value class
            tracker = self._cp.trackers.get(obj)
            if tracker is not None:
                args = tuple(None if isinstance(a, str) and a == tracker else a for a in args)
        _PURE._Pickler.save_reduce(self, func, args, *rest, obj=obj)


def _dumps(obj: object, seen: Mapping[int, type]) -> bytes:
    buffer = io.BytesIO()
    _KeyPickler(buffer, seen).dump(obj)
    return buffer.getvalue()


def _key_digest(obj: object) -> bytes:
    """The sha256 of ``obj``'s key pickling: cloudpickle's by-value pickling of what a worker
    resolves, made the same in every interpreter that holds the same values.

    - sets, frozensets, a ``weakref.WeakSet`` and an ABC's registry are written sorted by their
      elements' key bytes;
    - interpreter-filled class caches (``__slotnames__``, typing's protocol-member caches) are dropped,
      and a class's ``__annotations_cache__`` is written as ``__annotations__``;
    - an object other than a function, class or module that its ``__module__``/``__qualname__``
      name in a module other than ``__main__`` is written by that name;
    - a module without ``__spec__`` is written by value; a name into ``__main__`` and the ``__main__``
      module are refused (``TypeError``), since their content cannot be keyed;
    - cloudpickle's per-process tracker ids, code locations (``co_filename``, ``co_firstlineno``,
      line and position tables), a function's ``__file__`` global and doc, and a class's
      ``__firstlineno__`` and doc are blanked, and no immutable or closure cell is memoized;
    - an instance whose class declares ``checkpoint_ignore`` is written with those fields ``None``,
      and one whose class declares ``checkpoint_resolve`` with each declared field as
      ``(value, resolver(value))``.

    The pickling is hashed as it is written, so a payload many objects share is never held twice.
    """
    sink = _HashSink()
    _KeyPickler(sink, {}).dump(obj)
    return sink.hash.digest()


# ---- the per-task channel and the worker side -------------------------------------------------
@dataclass(frozen=True)
class _Keyed(Partition):
    """A task's partition, with its key and, when the store held it at plan time, its blob."""

    task_id: str = ""
    blob: str = ""


def _plain(partition: Partition) -> Partition:
    return Partition(**{f.name: getattr(partition, f.name) for f in dataclasses.fields(Partition)})


def _partition_tag(p: Partition) -> str:
    return f"{p.uri}@{p.entry_start}:{p.entry_stop}"


_OPEN = threading.local()
#: filesystems with no directories, so a cached store cannot outlive its root
_DIRLESS = frozenset({"s3", "s3a", "gcs", "gs", "abfs", "az", "adl", "memory"})
_TOKEN: tuple[int, str] = (os.getpid(), uuid.uuid4().hex)


def open_store(root: str, node: str | None = None, storage_options: Mapping[str, Any] | None = None) -> Any:
    """``root`` as a checkpoint store: an :class:`FsspecStore` when it contains ``://``, else a
    directory :class:`Store`."""
    if is_url(root):
        return FsspecStore(root, node, **dict(storage_options or {}))
    return Store(root, node)


def task_store(root: str, storage_options: Sequence[tuple[str, Any]] = ()) -> Any:
    """A worker's store at ``root``, writing its own journal per process and thread.

    An object-store root's store is opened once per (root, options, process, thread): reopening
    costs a bucket request per task. Any other root is opened per call, since a cached store would
    outlive a removed directory."""
    global _TOKEN
    if _TOKEN[0] != os.getpid():  # a forked child must not share its parent's journals
        _TOKEN = (os.getpid(), uuid.uuid4().hex)
    cache: dict[tuple[int, str, str], Any] = _OPEN.__dict__.setdefault("stores", {})
    key = (os.getpid(), root, repr(storage_options))
    if key in cache:
        return cache[key]
    store = open_store(root, f"{_TOKEN[1]}-{threading.get_ident()}", dict(storage_options))
    protocols = getattr(getattr(store, "fs", None), "protocol", ())
    if is_url(root) and _DIRLESS & set(protocols if isinstance(protocols, (tuple, list)) else (protocols,)):
        cache[key] = store
    return store


def _guarded(root: str, call: Callable[[], _T]) -> _T:
    try:
        return call()
    except Exception as exc:
        raise StoreUnavailable(
            f"checkpoint store {root} is unavailable: {type(exc).__name__}: {exc}"
        ) from exc


def _served_or_run(
    root: str,
    storage_options: Sequence[tuple[str, Any]],
    partition: Partition,
    run: Callable[[Partition], bytes],
    stage: str = "",
) -> bytes:
    """The stored bytes of a done task, else ``run(plain partition)``'s, recorded."""
    keyed = cast("_Keyed", partition)
    store = _guarded(root, lambda: task_store(root, storage_options))
    if keyed.blob:
        stored: bytes | None = _guarded(root, lambda: store.get(keyed.blob))
        if stored is not None:
            return stored
    plain = _plain(keyed)
    data = run(plain)
    _guarded(
        root, lambda: store.record_done(keyed.task_id, _partition_tag(plain), store.put(data), stage=stage)
    )
    return data


@dataclass(frozen=True)
class _Resumable:
    """A runtime plan's resumable process; ``reused`` counts the tasks the store held at plan time."""

    inner: Callable[[Partition, Any], Any]
    root: str
    storage_options: tuple[tuple[str, Any], ...]
    codec: Codec
    reused: int

    def __call__(self, partition: Partition, resources: Any) -> Any:
        def run(plain: Partition) -> bytes:
            return self.codec.encode(self.inner(plain, resources))

        return self.codec.decode(_served_or_run(self.root, self.storage_options, partition, run))

    def bind_services(self, endpoints: Mapping[str, str]) -> _Resumable:
        return replace(
            self,
            inner=self.inner.bind_services(endpoints) if isinstance(self.inner, Bindable) else self.inner,
        )

    def resolve_services(self, value: Any) -> Any:
        return self.inner.resolve_services(value) if isinstance(self.inner, Resolvable) else value

    def part_paths(self, partition: Partition) -> Sequence[str]:
        hook = getattr(self.inner, "part_paths", None)
        return () if hook is None else cast("Sequence[str]", hook(_plain(partition)))


@dataclass(frozen=True)
class _ResumableStage:
    """A ``DurablePlanV2`` stage's resumable process: payload bytes are stored as they are."""

    inner: Callable[[Task, Sequence[bytes], Any], bytes]
    root: str
    storage_options: tuple[tuple[str, Any], ...]
    kind: str
    reused: int

    def __call__(self, task: Task, inputs: Sequence[bytes], resources: Any) -> bytes:
        def run(plain: Partition) -> bytes:
            return self.inner(Task(task.key, plain), inputs, resources)

        return _served_or_run(self.root, self.storage_options, task.partition, run, self.kind)

    def bind_services(self, endpoints: Mapping[str, str]) -> _ResumableStage:
        return replace(
            self,
            inner=self.inner.bind_services(endpoints) if isinstance(self.inner, Bindable) else self.inner,
        )

    def resolve_services(self, value: Any) -> Any:
        return self.inner.resolve_services(value) if isinstance(self.inner, Resolvable) else value


# ---- the environment record -------------------------------------------------------------------
def _environment() -> bytes:
    """Canonical JSON of the installed distributions (the copy Python imports for a name installed
    twice) and the interpreter's cache tag."""
    versions: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            versions.setdefault(re.sub(r"[-_.]+", "-", name).lower(), dist.version)
    doc = {"cache_tag": sys.implementation.cache_tag, "distributions": versions}
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


def _changes(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    was, now = old.get("distributions", {}), new["distributions"]
    out = [f"{n} {now[n]} added" for n in sorted(now.keys() - was.keys())]
    out += [f"{n} {was[n]} removed" for n in sorted(was.keys() - now.keys())]
    out += [f"{n} {was[n]} -> {now[n]}" for n in sorted(was.keys() & now.keys()) if was[n] != now[n]]
    if old.get("cache_tag") != new["cache_tag"]:
        out.append(f"interpreter {old.get('cache_tag')} -> {new['cache_tag']}")
    return out


def check_environment(
    store: CheckpointStore, done: Mapping[str, JournalEntry], salt: str, accept: bool
) -> None:
    """Hold ``store``'s environment record for ``salt`` against this environment, from the
    ``completed()`` mapping ``done``: none is written as record 1; one equal to the record in force
    passes; any other raises :class:`EnvironmentChanged`, unless ``accept`` writes this one next."""
    mine = _environment()
    digest = Store.content_hash(mine)
    prefix = f"{_ENV_DOMAIN}:{_sha256_hex(salt.encode())}:"
    records = []
    for entry in getattr(done, "environments", ()):
        if entry.task_id.startswith(prefix):
            n, _, blob = entry.task_id[len(prefix) :].partition(":")
            records.append((int(n), blob))
    top = max((n for n, _ in records), default=0)
    in_force = {blob for n, blob in records if n == top}
    if in_force == {digest}:
        return
    if in_force and not accept:
        new = json.loads(mine)
        changes = [
            change
            for blob in sorted(in_force - {digest})
            for change in _changes(json.loads(store.get(blob) or b"{}"), new)
        ]
        raise EnvironmentChanged(
            f"this environment differs from the one the checkpoint store recorded ({'; '.join(changes)});"
            " pass accept_environment=True to resume in this one, or use a new salt or store"
        )
    blob = store.put(mine)
    cast("Any", store).record_environment(f"{prefix}{top + 1:08d}:{blob}", blob)


def require_environment_store(store: object) -> None:
    """Refuse a store that cannot hold the environment record, before any call to it."""
    if not callable(getattr(store, "record_environment", None)):
        raise TypeError(
            f"{type(store).__name__} has no record_environment, so it cannot hold the environment record"
        )


# ---- the transform ----------------------------------------------------------------------------
def _processes(plan: Plan[Any] | DurablePlan | DurablePlanV2) -> list[Any]:
    """The plan's processes, after the refusals that must precede any pickling."""
    if isinstance(plan, Plan) and plan.next_tasks is not None:
        raise TypeError("resumable needs a fixed task set; this plan pulls tasks from next_tasks")
    if isinstance(plan, DurablePlanV2):
        processes = [stage.process.resolve() for stage in plan.stages]
    else:
        processes = [plan.process.resolve() if isinstance(plan, DurablePlan) else plan.process]
    for process in processes:
        if isinstance(process, (_Resumable, _ResumableStage)):
            raise TypeError("this plan is already resumable")
        if plan.services and not getattr(process, "checkpointable", False):
            raise TypeError(
                f"this plan has services and its process {type(process).__qualname__} does not declare"
                " checkpointable, so its results may stand for server state a store cannot hold; a"
                " process whose result is self-contained may set checkpointable = True"
            )
    return processes


def _behaviors() -> tuple[bytes, ...]:
    """The sorted key digests of the global ``ak.behavior`` entries, which a backend built without its
    own behavior dict reads at run time."""
    ak = sys.modules.get("awkward")
    if ak is None:
        return ()
    entries = ak.behavior.items()
    return tuple(
        sorted(_keyed(entry, functools.partial("ak.behavior[{!r}]".format, entry[0])) for entry in entries)
    )


def _keyed(obj: object, what: Callable[[], str]) -> bytes:
    try:
        return _key_digest(obj)
    except Exception as exc:
        raise TypeError(f"resumable cannot key {what()}: {type(exc).__name__}: {exc}") from exc


def _of(process: object) -> Callable[[], str]:
    return lambda: f"the process {reprlib.repr(process)}"  # a repr can be as large as the payloads it shows


def _task_id(ident: str, salt: str, partition: Partition) -> str:
    return _sha256_hex(_TASK_DOMAIN, ident.encode(), salt.encode(), _partition_bytes(partition))


def _task_ids(
    plan: Plan[Any] | DurablePlan | DurablePlanV2, codec: Codec, salt: str
) -> tuple[list[Any], list[list[str]]]:
    """The plan's processes and each one's task keys, in task order."""
    _cloudpickle()  # refuses a missing private name itself, not as the first thing keyed
    processes = _processes(plan)
    behaviors = _behaviors()
    if isinstance(plan, DurablePlanV2):
        ids: list[list[str]] = []
        for si, (stage, process) in enumerate(zip(plan.stages, processes, strict=True)):
            body = _keyed((process, behaviors), _of(process))
            routing = json.dumps(dict(stage.routing), sort_keys=True, separators=(",", ":")).encode()
            upstream = [tid.encode() for dep in stage.inputs for tid in ids[dep]]
            head = (plan.ir, str(si).encode(), stage.kind.encode(), routing)
            ids.append(
                [
                    _task_id(_sha256_hex(*head, str(t.key).encode(), body, *upstream), salt, t.partition)
                    for t in stage.tasks
                ]
            )
        return processes, ids
    (process,) = processes
    if isinstance(plan, DurablePlan):
        ident = _sha256_hex(plan.ir, _keyed((process, codec, behaviors), _of(process)))
        return processes, [[_task_id(ident, salt, p) for p in plan.partitions]]
    ident = _keyed((process, codec, behaviors), _of(process)).hex()
    return processes, [[_task_id(ident, salt, t.partition) for t in plan.tasks]]


def check_resumable(plan: Plan[Any] | DurablePlan | DurablePlanV2) -> None:
    """Raise the ``TypeError`` :func:`resumable` would raise for ``plan``, without any I/O."""
    _task_ids(plan, PickleCodec(), "")


def _tagged(task: Task, task_id: str, done: Mapping[str, JournalEntry]) -> Task:
    entry = done.get(task_id)
    return Task(
        task.key,
        _Keyed(**vars(_plain(task.partition)), task_id=task_id, blob="" if entry is None else entry.blob),
    )


@overload
def resumable(
    plan: DurablePlanV2,
    store: str,
    *,
    storage_options: Mapping[str, Any] | None = ...,
    codec: Codec | None = ...,
    salt: str = ...,
    accept_environment: bool = ...,
) -> DurablePlanV2: ...
@overload
def resumable(
    plan: DurablePlan,
    store: str,
    *,
    storage_options: Mapping[str, Any] | None = ...,
    codec: Codec | None = ...,
    salt: str = ...,
    accept_environment: bool = ...,
) -> Plan[Any]: ...
@overload
def resumable(
    plan: Plan[R],
    store: str,
    *,
    storage_options: Mapping[str, Any] | None = ...,
    codec: Codec | None = ...,
    salt: str = ...,
    accept_environment: bool = ...,
) -> Plan[R]: ...
def resumable(
    plan: Plan[Any] | DurablePlan | DurablePlanV2,
    store: str,
    *,
    storage_options: Mapping[str, Any] | None = None,
    codec: Codec | None = None,
    salt: str = "",
    accept_environment: bool = False,
) -> Plan[Any] | DurablePlanV2:
    """``plan`` made to resume from the checkpoint store at ``store`` on whatever runner runs it.

    ``store`` is the root every worker opens: a directory, or an fsspec URL with
    ``storage_options``. Each task's result is recorded there as it finishes; a task the store
    already holds is decoded by the worker that runs it instead of recomputed. ``codec`` (default
    :class:`PickleCodec`) turns a runtime plan's partials into bytes; a ``DurablePlanV2``'s stage
    payloads are stored as they are. ``salt`` goes into every key: change it to recompute what the
    key cannot see (an editable install, an environment variable or file a task reads).

    A ``DurablePlan`` becomes a runtime :class:`~graphed.core.Plan`; a ``DurablePlanV2`` stays one.
    The returned process (each stage's) has ``reused``, the tasks the store held. A plan without a
    fixed task set, an already-resumable one, one with ``services`` whose process does not declare
    ``checkpointable``, and a process that cannot be keyed are refused with ``TypeError`` before
    any store I/O. A store whose environment record differs raises :class:`EnvironmentChanged`
    unless ``accept_environment``."""
    codec = PickleCodec() if codec is None else codec
    processes, ids = _task_ids(plan, codec, salt)
    options = tuple((storage_options or {}).items())
    st = open_store(store, None, dict(options))
    done = st.completed()
    check_environment(st, done, salt, accept_environment)
    reused = [sum(tid in done for tid in stage) for stage in ids]
    _LOG.info("%d of %d tasks reused from %s", sum(reused), sum(map(len, ids)), store)
    if isinstance(plan, DurablePlanV2):
        stages = []
        for stage, inner, stage_ids, n in zip(plan.stages, processes, ids, reused, strict=True):
            op = stage.process
            live = _ResumableStage(inner, store, options, stage.kind, n)
            stages.append(
                replace(
                    stage,
                    process=OpSpec(op.kind, op.ref, op.blob_b64, live=live),
                    tasks=tuple(_tagged(t, tid, done) for t, tid in zip(stage.tasks, stage_ids, strict=True)),
                )
            )
        return replace(plan, stages=tuple(stages))
    if isinstance(plan, DurablePlan):
        tasks: Iterable[Task] = (Task(i, p) for i, p in enumerate(plan.partitions))
        base: Plan[Any] = Plan(
            process=processes[0],
            combine=plan.combine.resolve(),
            empty=plan.empty.resolve(),
            services=plan.services,
        )
    else:
        tasks, base = plan.tasks, plan
    keyed = tuple(_tagged(t, tid, done) for t, tid in zip(tasks, ids[0], strict=True))
    return replace(base, process=_Resumable(processes[0], store, options, codec, reused[0]), tasks=keyed)
