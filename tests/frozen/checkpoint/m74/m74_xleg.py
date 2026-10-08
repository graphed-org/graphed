"""T2's cross-interpreter legs: ``python m74_xleg.py <leg> <store> <data dir> [args...]``. Everything a
leg keys is defined in this ``__main__``; the last stdout line is a JSON report."""

from __future__ import annotations

import abc
import copyreg
import dataclasses
import hashlib
import json
import pickle
import sys
import typing
import weakref
from typing import Any

import awkward as ak
import cloudpickle
import m74_helpers as h

import graphed
import graphed.awkward as ga
from graphed import Session, aggregate_plan, join, join_plan, repartition, shuffle_plan
from graphed import parquet as gpq
from graphed.awkward import AwkwardBackend, AwkwardForm, gak
from graphed.awkward.io import from_parquet, to_parquet
from graphed.checkpoint import resumable
from graphed.core import DurablePlanV2, Plan, SequentialRunner

LEG, STORE, DATA, *ARGS = sys.argv[1:]
FOLD: dict[str, Any] = {"reduce": h.agg_reduce, "combine": h.agg_combine, "empty": h.agg_empty}


def plain(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value if isinstance(value, (int, float, str)) or value is None else repr(value)


def finish(plan: Any, **extra: Any) -> None:
    rp = resumable(plan, STORE)
    value = SequentialRunner().run(rp).value
    if isinstance(rp, DurablePlanV2):
        reused, tasks = h.stage_reused(rp), [len(stage.tasks) for stage in rp.stages]
    else:
        reused, tasks = rp.process.reused, len(rp.tasks)
    print(json.dumps({"reused": reused, "tasks": tasks, "value": plain(value), **extra}))


def run_cell(source: str, filename: str) -> dict[str, Any]:
    """Run notebook-cell source in this ``__main__``, as ipykernel does."""
    exec(compile(source, filename, "exec"), globals())
    return globals()


def events() -> Any:
    return from_parquet(Session(AwkwardBackend()), "events", h.events(DATA), open_files=False, steps_per_file=2)


if LEG == "frozenset":
    # (i)
    @dataclasses.dataclass(frozen=True)
    class Tagged:
        tags: frozenset[str]

        def __call__(self, p: Any, r: Any) -> float:
            return float(len(self.tags) + p.entry_start)

    tagged = Tagged(frozenset({"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"}))
    finish(h.plan_of(tagged), plain_digest=hashlib.sha256(pickle.dumps(tagged)).hexdigest())

elif LEG == "agg":
    # (ii)
    finish(h.agg_plan(h.events(DATA)))

elif LEG == "cell":
    # (iii): ARGS = [cell filename]
    ns = run_cell(
        """
ev = events()
def agg_reduce(values):
    return tuple(float(v) for v in values)
def agg_combine(a, b):
    return tuple(x + y for x, y in zip(a, b))
def agg_empty():
    return (0.0,)
plan = aggregate_plan(gak.sum((ev.x + ev.y) * 2), reduce=agg_reduce, combine=agg_combine, empty=agg_empty)
""",
        ARGS[0],
    )
    finish(ns["plan"])

elif LEG == "shuffle_cell":
    # (iv) and (v): ARGS = [cell filename, "class" | "def" | "load", plan file to write or load]
    if ARGS[1] == "load":
        with open(ARGS[2], "rb") as f:
            finish(pickle.load(f))
    else:
        variant = ARGS[1]
        ns = run_cell(
            """
ev = events()
class Red:
    def __call__(self, values):
        return tuple(float(v) for v in values)
def red(values):
    return tuple(float(v) for v in values)
def comb(a, b):
    return tuple(x + y for x, y in zip(a, b))
def emp():
    return (0.0,)
""",
            ARGS[0],
        )
        reduce = ns["red"] if variant == "def" else ns["Red"]()
        plan = shuffle_plan(gak.sum(repartition(ns["ev"], n=2).x), reduce=reduce, combine=ns["comb"], empty=ns["emp"])
        with open(ARGS[2], "wb") as f:
            pickle.dump(plan, f)
        finish(plan)

elif LEG == "defaults":
    # (vi)
    @dataclasses.dataclass(frozen=True)
    class Cut:
        names: frozenset[str] = frozenset({"a", "b", "c", "d", "e", "f", "g"})
        weights: list[float] = dataclasses.field(default_factory=lambda: [1.0, 2.0])

        def __call__(self, p: Any, r: Any) -> float:
            return len(self.names) * self.weights[1] + p.entry_start

    finish(h.plan_of(Cut()))

elif LEG == "bins":
    # (vii): ARGS = [lines inserted above the def, BINS, body "a" | "b"]
    pad, bins, body = int(ARGS[0]), ARGS[1], ARGS[2]
    expr = "float(v) % BINS" if body == "a" else "(float(v) + 1.0) % BINS"
    comment = "    # a comment\n" if pad else ""
    ns = run_cell("\n" * pad + f"BINS = {bins}\ndef reduce(values):\n{comment}    return tuple({expr} for v in values)\n", "m74_bins")
    finish(aggregate_plan(gak.sum(events().x), reduce=ns["reduce"], combine=h.agg_combine, empty=h.agg_empty))

elif LEG == "generic":
    # (viii)
    T_ = typing.TypeVar("T_")

    class GenProc(typing.Generic[T_]):
        def __init__(self, scale: float) -> None:
            self.scale = scale

        def __call__(self, p: Any, r: Any) -> float:
            return float(p.entry_start) * self.scale

    finish(h.plan_of(GenProc[float](2.0)))

elif LEG == "cloudpickled":
    # (ix): ARGS = ["build" | "load", plan file]
    if ARGS[0] == "build":

        def tripled(x: float) -> float:
            return x * 3.0

        @dataclasses.dataclass(frozen=True)
        class Proc9:
            tags: frozenset[str] = frozenset({"u", "v", "w", "x", "y", "z"})

            def __call__(self, p: Any, r: Any) -> float:
                return tripled(float(p.entry_start)) + len(self.tags)

        plan9 = h.plan_of(Proc9())
        with open(ARGS[1], "wb") as f:
            f.write(cloudpickle.dumps(plan9))
        finish(plan9)
    else:
        with open(ARGS[1], "rb") as f:
            payload = f.read()
        loaded = [resumable(cloudpickle.loads(payload), STORE) for _ in range(2)]
        value = SequentialRunner().run(loaded[1]).value
        print(json.dumps({"reused": [rp.process.reused for rp in loaded], "value": value}))

elif LEG == "copyreg":
    # (x): the reducer is registered after graphed.checkpoint is imported
    class Cfg:
        def __init__(self, k: int) -> None:
            self.k = k

        def __reduce_ex__(self, protocol: Any) -> Any:
            raise TypeError("Cfg does not reduce itself")

    def _cfg(c: Cfg) -> tuple[Any, ...]:
        return Cfg, (c.k,)

    copyreg.pickle(Cfg, _cfg)
    cfg = Cfg(3)

    def scaled(p: Any, r: Any) -> float:
        return float(p.entry_start * cfg.k)

    finish(h.plan_of(scaled))

elif LEG == "abc":
    # (xi): ARGS = ["fwd" | "rev"], the order the registered classes and the set's members are made in
    names = ["A", "B", "C", "D", "E", "F"]
    order = names if ARGS[0] == "fwd" else names[::-1]
    made = {n: type(n, (), {"f": lambda self: 1}) for n in order}
    KEEP = [made[n]() for n in order]
    MEMBERS: weakref.WeakSet[Any] = weakref.WeakSet(KEEP)

    class Scaled(abc.ABC):
        @abc.abstractmethod
        def scale(self) -> int: ...

    class Registry(Scaled):
        def scale(self) -> int:
            return len(MEMBERS)

        def __call__(self, p: Any, r: Any) -> float:
            return float(p.entry_start + self.scale())

    for n in names:
        Registry.register(made[n])
    finish(h.plan_of(Registry()))

elif LEG == "result_class":
    # (xii): one interpreter runs, then rebuilds
    @dataclasses.dataclass
    class Result:
        total: float

    def produce(p: Any, r: Any) -> Result:
        return Result(float(p.entry_start))

    def merge(a: Result, b: Result) -> Result:
        return Result(a.total + b.total)

    def nothing() -> Result:
        return Result(0.0)

    def build() -> Plan[Any]:
        return Plan(process=produce, combine=merge, empty=nothing, tasks=h.plan_of(produce).tasks)

    first = SequentialRunner().run(resumable(build(), STORE)).value
    again = resumable(build(), STORE)
    second = SequentialRunner().run(again).value
    print(json.dumps({"reused": again.process.reused, "value": [first.total, second.total]}))

elif LEG == "mixin":
    # (xiii): ARGS = [variant, V, destination]
    variant, v, dest = ARGS
    target = "ak.behavior" if variant == "global" else "h.behavior"
    run_cell(f"@ak.mixin_class({target})\nclass Point:\n    @property\n    def mag(self):\n        return self.x * {v}\n", "m74_mixin")
    backend = AwkwardBackend() if variant == "global" else AwkwardBackend(behavior=h.behavior)
    ev = from_parquet(Session(backend), "events", h.events(DATA), open_files=False, steps_per_file=2)
    if variant in ("parquet", "varied"):
        if variant == "parquet":
            plan = to_parquet(gak.zip({"x": ev.x}, with_name="Point").mag, dest, compute=False, behavior="m74_helpers:behavior")
            column = "data"
        else:
            evs = ga.gnano.events(ev)
            w = evs.x * 0 + 1.0
            ctx = graphed.vary(evs, "murf", w, is_weight=True, points={"1": w, "0.5": w * 0.5, "2": w * 2.0})
            rec = gak.zip({"m": gak.zip({"x": ctx.x}, with_name="Point").mag, "w": graphed.weight(ctx)}, depth_limit=1)
            plan = to_parquet(rec, dest, compute=False, behavior="m74_helpers:behavior", select={0: ctx.x > 10})
            column = "m"
        rp = resumable(plan, STORE)
        files = SequentialRunner().run(rp).value
        written = sum(float(ak.sum(ak.from_parquet(f)[column])) for f in files)
        print(json.dumps({"reused": rp.process.reused, "tasks": len(rp.tasks), "written": written, "kind": type(plan.process).__name__}))
    else:
        if variant in ("global", "ref", "callable"):
            ref = {"global": None, "ref": "m74_helpers:make_backend", "callable": h.make_backend}[variant]
            plan = aggregate_plan(gak.sum(gak.zip({"x": ev.x}, with_name="Point").mag), backend=ref, **FOLD)
        elif variant == "shuffle_before":
            mag = gak.zip({"x": ev.x}, with_name="Point").mag
            out = gak.sum(repartition(gak.zip({"m": mag}), n=2).m)
            plan = shuffle_plan(out, backend="m74_helpers:make_backend", **FOLD)
        else:
            out = gak.sum(gak.zip({"x": repartition(ev, n=2).x}, with_name="Point").mag)
            plan = shuffle_plan(out, backend="m74_helpers:make_backend", **FOLD)
        finish(plan, fresh=plain(SequentialRunner().run(plan).value))

elif LEG == "uuid_helper":
    # (xiv)
    from m74_helpers import KERNEL

    def kernel_proc(p: Any, r: Any) -> float:
        return KERNEL(float(p.entry_start))

    finish(h.plan_of(kernel_proc), reduce_varies=pickle.dumps(KERNEL.__reduce__()) != pickle.dumps(KERNEL.__reduce__()))

elif LEG == "upstream":
    # (xv): ARGS = [variant "gather" | "reader" | "join", V]
    variant, v = ARGS
    ns = run_cell(
        f"""
from graphed.awkward.io import read_parquet_partition
def red(values):
    return tuple(float(v) * {v} for v in values)
class Loader:
    def __init__(self, paths):
        self.paths = tuple(paths)
    def __call__(self):
        return ak.from_parquet(list(self.paths))
    def partitions(self, steps_per_file=1):
        return gpq.make_partitions(self.paths, steps_per_file=steps_per_file, open_files=False)
    def read_partition(self, partition, columns, resources):
        a = read_parquet_partition(partition, None)
        a = ak.with_field(a, a.x * {v}, "x")
        return a if columns is None else a[list(columns)]
""",
        "m74_upstream",
    )
    paths = h.events(DATA)
    s = Session(AwkwardBackend())

    def scaled_source(name: str, files: list[str]) -> Any:
        form = AwkwardForm(ak.Array(ak.from_parquet(files).layout.to_typetracer(forget_length=True)))
        return gpq.deferred_source(s, name, paths=tuple(files), form=form, loader=ns["Loader"](files))

    if variant == "gather":
        ev = from_parquet(s, "events", paths, open_files=False, steps_per_file=2)
        plan = shuffle_plan(gak.sum(repartition(ev, n=2).x), reduce=ns["red"], combine=h.agg_combine, empty=h.agg_empty, steps_per_file=2)
    elif variant == "reader":
        plan = shuffle_plan(gak.sum(repartition(scaled_source("events", paths), n=2).x), steps_per_file=2, **FOLD)
    else:
        left = from_parquet(s, "left", paths[:1], open_files=False, steps_per_file=2)
        right = scaled_source("right", paths[1:])
        right = gak.zip({"y": right.y, "z": right.x})
        plan = join_plan(gak.sum(join(left, right, on=["y"]).z), steps_per_file=2, **FOLD)
    finish(plan, fresh=plain(SequentialRunner().run(plan).value), kinds=[st.kind for st in plan.stages])

else:
    raise SystemExit(f"unknown leg {LEG!r}")
