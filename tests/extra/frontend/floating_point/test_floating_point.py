"""``raise_on_floating_point``, and a worker process's warnings reaching the driver."""

from __future__ import annotations

import functools
import multiprocessing
import os
import pickle
import subprocess
import sys
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import Session, raise_on_floating_point
from graphed.aggregate import _Collated, aggregate_plan, collate
from graphed.awkward import AwkwardBackend, AwkwardForm
from graphed.checkpoint import RetryN, RetrySmallerChunk, Store, resumable, run_resumable
from graphed.checkpoint.resume import _key_digest
from graphed.core import DurablePlan, OpSpec, Partition, Plan, SequentialRunner, worker_warnings
from graphed.core.execution import LocalResources
from graphed.core.worker_warnings import (
    Warned,
    WarnedCombine,
    WarnedFinalize,
    carrying,
    enter_worker,
    keeping_warnings,
    on_value,
    process_id,
    without_warnings,
)
from graphed.debug import replay
from graphed.errors import StageError
from graphed.services import resolve_services

ZEROS = ak.Array({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [0.0, 1.0, 0.0, 1.0, 1.0, 1.0]})
ONES = ak.Array({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 1.0, 1.0, 1.0]})


@dataclass
class Events:
    data: ak.Array

    def __call__(self) -> ak.Array:
        return self.data

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(
            Partition.blind("mem://events", "", s, steps_per_file) for s in range(steps_per_file)
        )

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> ak.Array:
        part = partition.resolve(len(self.data))
        return self.data[part.entry_start : part.entry_stop]


def as_list(values: list[object]) -> Any:
    return ak.to_list(values[0])


def concat(a: list[float], b: list[float]) -> list[float]:
    return [*a, *b]


def nothing() -> list[float]:
    return []


def _output(data: ak.Array, expr: Callable[[Any], Any]) -> Any:
    session = Session(AwkwardBackend())
    form = AwkwardForm(ak.Array(data.layout.to_typetracer(forget_length=True)))
    return expr(session.source("events", form=form, data=Events(data)))


def _plan(
    data: ak.Array = ZEROS,
    expr: Callable[[Any], Any] = lambda ev: ev.x / ev.y,
    finalize: Callable[[Any], Any] | None = None,
) -> Plan[list[float]]:
    return _planned(_output(data, expr), finalize)


def _planned(output: Any, finalize: Callable[[Any], Any] | None = None) -> Plan[list[float]]:
    plan = aggregate_plan(
        output,
        reduce=as_list,
        combine=concat,
        empty=nothing,
        steps_per_file=3,
        finalize=finalize,
    )
    assert isinstance(plan, Plan)
    return plan


def _task(process: Callable[..., Any], partition: Partition) -> Any:
    return process(partition, LocalResources())


def _spawned(plan: Plan[Any], pool: ProcessPoolExecutor) -> Any:
    run = functools.partial(_task, plan.process)
    parts = list(pool.map(run, [t.partition for t in plan.tasks]))
    return functools.reduce(plan.combine, parts, plan.empty())


@pytest.fixture(scope="module")
def pool() -> Any:
    with ProcessPoolExecutor(
        2,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=enter_worker,
        initargs=(process_id(),),
    ) as workers:
        yield workers


@pytest.fixture
def leave_driver(monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    """Mark this process a worker of another driver."""
    return lambda: monkeypatch.setattr(worker_warnings, "_worker_pid", os.getpid())


def _finalized(plan: Plan[Any], partial: Any) -> tuple[Any, list[warnings.WarningMessage]]:
    assert plan.finalize is not None
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        value = plan.finalize(partial)
    return value, seen


def test_a_worker_warning_reaches_the_driver_once(pool: ProcessPoolExecutor) -> None:
    plan = _plan()
    value, seen = _finalized(plan, _spawned(plan, pool))
    assert value[0] == float("inf")
    assert len(value) == 6
    assert [w.category for w in seen] == [RuntimeWarning]
    assert "divide by zero" in str(seen[0].message)
    assert "mem://events" in str(seen[0].message)


def test_a_collated_worker_warning_reaches_the_driver(pool: ProcessPoolExecutor) -> None:
    plan = collate({"a": _plan()})
    value, seen = _finalized(plan, _spawned(plan, pool))
    assert len(value["a"]) == 6
    assert [w.category for w in seen] == [RuntimeWarning]


def test_a_kept_worker_warns_again_on_the_next_run(pool: ProcessPoolExecutor) -> None:
    plan = _plan()
    _finalized(plan, _spawned(plan, pool))
    assert len(_finalized(plan, _spawned(plan, pool))[1]) == 1


def test_a_quiet_task_carries_nothing(pool: ProcessPoolExecutor) -> None:
    plan = _plan(ONES)
    partial = _spawned(plan, pool)
    assert partial == [1.0, 2.0, 3.0, 4.0]
    assert _finalized(plan, partial)[1] == []


def test_the_driver_leaves_warnings_alone() -> None:
    plan = _plan()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        partial = plan.process(plan.tasks[0].partition, LocalResources())
    assert not isinstance(partial, Warned)
    assert [w.category for w in seen] == [RuntimeWarning]
    assert warnings.showwarning is warnings._showwarning_orig  # type: ignore[attr-defined]


_LOADED = """
import pickle, sys, warnings
sys.path[:0] = {path!r}
from graphed.core import SequentialRunner
seen = []
warnings.showwarning = lambda message, *args, **kwargs: seen.append(str(message))
plan = pickle.loads(open({plan!r}, "rb").read())
shown, filters = warnings.showwarning, list(warnings.filters)
SequentialRunner().run(plan)
print(repr((seen, warnings.showwarning is shown, warnings.filters == filters)))
"""


def test_a_plan_loaded_in_a_fresh_driver_warns_there_as_usual(tmp_path: Any) -> None:
    plan = tmp_path / "plan.pkl"
    plan.write_bytes(pickle.dumps(_plan()))
    script = _LOADED.format(path=sys.path, plan=str(plan))
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    expected = (["divide by zero encountered in divide"], True, True)
    assert out.stdout.strip() == repr(expected), out.stderr


@pytest.mark.parametrize(
    ("driver", "forked", "kept"),
    [
        (process_id, False, []),
        (lambda: "elsewhere:1", False, ["task"]),
        (process_id, True, ["task"]),
    ],
)
def test_a_process_keeps_warnings_only_as_another_drivers_worker(
    driver: Callable[[], str], forked: bool, kept: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker_warnings, "_worker_pid", None)
    parent, pid = driver(), os.getpid()
    if forked:
        # a forked child keeps its parent's interpreter uuid under a new pid
        monkeypatch.setattr(os, "getpid", lambda: pid + 1)
    enter_worker(parent)
    with warnings.catch_warnings(record=True), keeping_warnings() as caught:
        warnings.simplefilter("always")
        warnings.warn("task", RuntimeWarning, stacklevel=1)
    assert [message for _, message in caught] == kept


def test_finalize_runs_on_the_plain_value(leave_driver: Callable[[], None]) -> None:
    plan = _plan(finalize=len)
    leave_driver()
    partial = plan.process(plan.tasks[0].partition, LocalResources())
    assert isinstance(partial, Warned)
    assert _finalized(plan, partial)[0] == 2


def test_a_warning_and_an_error_name_a_chunk_alike(leave_driver: Callable[[], None]) -> None:
    plan = _plan()
    leave_driver()
    partition = plan.tasks[0].partition
    partial = plan.process(partition, LocalResources())
    assert isinstance(partial, Warned)
    with pytest.raises(StageError) as raised:
        raise_on_floating_point(plan).process(partition, LocalResources())
    assert {chunk for *_, chunk in partial.caught.values()} == {raised.value.partition}
    assert raised.value.partition.startswith("chunk 1/3 of mem://events")


def test_a_task_outside_a_driver_keeps_only_its_own(leave_driver: Callable[[], None]) -> None:
    leave_driver()
    with keeping_warnings() as outer:
        warnings.warn("task", RuntimeWarning, stacklevel=1)
    assert [message for _, message in outer] == ["task"]


def test_outside_a_task_a_warning_is_shown_as_before(
    leave_driver: Callable[[], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    leave_driver()
    with keeping_warnings():
        pass
    shown: list[object] = []
    monkeypatch.setattr(worker_warnings, "_shown", lambda *args: shown.append(args[0]))
    warnings.warn_explicit("after", RuntimeWarning, "f.py", 1)
    assert [str(message) for message in shown] == ["after"]


def test_combining_keeps_the_left_chunks_warning() -> None:
    combine = WarnedCombine(concat)
    key = (RuntimeWarning, "m")
    left = Warned([1.0], {key: ("f", 1, "left")})
    right = Warned([2.0], {key: ("f", 1, "right")})
    assert combine(left, right) == Warned([1.0, 2.0], {key: ("f", 1, "left")})
    assert combine([0.0], left) == Warned([0.0, 1.0], left.caught)
    assert combine(left, [0.0]) == Warned([1.0, 0.0], left.caught)
    assert combine([1.0], [2.0]) == [1.0, 2.0]


def test_on_value_keeps_the_warnings() -> None:
    warned = Warned(1, {(RuntimeWarning, "m"): ("f", 1, "c")})
    assert on_value(str, warned) == Warned("1", warned.caught)
    assert on_value(str, 1) == "1"


def test_a_process_resolves_the_plain_value_and_the_warnings_stay() -> None:
    class Resolving:
        def __call__(self, partition: Partition, resources: Any) -> int:
            return 1

        def bind_services(self, endpoints: Any) -> Resolving:
            return self

        def resolve_services(self, value: int) -> int:
            return value + 1

    plan: Plan[Any] = Plan(process=Resolving(), combine=lambda a, b: a + b, empty=int)
    warned = Warned(1, {(RuntimeWarning, "m"): ("f", 1, "c")})
    assert resolve_services(plan, warned) == Warned(2, warned.caught)


def test_a_durable_plan_is_refused_by_its_process_not_as_staged() -> None:
    add = OpSpec("ref", "operator:add")
    durable: Any = DurablePlan(ir=b"fp", process=add, combine=add, empty=add)
    with pytest.raises(TypeError, match="got a OpSpec process"):
        raise_on_floating_point(durable)


def test_carrying_nothing_is_the_value() -> None:
    assert carrying(1, {}, "c") == 1


def test_a_durable_plan_folds_a_shipped_process_with_its_own_combine(
    leave_driver: Callable[[], None],
) -> None:
    plan = _plan()
    leave_driver()
    durable = DurablePlan(
        ir=b"fp",
        process=OpSpec.from_callable(plan.process),
        combine=OpSpec("ref", "operator:add"),
        empty=OpSpec("ref", "builtins:list"),
        partitions=tuple(t.partition for t in plan.tasks),
    )
    runtime = durable.to_plan()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert len(SequentialRunner().run(runtime).value) == 6
    assert "divide by zero" in str(seen[0].message)


def test_a_durable_plan_finalizes_a_shipped_collated_process_plainly(
    leave_driver: Callable[[], None],
) -> None:
    plan = collate({"a": _plan()})
    leave_driver()
    durable = DurablePlan(
        ir=b"fp",
        process=OpSpec.from_callable(plan.process),
        combine=OpSpec.from_callable(plan.combine),
        empty=OpSpec("ref", "builtins:dict"),
        partitions=tuple(t.partition for t in plan.tasks),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert len(SequentialRunner().run(durable.to_plan()).value["a"]) == 6


@pytest.mark.parametrize(
    "policy", [RetryN(), RetrySmallerChunk()], ids=["same-chunk", "smaller-chunk"]
)
def test_a_retried_task_gives_the_store_the_plain_partial(
    policy: Any, leave_driver: Callable[[], None], tmp_path: Any
) -> None:
    plan = _plan()
    leave_driver()
    process, failed = plan.process, set()

    def fails_once(partition: Partition, resources: Any) -> Any:
        if partition.entry_stop - partition.entry_start == 2 and partition not in failed:
            failed.add(partition)
            raise MemoryError
        return process(partition, resources)

    shipped = OpSpec.from_callable(fails_once)
    durable = DurablePlan(
        ir=b"retried",
        process=OpSpec(shipped.kind, shipped.ref, shipped.blob_b64, live=fails_once),
        combine=OpSpec("ref", "operator:add"),
        empty=OpSpec("ref", "builtins:list"),
        partitions=tuple(t.partition.resolve(6) for t in plan.tasks),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_resumable(durable, Store(str(tmp_path)), retry=policy)
    assert failed
    assert result.report.dead == 0
    assert len(result.value) == 6


class _Plain:
    def __init__(self) -> None:
        self.encoded: list[Any] = []

    def encode(self, value: Any) -> bytes:
        self.encoded.append(value)
        return b""

    def decode(self, data: bytes) -> Any:
        return self.encoded[-1]


def test_a_store_gets_the_plain_partial(leave_driver: Callable[[], None], tmp_path: Any) -> None:
    codec = _Plain()
    plan = resumable(_plan(), str(tmp_path), codec=codec)
    leave_driver()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SequentialRunner().run(plan)
    assert len(codec.encoded) == 3
    assert not any(isinstance(value, Warned) for value in codec.encoded)


def test_a_store_gets_a_collated_plain_partial(
    leave_driver: Callable[[], None], tmp_path: Any
) -> None:
    codec = _Plain()
    plan = resumable(collate({"a": _plan()}), str(tmp_path), codec=codec)
    leave_driver()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SequentialRunner().run(plan)
    assert [type(value["a"]) for value in codec.encoded] == [list] * 3


class _Importable(RuntimeWarning):
    pass


class _TwoArguments(UserWarning):
    def __init__(self, message: str, code: int) -> None:
        super().__init__(message, code)


def test_a_warning_class_that_cannot_travel_is_carried_as_its_builtin_base(
    leave_driver: Callable[[], None],
) -> None:
    class Local(DeprecationWarning):
        pass

    leave_driver()
    with warnings.catch_warnings(), keeping_warnings() as caught:
        warnings.simplefilter("always")
        warnings.warn("local", Local, stacklevel=1)
        warnings.warn(_TwoArguments("two", 2), stacklevel=1)
        warnings.warn("importable", _Importable, stacklevel=1)
    carried = carrying(0, caught, "c")
    assert {message: category for category, message in carried.caught} == {
        "local": DeprecationWarning,
        "('two', 2)": UserWarning,
        "importable": _Importable,
    }
    assert pickle.loads(pickle.dumps(carried)) == carried
    assert len(_issued(carried)) == 3


def _issued(warned: Warned) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        without_warnings(warned)
    return seen


def test_a_failed_task_adds_what_it_caught_to_the_error(leave_driver: Callable[[], None]) -> None:
    leave_driver()
    with pytest.raises(ValueError, match="boom") as raised, keeping_warnings():
        warnings.warn("the clue", RuntimeWarning, stacklevel=1)
        raise ValueError("boom")
    assert len(raised.value.__notes__) == 1
    assert "RuntimeWarning before this: the clue" in raised.value.__notes__[0]


def fail_after_evaluating(values: list[object]) -> Any:
    raise ValueError("boom")


def test_a_failed_workers_warnings_reach_the_driver_on_its_error(
    pool: ProcessPoolExecutor,
) -> None:
    plan = replace(
        _plan(),
        process=replace(_plan().process, reduce=fail_after_evaluating),  # type: ignore[type-var]
    )
    with pytest.raises(ValueError, match="boom") as raised:
        _spawned(plan, pool)
    assert "divide by zero" in raised.value.__notes__[0]


def test_a_resumable_run_issues_a_workers_warning_and_a_served_one_does_not(
    leave_driver: Callable[[], None], tmp_path: Any
) -> None:
    plan = _plan()
    leave_driver()
    for expected in (1, 0):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            SequentialRunner().run(resumable(plan, str(tmp_path)))
        assert len([w for w in seen if "[first in " in str(w.message)]) == expected


def test_without_warnings_issues_them_here() -> None:
    warned = Warned(1, {(RuntimeWarning, "m"): ("f", 1, "c")})
    with pytest.warns(RuntimeWarning, match=r"m \[first in c\]"):
        assert without_warnings(warned) == 1


def test_a_plan_without_finalize_gets_the_value() -> None:
    assert WarnedFinalize(None)(1) == 1


@pytest.mark.parametrize(
    ("expr", "name"),
    [
        (lambda ev: ev.x / ev.y, "divide"),
        (lambda ev: (ev.x - ev.x) / (ev.y - ev.y), "invalid"),
        (lambda ev: ev.x * 1e308 * 1e308, "overflow"),
    ],
    ids=["divide", "invalid", "overflow"],
)
def test_a_trapped_operation_raises_at_its_line(expr: Callable[[Any], Any], name: str) -> None:
    with pytest.raises(StageError) as raised:
        SequentialRunner().run(raise_on_floating_point(_plan(expr=expr)))
    assert isinstance(raised.value.__cause__, FloatingPointError)
    assert name in str(raised.value)
    assert "test_floating_point.py" in str(raised.value)
    assert "mem://events" in str(raised.value)


def test_a_worker_process_raises_alike(pool: ProcessPoolExecutor) -> None:
    with pytest.raises(StageError, match="divide"):
        _spawned(raise_on_floating_point(_plan()), pool)


def test_underflow_still_rounds_to_zero() -> None:
    plan = _plan(ONES, lambda ev: ev.x * 1e-300 * 1e-300)
    assert SequentialRunner().run(raise_on_floating_point(plan)).value == [0.0] * 4


def test_the_default_keeps_the_inf() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert SequentialRunner().run(_plan()).value[0] == float("inf")


def test_a_plan_with_nothing_to_trap_is_unchanged() -> None:
    plan = _plan(ONES)
    assert SequentialRunner().run(raise_on_floating_point(plan)).value == [1.0, 2.0, 3.0, 4.0]


def test_trapping_moves_no_task_key() -> None:
    plan = _plan()
    assert _key_digest(raise_on_floating_point(plan).process) == _key_digest(plan.process)


def test_replay_evaluates_under_the_same_trap() -> None:
    process = raise_on_floating_point(_plan()).process
    partition = _plan().tasks[0].partition
    with pytest.raises(StageError, match="divide"):
        process.evaluate(ZEROS[:2], partition.resolve(6))  # type: ignore[attr-defined]


def test_a_replayed_step_traps_as_the_task_did() -> None:
    output = _output(ZEROS, lambda ev: ev.x / ev.y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert replay(_planned(output), 0, output).value[0] == float("inf")
    with pytest.raises(StageError, match="divide"):
        replay(raise_on_floating_point(_planned(output)), 0, output).value  # noqa: B018


def test_a_collated_plan_traps_every_part() -> None:
    plans = {"a": _plan()}
    raising = raise_on_floating_point(collate(plans))
    assert isinstance(raising.process, _Collated)
    with pytest.raises(StageError, match="divide"):
        SequentialRunner().run(raising)


def test_any_other_plan_is_refused() -> None:
    plan: Plan[Any] = Plan(process=lambda part, _: part, combine=concat, empty=nothing)
    with pytest.raises(TypeError, match="aggregate_plan"):
        raise_on_floating_point(plan)
    assert graphed.raise_on_floating_point is raise_on_floating_point
