"""m74 T2, in process: what the task key separates and what it keeps equal (plan §2.3, §4.1 T2). Each pair
is built inside its test so the names do not resolve; the same builder called again is the control."""

from __future__ import annotations

import enum
import functools
import math
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import m74_helpers as h
import pytest

from graphed.checkpoint import NumpyCodec, resumable
from graphed.core import Partition, SequentialRunner


def fill(process: Any, root: Path, **kw: Any) -> None:
    rp = resumable(h.plan_of(process), str(root), **kw)
    assert rp.process.reused == 0
    SequentialRunner().run(rp)


def reused(process: Any, root: Path, **kw: Any) -> int:
    count: int = resumable(h.plan_of(process), str(root), **kw).process.reused
    return count


def separates(make: Callable[[Any], Any], a: Any, b: Any, root: Path) -> None:
    fill(make(a), root)
    assert reused(make(a), root) == h.T
    assert reused(make(b), root) == 0


def test_rebuild_salt_codec_process_and_partition(tmp_path: Path) -> None:
    h.reset()
    fill(h.Binned(5), tmp_path)
    assert reused(h.Binned(5), tmp_path) == h.T
    assert reused(h.Binned(5), tmp_path, salt="another") == 0
    assert reused(h.Binned(5), tmp_path, codec=NumpyCodec()) == 0
    assert reused(h.Binned(6), tmp_path) == 0
    parts = list(h.partitions())
    parts[2] = Partition(parts[2].uri, parts[2].tree, parts[2].entry_start, parts[2].entry_stop + 1)
    rp = resumable(h.plan_of(h.Binned(5), parts), str(tmp_path))
    assert rp.process.reused == h.T - 1
    assert [t.partition.blob == "" for t in rp.tasks] == [i == 2 for i in range(h.T)]


def test_wraps_closures_differing_in_a_captured_value(tmp_path: Path) -> None:
    def base(p: Any, r: Any) -> float:
        return float(p.entry_start)

    def make(f: float) -> Any:
        @functools.wraps(base)
        def inner(p: Any, r: Any) -> float:
            return base(p, r) * f

        return inner

    separates(make, 2.0, 3.0, tmp_path)


def test_closures_differing_in_a_default(tmp_path: Path) -> None:
    def make(k: float) -> Any:
        def inner(p: Any, r: Any, k: float = k) -> float:
            return float(p.entry_start) * k

        return inner

    separates(make, 2.0, 3.0, tmp_path)


def test_closures_differing_in_an_attribute(tmp_path: Path) -> None:
    def make(s: float) -> Any:
        def inner(p: Any, r: Any) -> float:
            return float(p.entry_start) * inner.scale  # type: ignore[attr-defined]

        inner.scale = s  # type: ignore[attr-defined]
        return inner

    separates(make, 2.0, 3.0, tmp_path)


def _call(self: Any, p: Any, r: Any) -> float:
    return float(p.entry_start) * self.factor


def test_type_classes_differing_in_a_class_attribute(tmp_path: Path) -> None:
    separates(lambda f: type("Proc", (), {"factor": f, "__call__": _call})(), 2.0, 3.0, tmp_path)


def test_type_classes_differing_in_a_base(tmp_path: Path) -> None:
    class Two:
        factor = 2.0

    class Three:
        factor = 3.0

    separates(lambda base: type("Proc", (base,), {"__call__": _call})(), Two, Three, tmp_path)


def test_type_classes_whose_call_closes_over_different_values(tmp_path: Path) -> None:
    def make(f: float) -> Any:
        def call(self: Any, p: Any, r: Any) -> float:
            return float(p.entry_start) * f

        return type("Proc", (), {"__call__": call})()

    separates(make, 2.0, 3.0, tmp_path)


def test_lambdas_differing_in_body(tmp_path: Path) -> None:
    def make(which: str) -> Any:
        if which == "times":
            return lambda p, r: float(p.entry_start) * 2.0
        return lambda p, r: float(p.entry_start) + 2.0

    separates(make, "times", "plus", tmp_path)


def test_class_holding_a_module_is_accepted_and_reused(tmp_path: Path) -> None:
    def make() -> Any:
        return type("Proc", (), {"lib": math, "__call__": lambda self, p, r: self.lib.sqrt(float(p.entry_start))})()

    fill(make(), tmp_path)
    assert reused(make(), tmp_path) == h.T


def test_enum_member_is_accepted_and_reused(tmp_path: Path) -> None:
    def make() -> Any:
        class Color(enum.Enum):
            RED = 2
            BLUE = 3

        color = Color.RED
        return lambda p, r: float(p.entry_start) * color.value

    fill(make(), tmp_path)
    assert reused(make(), tmp_path) == h.T


def test_class_holding_a_set_of_its_own_instances_is_accepted_and_reused(tmp_path: Path) -> None:
    def make() -> Any:
        class Single:
            ALL: frozenset[Any] = frozenset()

        Single.ALL = frozenset({Single(), Single()})
        return lambda p, r: float(p.entry_start) + len(Single.ALL)

    fill(make(), tmp_path)
    assert reused(make(), tmp_path) == h.T


def test_importable_class_with_class_level_lock_and_cache(tmp_path: Path) -> None:
    fill(h.LockedProc(), tmp_path)
    assert h.LockedProc.CACHE
    assert reused(h.LockedProc(), tmp_path) == h.T


TRY_BOTH = """
def proc(p, r):
    try:
        a = float(p.entry_start)
        b = a * 2.0
    except ZeroDivisionError:
        raise
    return b
"""
TRY_FIRST = """
def proc(p, r):
    try:
        a = float(p.entry_start)
    except ZeroDivisionError:
        raise
    b = a * 2.0
    return b
"""


def _main_def(source: str, **env: Any) -> Any:
    ns: dict[str, Any] = {"__name__": "__main__", **env}
    exec(source, ns)
    return ns["proc"]


def test_main_defs_whose_try_covers_different_statements(tmp_path: Path) -> None:
    separates(lambda src: _main_def(src), TRY_BOTH, TRY_FIRST, tmp_path)


@pytest.mark.parametrize("registered", [False, True])
def test_runtime_module_global_attribute_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered: bool) -> None:
    cfg = types.ModuleType("m74_runtime_cfg")
    if registered:
        monkeypatch.setitem(sys.modules, "m74_runtime_cfg", cfg)

    def make(bins: int) -> Any:
        cfg.BINS = bins  # type: ignore[attr-defined]
        return _main_def("def proc(p, r):\n    return float(p.entry_start % cfg.BINS)\n", cfg=cfg)

    fill(make(10), tmp_path)
    assert reused(make(10), tmp_path) == h.T
    assert reused(make(11), tmp_path) == 0


CODEC = """
import pickle
class Codec:
    def encode(self, value):
        return pickle.dumps(value, protocol={protocol})
    def decode(self, data):
        return pickle.loads(data)
"""


def test_main_codec_classes_with_different_encode_bodies(tmp_path: Path) -> None:
    def codec(protocol: int) -> Any:
        ns: dict[str, Any] = {"__name__": "__main__"}
        exec(CODEC.format(protocol=protocol), ns)
        return ns["Codec"]()

    fill(h.Binned(5), tmp_path, codec=codec(5))
    assert reused(h.Binned(5), tmp_path, codec=codec(5)) == h.T
    assert reused(h.Binned(5), tmp_path, codec=codec(4)) == 0
