"""m74 T2, cross-interpreter: each leg is ``python m74_xleg.py`` subprocesses against one store, the first
filling it, the next ones under another ``PYTHONHASHSEED`` (plan §4.1 T2 (i)-(xv))."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import m74_helpers as h
import pytest

pytest.importorskip("pyarrow")

CELL_A = "/tmp/ipykernel_1111/3071629813.py"
CELL_B = "/tmp/ipykernel_2222/3071629813.py"


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> str:
    where = tmp_path_factory.mktemp("m74data")
    h.events(where)
    return str(where)


def leg(name: str, store: Path, data: str, *args: str, seed: str = "1") -> dict[str, Any]:
    return h.run_script("m74_xleg.py", name, str(store), data, *args, seed=seed)


def filled_then_reused(name: str, store: Path, data: str, *args: str) -> dict[str, Any]:
    first = leg(name, store, data, *args, seed="1")
    assert first["reused"] == 0 or first["reused"] == [0] * len(first["tasks"])
    second = leg(name, store, data, *args, seed="2")
    assert second["reused"] == second["tasks"]
    assert second["value"] == first["value"]
    return second


def test_frozenset_dataclass_process(tmp_path: Path, data: str) -> None:
    # (i)
    first = leg("frozenset", tmp_path / "s", data, seed="1")
    second = leg("frozenset", tmp_path / "s", data, seed="2")
    assert first["plain_digest"] != second["plain_digest"]
    assert (first["reused"], second["reused"]) == (0, h.T)
    assert second["value"] == first["value"]


def test_aggregate_plan_from_an_importable_builder(tmp_path: Path, data: str) -> None:
    # (ii)
    filled_then_reused("agg", tmp_path / "s", data)


def test_aggregate_plan_under_two_cell_filenames(tmp_path: Path, data: str) -> None:
    # (iii)
    first = leg("cell", tmp_path / "s", data, CELL_A, seed="1")
    second = leg("cell", tmp_path / "s", data, CELL_B, seed="2")
    assert (first["reused"], second["reused"]) == (0, second["tasks"])
    assert second["value"] == first["value"]


@pytest.mark.parametrize("variant", ["class", "def"])
def test_shuffle_plan_main_reduce_under_two_cells(tmp_path: Path, data: str, variant: str) -> None:
    # (iv)
    first = leg("shuffle_cell", tmp_path / "s", data, CELL_A, variant, str(tmp_path / "a.pkl"), seed="1")
    second = leg("shuffle_cell", tmp_path / "s", data, CELL_B, variant, str(tmp_path / "b.pkl"), seed="2")
    assert first["reused"] == [0, 0, 0]
    assert second["reused"] == second["tasks"]
    assert second["value"] == first["value"]


def test_shuffle_plan_loaded_from_pickle_in_another_interpreter(tmp_path: Path, data: str) -> None:
    # (v)
    plan_file = str(tmp_path / "plan.pkl")
    first = leg("shuffle_cell", tmp_path / "s", data, CELL_A, "class", plan_file, seed="1")
    loaded = leg("shuffle_cell", tmp_path / "s", data, CELL_B, "load", plan_file, seed="2")
    assert first["reused"] == [0, 0, 0]
    assert loaded["reused"] == loaded["tasks"]
    assert loaded["value"] == first["value"]


def test_frozen_dataclass_with_defaults(tmp_path: Path, data: str) -> None:
    # (vi)
    filled_then_reused("defaults", tmp_path / "s", data)


def test_main_reduce_reading_a_module_global(tmp_path: Path, data: str) -> None:
    # (vii)
    store = tmp_path / "s"
    first = leg("bins", store, data, "0", "10", "a")
    moved = leg("bins", store, data, "4", "10", "a", seed="2")
    assert (first["reused"], moved["reused"]) == (0, moved["tasks"])
    assert leg("bins", store, data, "0", "11", "a", seed="3")["reused"] == 0
    assert leg("bins", store, data, "0", "10", "b", seed="4")["reused"] == 0


def test_generic_process_class(tmp_path: Path, data: str) -> None:
    # (viii)
    filled_then_reused("generic", tmp_path / "s", data)


def test_cloudpickled_plan_loaded_twice(tmp_path: Path, data: str) -> None:
    # (ix)
    plan_file = str(tmp_path / "plan.cpkl")
    first = leg("cloudpickled", tmp_path / "s", data, "build", plan_file, seed="1")
    loaded = leg("cloudpickled", tmp_path / "s", data, "load", plan_file, seed="2")
    assert first["reused"] == 0
    assert loaded["reused"] == [h.T, h.T]
    assert loaded["value"] == first["value"]


def test_copyreg_reducer_registered_after_import(tmp_path: Path, data: str) -> None:
    # (x)
    filled_then_reused("copyreg", tmp_path / "s", data)


def test_abc_registry_and_weakset_made_in_another_order(tmp_path: Path, data: str) -> None:
    # (xi)
    first = leg("abc", tmp_path / "s", data, "fwd", seed="1")
    second = leg("abc", tmp_path / "s", data, "rev", seed="2")
    assert (first["reused"], second["reused"]) == (0, h.T)


def test_main_result_class_rebuilt_in_one_interpreter(tmp_path: Path, data: str) -> None:
    # (xii)
    out = leg("result_class", tmp_path / "s", data)
    assert out["reused"] == h.T
    assert out["value"][0] == out["value"][1]


@pytest.mark.parametrize("variant", ["global", "ref", "callable"])
def test_mixin_edit_reaches_an_aggregate_key(tmp_path: Path, data: str, variant: str) -> None:
    # (xiii): global ak.behavior; backend="module:factory"; backend=module.factory
    store, dest = tmp_path / "s", str(tmp_path / "out")
    first = leg("mixin", store, data, variant, "2", dest, seed="1")
    same = leg("mixin", store, data, variant, "2", dest, seed="2")
    edited = leg("mixin", store, data, variant, "3", dest, seed="3")
    assert (first["reused"], same["reused"], edited["reused"]) == (0, same["tasks"], 0)
    assert same["value"] == first["value"]
    assert edited["value"] == edited["fresh"] != first["value"]


@pytest.mark.parametrize("layout", ["before", "after"])
def test_mixin_edit_reaches_shuffle_stages(tmp_path: Path, data: str, layout: str) -> None:
    # (xiii): mixin evaluated before or after repartition, backend="module:factory"
    store, dest = tmp_path / "s", str(tmp_path / "out")
    leg("mixin", store, data, f"shuffle_{layout}", "2", dest, seed="1")
    same = leg("mixin", store, data, f"shuffle_{layout}", "2", dest, seed="2")
    edited = leg("mixin", store, data, f"shuffle_{layout}", "3", dest, seed="3")
    assert same["reused"] == same["tasks"]
    evaluating = 0 if layout == "before" else 1
    assert edited["reused"][evaluating:] == [0] * (len(edited["tasks"]) - evaluating)
    assert edited["value"] == edited["fresh"] != same["value"]


@pytest.mark.parametrize(("variant", "kind"), [("parquet", "_WritePart"), ("varied", "_VariedWritePart")])
def test_mixin_edit_rewrites_parquet_parts(tmp_path: Path, data: str, variant: str, kind: str) -> None:
    # (xiii): to_parquet(behavior="module:attr"), plain and varied with select=, one fixed destination
    store, dest = tmp_path / "s", str(tmp_path / "out")
    first = leg("mixin", store, data, variant, "2", dest, seed="1")
    same = leg("mixin", store, data, variant, "2", dest, seed="2")
    edited = leg("mixin", store, data, variant, "3", dest, seed="3")
    assert first["kind"] == kind
    assert (first["reused"], same["reused"], edited["reused"]) == (0, same["tasks"], 0)
    assert same["written"] == first["written"]
    assert edited["written"] == first["written"] / 2 * 3


def test_importable_object_with_a_uuid_reduce(tmp_path: Path, data: str) -> None:
    # (xiv)
    first = leg("uuid_helper", tmp_path / "s", data, seed="1")
    second = leg("uuid_helper", tmp_path / "s", data, seed="2")
    assert first["reduce_varies"]
    assert (first["reused"], second["reused"]) == (0, h.T)


@pytest.mark.parametrize(
    ("variant", "kept"),
    [("gather", ["map_write"]), ("reader", []), ("join", ["map_write"])],
)
def test_upstream_edit_reaches_every_downstream_stage(tmp_path: Path, data: str, variant: str, kept: list[str]) -> None:
    # (xv): a __main__ gather reduce; a __main__ reader; a join whose second side's reader is __main__
    store = tmp_path / "s"
    first = leg("upstream", store, data, variant, "2", seed="1")
    same = leg("upstream", store, data, variant, "2", seed="2")
    edited = leg("upstream", store, data, variant, "3", seed="3")
    assert first["reused"] == [0] * len(first["tasks"])
    assert same["reused"] == same["tasks"]
    expected = [n if i < len(kept) else 0 for i, n in enumerate(edited["tasks"])]
    assert edited["reused"] == expected
    assert edited["kinds"][: len(kept)] == kept
    assert edited["value"] == edited["fresh"] != first["value"]
