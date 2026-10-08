"""m74 T3: the environment record. Each run is its own ``python m74_env.py`` interpreter on one store; fake
distributions are ``<name>-<version>.dist-info`` directories put on ``PYTHONPATH`` (plan §2.4a, §4.1 T3)."""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import Any

import m74_helpers as h
import pytest

from graphed.checkpoint import EnvironmentChanged, Store, run_resumable, run_shuffle_resumable


@pytest.fixture
def zz(tmp_path: Path) -> dict[str, list[Path]]:
    """``PYTHONPATH`` heads: no zzfake, or zzfake at one version."""
    return {"": []} | {v: [h.fake_dist(tmp_path / "dists", "zzfake", v)] for v in ("1.0", "2.0", "3.0")}


def run(store: Path, path: list[Path], *flags: str, runner: str = "resumable") -> dict[str, Any]:
    return h.run_script("m74_env.py", runner, str(store), *flags, path=path)


def refuses(store: Path, path: list[Path], *needles: str, runner: str = "resumable") -> dict[str, Any]:
    before = h.snapshot(store)
    out = run(store, path, runner=runner)
    assert out["outcome"] == "EnvironmentChanged"
    assert out["calls"] == 0
    assert h.snapshot(store) == before
    for needle in needles:
        assert needle in out["message"]
    return out


def test_same_environment_resumes_and_an_added_distribution_refuses(tmp_path: Path, zz: dict[str, list[Path]]) -> None:
    store = tmp_path / "store"
    first = run(store, zz[""])
    assert (first["outcome"], first["reused"], first["calls"]) == ("ok", 0, h.T)
    again = run(store, zz[""])
    assert (again["outcome"], again["reused"], again["calls"]) == ("ok", h.T, 0)
    refuses(store, zz["1.0"], "zzfake", "added")


def test_changed_and_removed_distributions_are_named(tmp_path: Path, zz: dict[str, list[Path]]) -> None:
    store = tmp_path / "store"
    assert run(store, zz["1.0"])["outcome"] == "ok"
    refuses(store, zz["2.0"], "zzfake", "1.0 -> 2.0")
    refuses(store, zz[""], "zzfake", "removed")


def test_accepting_alternating_environments(tmp_path: Path, zz: dict[str, list[Path]]) -> None:
    store = tmp_path / "store"
    assert run(store, zz[""])["outcome"] == "ok"
    for _ in range(2):
        for env in ("1.0", ""):
            accepted = run(store, zz[env], "--accept")
            assert (accepted["outcome"], accepted["reused"]) == ("ok", h.T)
            plain = run(store, zz[env])
            assert (plain["outcome"], plain["reused"], plain["calls"]) == ("ok", h.T, 0)
    assert run(store, zz["1.0"], "--accept")["outcome"] == "ok"
    refuses(store, zz[""], "zzfake")


def test_the_copy_python_imports_is_the_one_fingerprinted(tmp_path: Path, zz: dict[str, list[Path]]) -> None:
    store = tmp_path / "store"
    assert run(store, zz["2.0"] + zz["1.0"])["outcome"] == "ok"
    refuses(store, zz["3.0"] + zz["1.0"], "zzfake", "2.0 -> 3.0")


def test_two_first_drivers_in_two_environments(tmp_path: Path, zz: dict[str, list[Path]]) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    assert run(a, zz[""])["outcome"] == "ok"
    assert run(b, zz["1.0"])["outcome"] == "ok"
    assert len(Store(a).completed()) == len(Store(b).completed()) == h.T
    for blob in (b / "objects").iterdir():
        shutil.copyfile(blob, a / "objects" / blob.name)
    for journal in b.glob("journal*.log"):
        with open(a / journal.name, "ab") as f:
            f.write(journal.read_bytes())
    assert len(Store(a).completed().environments) == 2
    refuses(a, zz[""], "zzfake")
    refuses(a, zz["1.0"], "zzfake")
    accepted = run(a, zz[""], "--accept")
    assert (accepted["outcome"], accepted["reused"]) == ("ok", h.T)


def test_a_cloudpickle_version_change_refuses(tmp_path: Path) -> None:
    store = tmp_path / "store"
    assert run(store, [])["outcome"] == "ok"
    refuses(store, [h.fake_dist(tmp_path / "dists", "cloudpickle", "0.0.1")], "cloudpickle")


def test_a_new_salt_reuses_nothing_and_does_not_refuse(tmp_path: Path, zz: dict[str, list[Path]]) -> None:
    store = tmp_path / "store"
    assert run(store, zz[""])["outcome"] == "ok"
    salted = run(store, zz["1.0"], "--salt", "another")
    assert (salted["outcome"], salted["reused"], salted["calls"]) == ("ok", 0, h.T)
    again = run(store, zz["1.0"], "--salt", "another")
    assert (again["outcome"], again["reused"], again["calls"]) == ("ok", h.T, 0)


def test_environment_changed_survives_pickle(tmp_path: Path, zz: dict[str, list[Path]]) -> None:
    store, saved = tmp_path / "store", tmp_path / "error.pkl"
    assert run(store, zz[""])["outcome"] == "ok"
    out = run(store, zz["1.0"], "--error", str(saved))
    assert out["outcome"] == "EnvironmentChanged"
    with open(saved, "rb") as f:
        err = pickle.load(f)
    assert type(err) is EnvironmentChanged
    assert isinstance(err, ValueError)
    assert str(err) == out["message"]
    assert "zzfake" in str(err)


@pytest.mark.parametrize(("runner", "total"), [("run_resumable", h.T), ("run_shuffle_resumable", 6)])
def test_serial_runners_check_the_environment(tmp_path: Path, zz: dict[str, list[Path]], runner: str, total: int) -> None:
    store, k = tmp_path / "store", 2
    killed = run(store, zz["1.0"], "--kill", str(k), runner=runner)
    assert (killed["outcome"], killed["calls"]) == ("interrupted", k)
    assert len(Store(store).completed()) == k
    refuses(store, zz["2.0"], "zzfake", "1.0 -> 2.0", runner=runner)
    resumed = run(store, zz["1.0"], runner=runner)
    assert (resumed["outcome"], resumed["calls"]) == ("ok", total - k)
    again = run(store, zz["1.0"], runner=runner)
    assert (again["outcome"], again["calls"]) == ("ok", 0)
    accepted = run(store, zz["2.0"], "--accept", runner=runner)
    assert (accepted["outcome"], accepted["calls"], accepted["value"]) == ("ok", 0, resumed["value"])


class SixMethodStore:
    """A CheckpointStore without ``record_environment``; ``calls`` names every method called."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def put(self, data: bytes) -> str:
        self.calls.append("put")
        return "x"

    def get(self, digest: str) -> bytes | None:
        self.calls.append("get")
        return None

    def record_done(self, task_id: str, partition: str, blob: str, *, stage: str = "", deps: tuple[str, ...] = ()) -> None:
        self.calls.append("record_done")

    def completed(self) -> dict[str, Any]:
        self.calls.append("completed")
        return {}

    def record_dead(self, descriptor: Any) -> None:
        self.calls.append("record_dead")

    def dead_letters(self) -> list[dict[str, object]]:
        self.calls.append("dead_letters")
        return []


def test_serial_runners_refuse_a_store_without_record_environment() -> None:
    for call in (
        lambda s: run_resumable(h.durable("m74_helpers:marked"), s),
        lambda s: run_shuffle_resumable(h.staged(), s),
    ):
        store = SixMethodStore()
        h.reset()
        with pytest.raises(TypeError, match="SixMethodStore"):
            call(store)
        assert store.calls == []
        assert h.STATE["calls"] == 0
