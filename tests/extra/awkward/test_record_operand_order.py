"""A binary op over record operands keeps its recorded operand order through reduction (#263)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import compile_ir
from graphed.awkward import AwkwardBackend, from_parquet
from graphed.core import SequentialRunner

vector = pytest.importorskip("vector")
vector.register_awkward()

RNG = np.random.default_rng(1)
SRC = ak.zip(
    {
        k: RNG.uniform(1, 100, 1000) if "pt" in k else RNG.uniform(-3, 3, 1000)
        for k in ("apt", "aphi", "bpt", "bphi")
    }
)


def _z(pt: Any, phi: Any) -> Any:
    return ak.zip({"pt": pt, "phi": phi}, depth_limit=1, with_name="Momentum2D")


def _pair(e: Any, k: float = 1.0) -> tuple[Any, Any]:
    return _z(e.apt * k, e.aphi), _z(e.bpt * k, e.bphi)


def _deferred(tmp_path: Path, incremental: bool) -> tuple[graphed.Session, Any]:
    pytest.importorskip("pyarrow")
    path = str(tmp_path / "src.parquet")
    ak.to_parquet(SRC, path)
    session = graphed.Session(AwkwardBackend(), incremental=incremental)
    return session, from_parquet("src", path, session=session)


def _run(*outs: Any) -> list[np.ndarray]:
    plan = graphed.aggregate_plan(*outs, reduce=list, combine=lambda a, b: a + b, empty=list)
    values = SequentialRunner().run(plan).value
    return [ak.to_numpy(ak.concatenate(v) if isinstance(v, list) else v) for v in values]


def test_the_instrument_sees_vector_add_is_not_bitwise_commutative() -> None:
    a, b = _pair(SRC)
    assert np.sum(ak.to_numpy((a + b).rho) != ak.to_numpy((b + a).rho)) > 0


def _both(e: Any, k: float) -> list[Any]:
    a, b = _pair(e, k)
    return [(a + b).rho, (b + a).rho]


@pytest.mark.parametrize("incremental", [False, True])
def test_a_planned_record_add_equals_eager_in_each_order(
    tmp_path: Path, incremental: bool
) -> None:
    _session, d = _deferred(tmp_path, incremental)
    ks = [1.0 + i / 8 for i in range(16)]
    got = _run(*(out for k in ks for out in _both(d, k)))
    want = [ak.to_numpy(out) for k in ks for out in _both(SRC, k)]
    assert len(got) == len(want)
    assert [i for i, (g, w) in enumerate(zip(got, want, strict=True)) if np.any(g != w)] == []


@pytest.mark.parametrize("incremental", [False, True])
def test_record_twins_stay_two_nodes_and_numeric_twins_one(
    tmp_path: Path, incremental: bool
) -> None:
    session, d = _deferred(tmp_path, incremental)
    a, b = _pair(d)
    assert compile_ir(session, a + b, b + a).ir != compile_ir(session, a + b, a + b).ir
    x, y = d.apt, d.bpt
    assert compile_ir(session, x + y, y + x).ir == compile_ir(session, x + y, x + y).ir


@pytest.mark.parametrize("op", ["add", "eq", "ne"])
def test_every_symmetric_op_over_records_keeps_its_order(tmp_path: Path, op: str) -> None:
    session, d = _deferred(tmp_path, False)
    a, b = _pair(d)
    ab = session.record_op(op, [a, b])
    ba = session.record_op(op, [b, a])
    assert compile_ir(session, ab, ba).ir != compile_ir(session, ab, ab).ir


def test_a_record_scalar_add_keeps_its_order(tmp_path: Path) -> None:
    session, d = _deferred(tmp_path, False)
    a, b = (v[0] for v in _pair(d))
    assert compile_ir(session, a + b, b + a).ir != compile_ir(session, a + b, a + b).ir
