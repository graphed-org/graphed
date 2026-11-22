"""evaluate_ir runs a later output's short branch as soon as it can, so its reads die early."""

from __future__ import annotations

import tracemalloc
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import graphed
import graphed.execute as execute
from graphed import Session
from graphed.core import GraphStore
from graphed.numpy import NumpyBackend, NumpyForm

DATA: npt.NDArray[np.float64] = np.arange(1_000_000, dtype=np.float64)
STEPS = 6


def _peak() -> tuple[int, list[Any]]:
    session = Session(NumpyBackend())
    x = session.source("x", form=NumpyForm(DATA.dtype, shape=(None,)), data=DATA)
    sums = []
    for _ in range(STEPS):
        x = x + 1.0
        sums.append(np.sum(x))
    compiled = graphed.compile_ir(session, x, *sums)  # the chain's end marked before the sums
    nodes = GraphStore.deserialize(bytes(compiled.ir)).nodes()
    # the IR lists every step before the first sum, so its own order holds all of them at once
    assert [nd["kind"] for nd in nodes] == ["source"] + ["stage"] * STEPS + ["reduction"] * STEPS
    tracemalloc.start()
    try:
        values = execute.evaluate_ir(compiled, NumpyBackend(), {"x": DATA})
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return peak, values


def test_a_short_branch_runs_before_the_chain_moves_on(monkeypatch: pytest.MonkeyPatch) -> None:
    early, early_values = _peak()
    monkeypatch.setattr(execute, "_evaluation_order", lambda inputs: list(range(len(inputs))))
    late, late_values = _peak()
    assert [float(v) for v in early_values[1:]] == [float(v) for v in late_values[1:]]
    assert np.array_equal(early_values[0], late_values[0])
    assert early <= late - (STEPS - 3) * DATA.nbytes


def test_the_order_is_topological_and_takes_the_nearest_sink_first() -> None:
    # 0 -> 1 -> 2 -> 3 (a chain) with 4 reading 1: 4 is one step from a sink, 2 is two
    order = execute._evaluation_order([(), (0,), (1,), (2,), (1,)])
    assert order == [0, 1, 4, 2, 3]
