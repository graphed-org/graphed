"""evaluate_ir drops each intermediate after its last consumer, inside a fused stage and across nodes."""

from __future__ import annotations

import hashlib
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import graphed
import graphed.execute as execute
from graphed import Session
from graphed.aggregate import external_evaluators
from graphed.core import GraphStore
from graphed.numpy import NumpyBackend, NumpyForm

DATA: npt.NDArray[np.float64] = np.arange(1_000_000, dtype=np.float64)


def _stage_chain(x: Any) -> Any:
    for _ in range(6):
        x = (x + 1.0) * 1.5
    return x


def _external_chain(x: Any) -> Any:
    for _ in range(6):
        x = graphed.apply(np.negative, x)
    return x


def _peak_and_digest(chain: Callable[[Any], Any], kinds: list[str]) -> tuple[int, str]:
    session = Session(NumpyBackend())
    out = chain(session.source("x", form=NumpyForm(DATA.dtype, shape=(None,)), data=DATA))
    compiled = graphed.compile_ir(session, out)
    assert [node["kind"] for node in GraphStore.deserialize(bytes(compiled.ir)).nodes()] == kinds
    externals = external_evaluators(session, compiled)
    tracemalloc.start()
    try:
        (value,) = execute.evaluate_ir(compiled, NumpyBackend(), {"x": DATA}, externals=externals)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return peak, hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


@pytest.mark.parametrize(
    ("chain", "kinds"),
    [(_stage_chain, ["source", "stage"]), (_external_chain, ["source"] + ["external"] * 6)],
    ids=["stage-members", "nodes"],
)
def test_an_intermediate_is_released_after_its_last_consumer(
    monkeypatch: pytest.MonkeyPatch, chain: Callable[[Any], Any], kinds: list[str]
) -> None:
    released = _peak_and_digest(chain, kinds)
    monkeypatch.setattr(execute, "_last_consumers", lambda inputs, keep: {}, raising=False)
    retained = _peak_and_digest(chain, kinds)
    assert released[1] == retained[1]
    assert released[0] <= retained[0] - 2 * DATA.nbytes
