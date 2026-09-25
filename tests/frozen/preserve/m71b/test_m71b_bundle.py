"""m71b B-C7: a declared External preserves — its bundle reproduces the in-process histogram, the
reproduce run evaluates with the declared type in its params, and the evaluator survives cloudpickle."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import awkward as ak
import cloudpickle
import numpy as np
import pytest
from m71b_fixtures import EVENTS, MASK, SEEN, recorded

from graphed.preserve import build_bundle, register_plugin, reproduce
from graphed.preserve.externals import _base, record_external

HIST: dict[str, Any] = {"name": "x", "bins": 4, "lo": 0.0, "hi": 4.0}


@pytest.fixture
def registered() -> Iterator[None]:
    register_plugin(MASK, validate=False)
    try:
        yield
    finally:
        _base._REGISTRY.pop(MASK.kind, None)
        SEEN.clear()


def _histogram(value: object, weight: object) -> np.ndarray:
    counts, _ = np.histogram(
        np.asarray(ak.to_numpy(ak.Array(value)), dtype="float64"),
        bins=int(HIST["bins"]),
        range=(HIST["lo"], HIST["hi"]),
        weights=np.asarray(ak.to_numpy(ak.Array(weight)), dtype="float64"),
    )
    return np.round(counts, 6)


def test_a_declared_external_reproduces_from_its_bundle(registered: None, tmp_path: Path) -> None:
    s, ev = recorded()
    mask = record_external(s, MASK, b"m", [ev.x], output_type="bool")
    value = ev.x[mask]
    weight = value * 2.0
    reference = _histogram(s.materialize(value), s.materialize(weight))
    bundle = build_bundle(
        tmp_path / "bundle",
        session=s,
        value=value,
        weight=weight,
        datasets={"events": EVENTS},
        payloads={MASK.content_hash(b"m"): b"m"},
        histogram=HIST,
    )

    SEEN.clear()
    out = reproduce(bundle)
    assert reference.sum() > 0
    assert np.array_equal(out, reference)
    assert SEEN and all(params["output_type"] == "bool" for params in SEEN)


def test_the_recorded_evaluator_survives_cloudpickle(registered: None) -> None:
    s, ev = recorded()
    mask = record_external(s, MASK, b"m", [ev.x], output_type="bool")
    evaluator = s._externals[mask.node_id][0]
    clone = cloudpickle.loads(cloudpickle.dumps(evaluator))

    assert ak.to_list(clone(EVENTS.x)) == ak.to_list(evaluator(EVENTS.x)) == [False, True, True, True]
