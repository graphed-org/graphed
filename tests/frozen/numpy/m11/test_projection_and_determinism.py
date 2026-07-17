"""M11: column projection tracks through the new elementwise tier; recording stays deterministic.

The M5 projection guarantee must survive the widened op surface: a computation touching two of
five record columns through new ufuncs projects to exactly those two. And the M4/M8 determinism
gate must hold over programs built from the new ops: two independent recordings of the same
program serialize to byte-identical reduced IR.
"""

from __future__ import annotations

import numpy as np
from graphed import Session

import graphed_numpy as gn


def _events(s: Session) -> object:
    return gn.from_record(
        s,
        "events",
        pt=np.array([10.0, 40.0, 25.0, 5.0]),
        eta=np.array([0.1, -2.0, 1.5, 0.3]),
        phi=np.array([0.0, 1.0, 2.0, 3.0]),
        charge=np.array([1, -1, 1, -1]),
        mask=np.array([True, True, False, True]),
    )


def test_projection_tracks_through_new_ufuncs() -> None:
    s = Session(gn.NumpyBackend())
    events = _events(s)
    out = (np.exp(events["pt"]) + np.log1p(np.absolute(events["eta"]))).reduce("sum")
    proj = gn.project(out)
    assert proj.columns_for("events") == frozenset({"pt", "eta"})


def test_new_op_evaluation_through_records() -> None:
    s = Session(gn.NumpyBackend())
    events = _events(s)
    out = np.hypot(events["pt"], events["eta"])
    np.testing.assert_array_equal(
        np.asarray(s.materialize(out)),
        np.hypot(np.array([10.0, 40.0, 25.0, 5.0]), np.array([0.1, -2.0, 1.5, 0.3])),
    )


def _program_bytes() -> bytes:
    s = Session(gn.NumpyBackend())
    a = gn.from_array(s, "a", np.array([0.5, 1.5]))
    b = gn.from_array(s, "b", np.array([2.0, 3.0]))
    out = (np.logaddexp(np.exp(a), np.floor(b)) + np.fmax(a, b)).reduce("sum")
    return s.serialized_ir(out)


def test_recording_new_ops_is_deterministic() -> None:
    assert _program_bytes() == _program_bytes()
