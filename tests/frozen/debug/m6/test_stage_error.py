"""`StageError` is structured + picklable and never degrades to an opaque string (plan M6)."""

from __future__ import annotations

import copy
import pickle

import pytest

from graphed_debug import SourceFrame, StageError


def _err() -> StageError:
    return StageError(
        op="index",
        frames=(
            SourceFrame("analysis.py", 42, "dimuon_mass", "mu[:, 1].pt"),
            SourceFrame("analysis.py", 40, "dimuon_mass", "events.Muon"),
        ),
        input_forms=("var * float64",),
        partition="file.root:Events[0:1000]",
        cause_type="IndexError",
        cause_message="index 1 is out of range",
        opt_level=1,
    )


def test_carries_structured_fields() -> None:
    e = _err()
    assert e.op == "index"
    assert e.user_frame == SourceFrame("analysis.py", 42, "dimuon_mass", "mu[:, 1].pt")
    assert e.input_forms == ("var * float64",)
    assert e.partition == "file.root:Events[0:1000]"
    assert e.cause_type == "IndexError"


def test_pickle_round_trip_is_intact() -> None:
    e = _err()
    e2 = pickle.loads(pickle.dumps(e))
    assert isinstance(e2, StageError)
    assert e2 == e  # every structured field survives, not just a string
    assert e2.frames == e.frames
    assert e2.user_frame.source == "mu[:, 1].pt"


def test_pickle_preserves_exception_message() -> None:
    e2 = pickle.loads(pickle.dumps(_err()))
    assert str(e2) == _err().summary()
    assert "IndexError" in str(e2) and "index" in str(e2)


def test_is_a_real_exception_and_raisable() -> None:
    with pytest.raises(StageError) as info:
        raise _err()
    assert info.value.user_frame.lineno == 42


def test_deepcopy_equivalent_to_pickle() -> None:
    e = _err()
    assert copy.deepcopy(e) == e
