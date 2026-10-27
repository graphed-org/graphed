"""The numpy backend's input guards refuse with their own messages."""

from __future__ import annotations

import numpy as np
import pytest

from graphed import Session
from graphed.numpy import NumpyBackend, from_record
from graphed.numpy.forms import NumpyForm, unit_meta
from graphed.numpy.shuffle import merge_records, pack_key, take


def test_merge_records_refuses_a_shared_non_key_column() -> None:
    left = np.array([(1, 2)], dtype=[("k", "i8"), ("v", "i8")])
    right = np.array([(1, 3)], dtype=[("k", "i8"), ("v", "i8")])
    with pytest.raises(ValueError, match=r"shared non-key column\(s\) \['v'\]"):
        merge_records(left, right, on=["k"])


def test_pack_key_refuses_more_fields_than_u64_holds() -> None:
    record: np.ndarray = np.zeros(1, dtype=[(name, "i8") for name in "abcd"])
    with pytest.raises(ValueError, match="4 fields x 20 bits exceeds u64"):
        pack_key(record, ["a", "b", "c", "d"])


def test_take_from_an_empty_carrier_is_all_null_rows() -> None:
    carrier: np.ndarray = np.zeros(0, dtype=[("a", "i8")])
    out = take(carrier, np.array([-1, -1]))
    assert isinstance(out, np.ma.MaskedArray)
    assert len(out) == 2
    assert bool(np.ma.getmaskarray(out)["a"].all())


def test_a_reduction_over_a_record_form_is_refused() -> None:
    session = Session(NumpyBackend())
    record = from_record(session, "ev", pt=np.arange(1.0, 4.0))
    form = session.form(record)
    assert isinstance(form, NumpyForm)
    with pytest.raises(TypeError, match="reductions need array operands"):
        unit_meta(form)
