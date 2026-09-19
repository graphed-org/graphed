"""m59 extra — the numpy backend rebuilds a scalar operand in its recorded dtype.

The awkward frozen suite pins the dtype through its own evaluator; the numpy backend reads the
same params, so it keeps its own leg: without the rebuild the operand is a plain Python `int` and
numpy answers `int64`.
"""

from __future__ import annotations

import numpy as np

from graphed import Session
from graphed.numpy import NumpyBackend, from_array


def test_a_numpy_scalar_operand_keeps_its_dtype_through_the_numpy_backend() -> None:
    session = Session(NumpyBackend())
    a = from_array(session, "a", np.array([True, False, True]))

    packed = session.materialize(a * np.uint64(1 << 3))

    assert np.asarray(packed).dtype == np.dtype("uint64")
    np.testing.assert_array_equal(np.asarray(packed), np.array([True, False, True]) * np.uint64(8))
    # control: a Python operand still records and evaluates as it did
    assert np.asarray(session.materialize(a * 8)).dtype == np.dtype("int64")
