"""m71: a gufunc is typed by its signature, so an `output_type` declaration on one is refused."""

from __future__ import annotations

import numpy as np
import pytest

from graphed import GraphedTypeError, Session
from graphed.numpy import NumpyBackend, from_array


def test_a_gufunc_refuses_a_declared_output_type() -> None:
    sn = Session(NumpyBackend())
    x = from_array(sn, "x", np.zeros((2, 3), dtype=np.float32))
    params = {"fn": "g", "signature": "(n)->()", "dtype": "<f4"}
    assert sn.form(sn.record_external("gufunc", np.sum, [x], params)).describe() == "vector[float32]"
    with pytest.raises(GraphedTypeError, match="declared only on map"):
        sn.record_external("gufunc", np.sum, [x], params, output_type="bool")
