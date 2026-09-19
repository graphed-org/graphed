"""m59 extra — a numpy ARRAY operand is not a scalar operand.

The dtype-keeping path is entered on a 0-d value only; an ndarray has a `dtype` too, so without
the shape guard it would reach `.item()` and fail with numpy's own `ValueError` instead of the
`TypeError` an unsupported operand has always raised.
"""

from __future__ import annotations

import numpy as np
import pytest
from m13_toy import ToyBackend, source

from graphed import Session


def test_a_multi_element_ndarray_operand_is_refused_as_before() -> None:
    session = Session(ToyBackend())
    a = source(session, "a")

    with pytest.raises(TypeError):
        a * np.array([1.0, 2.0])

    assert session.node_count() == 1  # nothing recorded on the way out
    assert (a * np.float32(0.5)).node_id != a.node_id  # control: a 0-d value does record
