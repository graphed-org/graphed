"""M11: the ``__array_function__`` protocol routes numpy API calls into recorded ops (P0.3).

M11 wires the protocol itself plus the whole-array ``np.sum`` mapping; later milestones extend the
dispatch table (axis-aware reductions M12, concatenate/where M13) without touching the protocol.
"""

from __future__ import annotations

import numpy as np
import pytest
from m11_toy import ToyBackend, source

from graphed import Session


def test_np_sum_records_a_sum_reduction() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    n = s.node_count()
    out = np.sum(a)
    assert s.form(out).describe() == "sum"
    assert s.node_count() == n + 1


def test_np_sum_interns_with_method_form() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    assert np.sum(a).node_id == a.reduce("sum").node_id


def test_unsupported_numpy_function_raises_type_error() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    with pytest.raises(TypeError):
        np.cross(a, a)
