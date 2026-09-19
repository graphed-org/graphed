"""m59 extra — the one key branch the frozen suites do not reach: the EMPTY tuple key.

`a[()]` has no first member to check, so it must be refused by the same record-time `TypeError`
the axis-0 rule raises; without the guard `key[0]` raises `IndexError` instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from graphed import Session
from graphed.numpy import NumpyBackend, from_array


def test_an_empty_tuple_key_is_refused_and_records_nothing() -> None:
    session = Session(NumpyBackend())
    a = from_array(session, "a", np.arange(6.0).reshape(3, 2))
    before = session.node_count()

    with pytest.raises(TypeError):
        a[()]

    assert session.node_count() == before
    assert a[:, 0].node_id != a.node_id  # control: a well-formed key on the same array records
