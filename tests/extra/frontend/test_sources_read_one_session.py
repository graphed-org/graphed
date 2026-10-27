"""`graphed.write.sources_read` answers over one Session's sources, and refuses outputs of two."""

from __future__ import annotations

import numpy as np
import pytest

from graphed import Session
from graphed.numpy import NumpyBackend, from_record
from graphed.write import sources_read


def test_outputs_of_two_sessions_are_refused() -> None:
    first, second = Session(NumpyBackend()), Session(NumpyBackend())
    a = from_record(first, "a", x=np.arange(3.0))["x"]
    b = from_record(second, "b", y=np.arange(3.0))["y"]
    with pytest.raises(TypeError, match="must record into one session"):
        sources_read([a * 1.0, b * 2.0])


def test_outputs_of_one_session_read_every_source_they_reach() -> None:
    session = Session(NumpyBackend())
    a = from_record(session, "a", x=np.arange(3.0))["x"]
    b = from_record(session, "b", y=np.arange(3.0))["y"]
    from_record(session, "c", z=np.arange(3.0))
    read = sources_read([a * 1.0, a + b])
    assert sorted(session.source_name(nid) for nid in read) == ["a", "b"]
