"""Frontend guards refuse with their own messages."""

from __future__ import annotations

import numpy as np
import pytest

import graphed
from graphed import Session
from graphed.array import BoundMethod
from graphed.context import EventContext
from graphed.errors import GraphedError, GraphedTypeError
from graphed.numpy import NumpyBackend, from_record
from graphed.systematics.registration import _flatten


def _context() -> tuple[Session, EventContext]:
    session = Session(NumpyBackend())
    record = from_record(session, "ev", pt=np.arange(1.0, 4.0))
    return session, EventContext(session, record["pt"], collections={"pt": record["pt"]})


def test_an_event_context_is_not_iterable() -> None:
    _, ctx = _context()
    with pytest.raises(TypeError, match="an event context is not iterable"):
        iter(ctx)


def test_the_shift_form_needs_a_collection() -> None:
    _, ctx = _context()
    with pytest.raises(GraphedError, match="needs at least one collection"):
        graphed.vary(ctx, "jes", collections={})


def test_a_method_call_on_a_backend_without_methods_is_refused() -> None:
    session = Session(NumpyBackend())
    record = from_record(session, "ev", pt=np.arange(1.0, 4.0))
    with pytest.raises(GraphedTypeError, match="records no behavior methods"):
        BoundMethod(record["pt"], "frob")(1)


def test_a_variation_member_must_be_an_array_or_a_varied() -> None:
    with pytest.raises(GraphedError, match="must be an Array or a Varied, got int"):
        _flatten(3)
