"""`graphed.vary`'s target and name refusals, and `with_context` on a value with no handle."""

from __future__ import annotations

import numpy as np
import pytest

import graphed
from graphed import Session
from graphed.errors import GraphedError
from graphed.numpy import NumpyBackend, from_record
from graphed.systematics.handles import with_context


def test_a_name_that_is_not_an_identifier_is_refused() -> None:
    x = from_record(Session(NumpyBackend()), "a", x=np.arange(3.0))["x"]
    with pytest.raises(
        GraphedError, match="a variation name must be a Python identifier, got 'jes up'"
    ):
        graphed.vary(x, "jes up", up=x * 1.1)


def test_a_target_that_is_no_array_varied_or_context_is_refused() -> None:
    with pytest.raises(
        GraphedError, match="takes an Array, a Varied or an event context, got int"
    ):
        graphed.vary(3, "jes")


def test_a_value_with_no_handle_passes_through_unstamped() -> None:
    value = object()
    assert with_context(value, None) is value
