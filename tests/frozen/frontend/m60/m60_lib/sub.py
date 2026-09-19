"""A submodule of the stand-in library: `register_internal("m60_lib")` covers it too."""

from __future__ import annotations

from typing import Any


def scale(array: Any) -> Any:
    return array.session.record_op("scale", [array])


def _burst(value: Any) -> Any:
    raise ValueError("m60 burst")


def failing(array: Any) -> Any:
    """An op the library records that raises when it runs — V3's `StageError` source."""
    return array.map(_burst)
