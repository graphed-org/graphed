"""The sibling that shares a string prefix with `m60x_lib` but no dotted component."""

from __future__ import annotations

from typing import Any


def select(array: Any) -> Any:
    return array.session.record_op("select", [array])
