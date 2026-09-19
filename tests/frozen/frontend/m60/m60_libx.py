"""A module whose name SHARES a string prefix with `m60_lib` but no dotted component.

Registering `"m60_lib"` must leave this module's frames the user's, and its `q` must not intern
with `m60_lib.q` — one module name, two decisions.
"""

from __future__ import annotations

from typing import Any


def q(value: Any) -> Any:
    """The other half of the same-name-different-module pair."""
    return value * 100


def select(array: Any) -> Any:
    return array.session.record_op("select", [array])
