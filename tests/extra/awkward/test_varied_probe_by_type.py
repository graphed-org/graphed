"""An operand is told from a universe container by its class, never by an attribute lookup."""

from __future__ import annotations

from typing import Any

import awkward as ak
import pytest

from graphed import Session
from graphed.awkward import AwkwardArray, AwkwardBackend, from_awkward

asked: list[str] = []


class _Asking(AwkwardArray):
    """An array class whose attribute fallback is consulted for ``_`` names, as coffea's is."""

    def __getattr__(self, name: str) -> Any:
        asked.append(name)
        return super().__getattr__(name)


class _AskingKey:
    def __getattr__(self, name: str) -> Any:
        asked.append(name)
        raise AttributeError(name)


class _AskingBackend(AwkwardBackend):
    def array_type(self) -> type[AwkwardArray]:
        return _Asking


def test_an_array_operand_is_never_asked_for_the_container_marker() -> None:
    session = Session(_AskingBackend())
    ev = from_awkward("ev", ak.Array({"a": [1.0, 2.0], "m": [True, False]}), session=session)
    a, m = ev["a"], ev["m"]
    asked.clear()
    a + a, a.filter(m)
    with pytest.raises(TypeError, match="unsupported index"):
        a[_AskingKey()]
    assert "_varied_container" not in asked
    assert isinstance(a, _Asking)
