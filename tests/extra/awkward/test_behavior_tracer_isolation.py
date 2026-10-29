"""Record-time behavior code never mutates a Session's stored typetracer."""

from __future__ import annotations

from typing import Any

import awkward as ak
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, AwkwardForm


class Thing(ak.Record):  # type: ignore[misc]
    pass


class ThingArray(ak.Array):  # type: ignore[misc]
    @property
    def tagged(self) -> Any:
        # the coffea pattern: a property that caches a field and a flag on its `self`
        self["extra"] = self.x
        self.attrs["seen"] = True
        return self.x

    def tag(self) -> Any:
        return self.tagged


BEHAVIOR: dict[Any, Any] = {"thing": Thing, ("*", "thing"): ThingArray}
DATA = ak.zip(
    {"x": [1.0, 2.0], "sub": ak.zip({"x": [3.0, 4.0]}, with_name="thing")}, with_name="thing"
)


class Rows:
    def __call__(self) -> ak.Array:
        return DATA


def _tracer() -> AwkwardForm:
    return AwkwardForm(ak.Array(DATA.layout.to_typetracer(forget_length=True), behavior=BEHAVIOR))


# a stored tracer reaches behavior code as itself when it carries the backend's own behavior (a
# derived node) or no backend behavior is set
CASES = [
    ("source", False),
    ("derived", False),
    ("derived", True),
]


@pytest.mark.parametrize(("target", "backend_behavior"), CASES)
@pytest.mark.parametrize("call", ["property", "method"])
def test_a_reread_keeps_its_recorded_form(target: str, backend_behavior: bool, call: str) -> None:
    s = Session(AwkwardBackend(behavior=BEHAVIOR if backend_behavior else None))
    rows = Rows()

    def read() -> Any:
        events = s.source("events", form=_tracer(), data=rows)
        return events if target == "source" else events.sub

    first = read()
    stored = s.form(first)
    assert isinstance(stored, AwkwardForm)
    before = stored.describe()
    fields = stored.tt.fields
    tagged = first.tagged if call == "property" else first.tag()
    assert s.form(tagged).describe() == s.form(first.x).describe()
    again = read()
    assert again.node_id == first.node_id
    assert stored.describe() == before
    assert stored.tt.fields == fields
    assert "seen" not in stored.tt.attrs
