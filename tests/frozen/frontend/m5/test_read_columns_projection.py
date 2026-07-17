"""M5 column projection for plans (``read_columns``): the minimal source columns a recorded graph
reads — the dask-awkward ``necessary_columns`` analogue a partition-wise plan passes to its reader.
A NEW frozen file (test-authoring deliverable); the existing m5 suite is unchanged.

The bug this guards: a plan that does not project reads EVERY branch of a wide file (e.g. all 86 of a
NanoAOD) for a query that needs one, an order-of-magnitude I/O blowup on a real dataset.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from graphed import Array, Session, read_columns


class _Backend:
    def op_form(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> str:
        return op

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        return inputs[0]

    def boundary_ops(self) -> frozenset[str]:
        return frozenset({"source"})

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        return used

    def external_payload(self, op: str, params: Mapping[str, object]) -> None:
        return None


def _events(s: Session) -> Array:
    return s.source("events", form="f", data={"x": 1})


def test_field_access_projects_to_that_column() -> None:
    s = Session(_Backend())
    ev = _events(s)
    assert read_columns([ev["MET_pt"]], ev.node_id) == ("MET_pt",)


def test_multiple_fields_union_sorted() -> None:
    s = Session(_Backend())
    ev = _events(s)
    assert read_columns([ev["pt"], ev["eta"]], ev.node_id) == ("eta", "pt")


def test_fields_op_projects_each_named_field() -> None:
    s = Session(_Backend())
    ev = _events(s)
    assert read_columns([ev[["pt", "eta"]]], ev.node_id) == ("eta", "pt")


def test_bare_source_read_widens_to_all() -> None:
    s = Session(_Backend())
    ev = _events(s)
    assert read_columns([ev], ev.node_id) is None  # the array IS the source -> read everything


def test_nonfield_op_on_source_widens_to_all() -> None:
    s = Session(_Backend())
    ev = _events(s)
    assert read_columns([ev + 1], ev.node_id) is None  # whole-record consumption -> cannot narrow


def test_any_conservative_fill_widens_the_whole_read() -> None:
    s = Session(_Backend())
    ev = _events(s)
    # one fill projects cleanly, another consumes the whole record -> the union must read everything
    assert read_columns([ev["pt"], ev + 1], ev.node_id) is None
