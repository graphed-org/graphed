"""Necessary-buffer (column) projection for the numpy backend via field-touch tracking (plan M5).

Replays the recorded computation on lightweight tracers that record which (source, column) pairs are
actually read — record sources project to only their touched fields; flat sources are whole-buffer.
Opaque `map` ops honor the on-fail policy. This is the numpy analogue of graphed-awkward's reporting
typetracer (the user asked for a genuine projection here, not trivial all-inputs).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from graphed import CONSERVATIVE, Array, Projection, handle_opaque


@dataclass(frozen=True)
class _Tracer:
    """A projection tracer: the (source, column) pairs read so far, and — if still a live record —
    which source it is and that source's available columns."""

    touched: frozenset[tuple[str, str]]
    record: tuple[str, tuple[str, ...]] | None


def _union(inputs: Sequence[object]) -> frozenset[tuple[str, str]]:
    out: frozenset[tuple[str, str]] = frozenset()
    for t in inputs:
        if isinstance(t, _Tracer):
            out |= t.touched
    return out


def project(array: Array, *, on_fail: str = "raise") -> Projection:
    """Compute the columns each source must read for ``array``."""
    session = array.session

    source_tracer: dict[int, _Tracer] = {}
    all_columns: dict[str, set[str]] = {}
    for nid in session.source_ids():
        name = session.source_name(nid)
        form = session.form_of(nid)
        fields = getattr(form, "fields", None)
        if fields:
            cols = tuple(f for f, _ in fields)
            all_columns[name] = set(cols)
            source_tracer[nid] = _Tracer(frozenset(), (name, cols))
        else:  # a flat source is a single whole-buffer "column" named after the source
            all_columns[name] = {name}
            source_tracer[nid] = _Tracer(frozenset({(name, name)}), None)

    conservative = False

    def on_op(_nid: int, name: str, ins: list[object], params: Mapping[str, object]) -> object:
        if name == "field":
            rec = ins[0]
            if isinstance(rec, _Tracer) and rec.record is not None:
                src_name, _ = rec.record
                return _Tracer(rec.touched | {(src_name, str(params["field"]))}, None)
        return _Tracer(_union(ins), None)

    def on_external(_nid: int, _fn: object, ins: list[object]) -> object:
        nonlocal conservative
        if handle_opaque("map", on_fail) is CONSERVATIVE:
            conservative = True
        return _Tracer(_union(ins), None)

    result = session.walk(array, source=lambda nid: source_tracer[nid], op=on_op, external=on_external)

    if conservative:
        return Projection({s: frozenset(cols) for s, cols in all_columns.items()})

    read: dict[str, set[str]] = {}
    if isinstance(result, _Tracer):
        for src_name, col in result.touched:
            read.setdefault(src_name, set()).add(col)
    return Projection({s: frozenset(cols) for s, cols in read.items()})
