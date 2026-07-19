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
    """A projection tracer: the (source, column) pairs read so far, and — if still a live record — a
    per-field provenance map ``field_name -> (source, source_column)``. A single-source record maps
    every field to its own source; a JOIN output (M40) maps each field to WHICHEVER side it came from,
    so a later ``field`` access on the joined record is attributed to the correct source."""

    touched: frozenset[tuple[str, str]]
    fields: Mapping[str, tuple[str, str]] | None


def _union(inputs: Sequence[object]) -> frozenset[tuple[str, str]]:
    out: frozenset[tuple[str, str]] = frozenset()
    for t in inputs:
        if isinstance(t, _Tracer):
            out |= t.touched
    return out


def _decode_on(params: Mapping[str, object]) -> list[str]:
    return [f for f in str(params.get("on", "")).split(",") if f]


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
            source_tracer[nid] = _Tracer(frozenset(), {f: (name, f) for f in cols})
        else:  # a flat source is a single whole-buffer "column" named after the source
            all_columns[name] = {name}
            source_tracer[nid] = _Tracer(frozenset({(name, name)}), None)

    conservative = False

    def on_op(_nid: int, name: str, ins: list[object], params: Mapping[str, object]) -> object:
        if name == "field":
            rec = ins[0]
            if isinstance(rec, _Tracer) and rec.fields is not None:
                origin = rec.fields.get(str(params["field"]))
                if origin is not None:
                    return _Tracer(rec.touched | {origin}, None)
        if name == "exchange":  # a pure data-movement boundary is identity on the projection (§3.3a)
            return ins[0]
        if name == "pack_key":  # M40: reads the `on` key fields; the record (its fields) passes through
            rec = ins[0]
            if isinstance(rec, _Tracer) and rec.fields is not None:
                read = {rec.fields[f] for f in _decode_on(params) if f in rec.fields}
                return _Tracer(rec.touched | frozenset(read), rec.fields)
        if name == "join":  # M40: merge the two sides' per-field provenance (intersection-dedup, left wins)
            left, right = ins[0], ins[1]
            lf = left.fields if isinstance(left, _Tracer) and left.fields is not None else {}
            rf = right.fields if isinstance(right, _Tracer) and right.fields is not None else {}
            merged = {**lf, **{f: v for f, v in rf.items() if f not in lf}}
            return _Tracer(_union(ins), merged)
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
