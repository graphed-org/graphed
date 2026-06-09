"""Necessary-buffer (column) projection via a reporting typetracer (plan M5 + M10).

Builds a reporting typetracer per source from its form, replays the recorded stages symbolically
(no event data is read), and collects the touched buffer form-keys. Two granularities share the
replay:

- `project` (M5) maps touched LEAF buffers to column names — the column-level view.
- `project_buffers` (M10) keeps the full buffer-level information awkward already reports: every
  touched structural node (list offsets / index / option masks) is attributed to its dotted path,
  so a count-only access reports `{collection: OFFSETS}` instead of the under-specified empty set,
  and a reader can serve it from a counter branch (TTree) or an index column (RNTuple) without the
  payload. `project_buffers(...).to_projection()` collapses exactly to the M5 column view.

Opaque ops (`map`) honor the on-fail policy. Projection is correct whether or not the graph has
been reduced (it replays the ops, which a fused stage contains unchanged).
"""

from __future__ import annotations

import awkward as ak
from graphed import (
    CONSERVATIVE,
    Array,
    BufferNeed,
    BufferProjection,
    Projection,
    handle_opaque,
)

_STRUCTURAL = ("is_list", "is_regular", "is_option", "is_indexed")


def _leaf_columns(form: object, path: tuple[str, ...] = ()) -> dict[str, str]:
    """Map each leaf buffer's form_key -> its dotted column path."""
    out: dict[str, str] = {}
    if getattr(form, "is_record", False):
        for field, content in zip(form.fields, form.contents, strict=True):  # type: ignore[attr-defined]
            out.update(_leaf_columns(content, (*path, field)))
    elif any(getattr(form, a, False) for a in _STRUCTURAL):
        out.update(_leaf_columns(form.content, path))  # type: ignore[attr-defined]
    elif getattr(form, "is_numpy", False):
        key = getattr(form, "form_key", None)
        if key is not None:
            out[key] = ".".join(path) if path else "<root>"
    return out


def _structural_paths(form: object, path: tuple[str, ...] = ()) -> dict[str, str]:
    """Map each STRUCTURAL buffer's form_key (list offsets / index / option masks) -> the dotted
    path it shapes. The outermost event structure (empty path) carries no column to read, so it is
    deliberately absent."""
    out: dict[str, str] = {}
    if getattr(form, "is_record", False):
        for field, content in zip(form.fields, form.contents, strict=True):  # type: ignore[attr-defined]
            out.update(_structural_paths(content, (*path, field)))
    elif any(getattr(form, a, False) for a in _STRUCTURAL):
        key = getattr(form, "form_key", None)
        if key is not None and path:
            out[key] = ".".join(path)
        out.update(_structural_paths(form.content, path))  # type: ignore[attr-defined]
    return out


def _replay(array: Array, on_fail: str) -> tuple[dict[int, tuple[object, object, str]], bool]:
    """Run the recorded graph on reporting typetracers. Returns per-source (report, form, name)
    keyed by source node id, plus whether an opaque op forced conservative projection."""
    session = array.session
    backend = session.backend

    reports: dict[int, tuple[object, object, str]] = {}
    tracers: dict[int, ak.Array] = {}
    for nid in session.source_ids():
        form = session.form_of(nid).tt.layout.form_with_key(form_key=f"s{nid}-n{{id}}")  # type: ignore[attr-defined]
        layout, report = ak.typetracer.typetracer_with_report(form, highlevel=False)
        tracers[nid] = ak.Array(layout)
        reports[nid] = (report, form, session.source_name(nid))

    conservative = False

    def on_external(_nid: int, _fn: object, inputs: list[object]) -> object:
        nonlocal conservative
        if handle_opaque("map", on_fail) is CONSERVATIVE:
            conservative = True
        # an opaque op reads its inputs fully -> mark their data touched so the input columns are
        # reported (dask-awkward does the same). Then continue with a stand-in.
        for t in inputs:
            if isinstance(t, ak.Array):
                t.layout._touch_data(recursive=True)
        return inputs[0]

    result = session.walk(
        array,
        source=lambda nid: tracers[nid],
        op=lambda _nid, name, ins, params: backend.eval_stage(name, ins, params),
        external=on_external,
    )
    # the output is materialized, so its own columns are read — touch them (covers a bare
    # field-access output that otherwise only touches shape, not leaf data).
    if isinstance(result, ak.Array):
        result.layout._touch_data(recursive=True)
    return reports, conservative


def project(array: Array, *, on_fail: str = "raise") -> Projection:
    """Compute the columns each source must read for ``array`` (metadata-only)."""
    reports, conservative = _replay(array, on_fail)
    read_columns: dict[str, frozenset[str]] = {}
    for report, form, name in reports.values():
        key_map = _leaf_columns(form)
        if conservative:
            cols = set(key_map.values())
        else:
            cols = {key_map[k] for k in report.data_touched if k in key_map}  # type: ignore[attr-defined]
        read_columns[name] = frozenset(cols)
    return Projection(read_columns)


def project_buffers(array: Array, *, on_fail: str = "raise") -> BufferProjection:
    """Compute, per source, each needed column with its :class:`graphed.BufferNeed`
    (metadata-only).

    ``DATA`` marks columns whose leaf values are read. ``OFFSETS`` marks paths whose list/option/
    index STRUCTURE alone is needed (a multiplicity, a mask) with no leaf data beneath them — the
    truthful, non-empty answer for a count-only analysis, which the column-level `project`
    necessarily reports as the empty set."""
    reports, conservative = _replay(array, on_fail)
    read_buffers: dict[str, dict[str, BufferNeed]] = {}
    for report, form, name in reports.values():
        leaf_map = _leaf_columns(form)
        if conservative:
            read_buffers[name] = dict.fromkeys(leaf_map.values(), BufferNeed.DATA)
            continue
        needs: dict[str, BufferNeed] = {}
        for key in report.data_touched:  # type: ignore[attr-defined]
            if key in leaf_map:
                needs[leaf_map[key]] = BufferNeed.DATA
        struct_map = _structural_paths(form)
        touched = set(report.data_touched) | set(report.shape_touched)  # type: ignore[attr-defined]
        data_cols = set(needs)
        for key, path in struct_map.items():
            if key not in touched:
                continue
            # a DATA read at-or-under this path already brings its structure along
            covered = any(c == path or c.startswith(path + ".") for c in data_cols)
            if not covered:
                needs.setdefault(path, BufferNeed.OFFSETS)
        read_buffers[name] = needs
    return BufferProjection(read_buffers)
