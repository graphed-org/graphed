"""Necessary-buffer (column) projection via a reporting typetracer (plan M5).

Builds a reporting typetracer per source from its form, replays the recorded stages symbolically
(no event data is read), and collects the touched buffer form-keys, mapping them back to column
names. Opaque ops (`map`) honor the on-fail policy. Projection is correct whether or not the graph
has been reduced (it replays the ops, which a fused stage contains unchanged).
"""

from __future__ import annotations

import awkward as ak
from graphed import CONSERVATIVE, Array, Projection, handle_opaque


def _leaf_columns(form: object, path: tuple[str, ...] = ()) -> dict[str, str]:
    """Map each leaf buffer's form_key -> its dotted column path."""
    out: dict[str, str] = {}
    if getattr(form, "is_record", False):
        for field, content in zip(form.fields, form.contents, strict=True):  # type: ignore[attr-defined]
            out.update(_leaf_columns(content, (*path, field)))
    elif any(getattr(form, a, False) for a in ("is_list", "is_regular", "is_option", "is_indexed")):
        out.update(_leaf_columns(form.content, path))  # type: ignore[attr-defined]
    elif getattr(form, "is_numpy", False):
        key = getattr(form, "form_key", None)
        if key is not None:
            out[key] = ".".join(path) if path else "<root>"
    return out


def project(array: Array, *, on_fail: str = "raise") -> Projection:
    """Compute the columns each source must read for ``array`` (metadata-only)."""
    session = array.session
    backend = session.backend

    reports: dict[int, tuple[object, dict[str, str], str]] = {}
    tracers: dict[int, ak.Array] = {}
    for nid in session.source_ids():
        form = session.form_of(nid).tt.layout.form_with_key(form_key=f"s{nid}-n{{id}}")  # type: ignore[attr-defined]
        layout, report = ak.typetracer.typetracer_with_report(form, highlevel=False)
        tracers[nid] = ak.Array(layout)
        reports[nid] = (report, _leaf_columns(form), session.source_name(nid))

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

    read_columns: dict[str, frozenset[str]] = {}
    for report, key_map, name in reports.values():
        if conservative:
            cols = set(key_map.values())
        else:
            cols = {key_map[k] for k in report.data_touched if k in key_map}  # type: ignore[attr-defined]
        read_columns[name] = frozenset(cols)
    return Projection(read_columns)
