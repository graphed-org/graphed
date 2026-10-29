"""The per-task read list of a parquet write: syntactic field accesses refined by projection.

The recorded graph's SYNTACTIC source-field accesses decide which fields must exist (R15.3); the
buffer projection decides which leaves each one reads (R15.8).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .backend import AwkwardForm
from .projection import project_buffers


def _syntactic_fields(array: Any, source_node_id: int) -> set[str] | None:
    """Return the top-level source fields the recorded graph ACCESSES (a session walk).

    ``None`` means the whole source is consumed (a bare source, or a non-field op applied to it
    directly).
    """
    found: set[str] = set()
    whole = [False]
    sentinel = object()

    def on_source(nid: int) -> object:
        return (sentinel, nid)

    def touches_source(ins: list[object]) -> bool:
        return any(
            isinstance(x, tuple) and len(x) == 2 and x[0] is sentinel and x[1] == source_node_id
            for x in ins
        )

    def on_op(_nid: int, name: str, ins: list[object], params: Any) -> object:
        if touches_source(ins):
            if name == "field":
                found.add(str(params["field"]))
            elif name == "fields":
                found.update(f for f in str(params["fields"]).split(",") if f)
            else:
                whole[0] = True
        return None

    def on_external(_nid: int, _fn: Any, ins: list[object]) -> object:
        """Return an External handed the source RECORD replays against every column.

        So the read list cannot narrow — the same rule `on_op` applies to a non-field op.
        """
        if touches_source(ins):
            whole[0] = True
        return None

    array.session.walk(array, source=on_source, op=on_op, external=on_external)
    return None if (whole[0] or not found) else found


def _evaluation_columns(
    array: Any, source_node_id: int, source_name: str, source_form: AwkwardForm
) -> tuple[str, ...]:
    """Return the per-task read list.

    The graph's SYNTACTIC source-field accesses, refined per field by the buffer projection.

    Evaluation replays EVERY recorded node — including field accesses whose buffers the output
    never touches (a zip's untouched legs) — so the syntactic set decides WHICH fields must
    exist; the buffer view then decides which LEAVES to read for each: DATA needs read their
    leaves, an offsets-only field reads its CHEAPEST CARRIER leaf (parquet has no standalone
    counter column — R15.8's translation). An empty tuple means "everything" (the whole source
    is consumed).
    """
    accessed = _syntactic_fields(array, source_node_id)
    if accessed is None:
        return ()
    leaves = source_form.tt.layout.form.columns()
    needs = project_buffers(array, on_fail="pass").buffers_for(source_name)
    out: set[str] = set()
    for f in sorted(accessed):
        data_paths = [
            p
            for p, need in needs.items()
            if need.value == "data" and (p == f or p.startswith(f + "."))
        ]
        if data_paths:
            for p in data_paths:
                out.update(c for c in leaves if c == p or c.startswith(p + "."))
        else:
            under = sorted(c for c in leaves if c == f or c.startswith(f + "."))
            out.add(under[0] if under else f)
    return tuple(sorted(out))


def _evaluation_columns_union(
    outputs: Sequence[Any], source_node_id: int, source_name: str, source_form: AwkwardForm
) -> tuple[str, ...]:
    """Return the read list widened over EVERY marked output (§6.4f).

    The union of each output's projected source-field accesses, `()` (read everything) absorbing
    when any output consumes the whole source.
    """
    union: set[str] = set()
    for out in outputs:
        cols = _evaluation_columns(out, source_node_id, source_name, source_form)
        if not cols:
            return ()
        union.update(cols)
    return tuple(sorted(union))
