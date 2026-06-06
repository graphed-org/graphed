"""The IR interpreter — `reproduce` runs the preserved graph node-by-node (plan M9).

This is what makes ``inspect`` faithful to ``reproduce``: both consume the SAME canonical IR
(``GraphStore.nodes()``). Op/reduction nodes evaluate through the backend (graphed-awkward's real
awkward dispatch); ``External`` nodes evaluate through the payload-backed evaluators
(``externals.evaluate_external``); sources are resolved from the bundle's content-addressed archive.

The preserved IR is captured at ``opt_level=0`` (no M4 stage fusion), so it is 1:1 with the user's
ops and contains no ``stage`` nodes — the auditable form (plan M6/M9).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .errors import PreserveError


def run_ir(
    nodes: list[dict[str, Any]],
    *,
    source: Callable[[dict[str, Any]], Any],
    external: Callable[[dict[str, Any], list[Any]], Any],
    eval_op: Callable[[str, list[Any], dict[str, Any]], Any],
) -> dict[int, Any]:
    """Evaluate every node in id order; return ``node_id -> value``.

    ``source(node)`` resolves a source node's data; ``eval_op(name, inputs, params)`` runs an
    op/reduction (the backend); ``external(node, inputs)`` runs a payload-backed External node.
    """
    values: dict[int, Any] = {}
    for node in nodes:
        nid = node["id"]
        kind = node["kind"]
        if kind == "source":
            values[nid] = source(node)
        elif kind in ("op", "reduction"):
            ins = [values[i] for i in node["inputs"]]
            values[nid] = eval_op(node["name"], ins, node["params"])
        elif kind == "external":
            ins = [values[i] for i in node["inputs"]]
            values[nid] = external(node, ins)
        else:  # "stage" — only present in an M4-reduced graph; preservation keeps opt_level=0
            raise PreserveError(
                f"cannot interpret a {kind!r} node; preservation captures the unfused IR (opt_level=0)"
            )
    return values
