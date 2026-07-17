"""Graph-introspection helper for the M13 suite."""

from __future__ import annotations

import graphed_core
from graphed import Array, Session


def recorded(s: Session, arr: Array) -> dict[str, object]:
    """The (kind, name, params) of the node ``arr`` denotes, read back from the serialized IR."""
    g = graphed_core.GraphStore.deserialize(s.serialized_ir(arr, optimize=False))
    (node,) = [n for n in g.nodes() if n["id"] == arr.node_id]
    return node
