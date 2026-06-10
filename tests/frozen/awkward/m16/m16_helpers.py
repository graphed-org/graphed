"""Shared fixtures + graph introspection for the M16 suite."""

from __future__ import annotations

import awkward as ak
import graphed_core
from graphed import Array, Session

from graphed_awkward import AwkwardBackend, from_awkward

EVENTS = ak.Array(
    {
        "Jet": [
            [{"pt": 50.0, "eta": 0.5}, {"pt": 30.0, "eta": -1.2}],
            [],
            [{"pt": 70.0, "eta": 2.1}, {"pt": 20.0, "eta": 0.0}, {"pt": 10.0, "eta": -0.7}],
            [{"pt": 45.0, "eta": 1.1}],
        ],
        "met": [25.0, 60.0, 15.0, 90.0],
    }
)


def session_events() -> tuple[Session, Array, ak.Array]:
    s = Session(AwkwardBackend())
    return s, from_awkward(s, "events", EVENTS), EVENTS


def recorded(s: Session, arr: Array) -> dict[str, object]:
    """The (kind, name, params) of the node ``arr`` denotes, read back from the serialized IR."""
    g = graphed_core.GraphStore.deserialize(s.serialized_ir(arr, optimize=False))
    (node,) = [n for n in g.nodes() if n["id"] == arr.node_id]
    return node
