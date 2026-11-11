"""The nanoevents-flavored event-context constructor.

The context MECHANISM — lineage, the ambient weight registry, the fill-inference seam — is neutral
and lives in `graphed` proper; this constructor is awkward-idiom, so it lives here. `is_data=True`
is the explicit flag that refuses registering a variation on a data context: the survey's universal
data special-casing, made explicit rather than inferred.
"""

from __future__ import annotations

from graphed import Array
from graphed.context import EventContext


def events(root: Array, *, is_data: bool = False) -> EventContext:
    """Wrap a root event record as an event context.

    `root` itself stays context-free — only reads performed THROUGH the returned context carry its
    handle.

    >>> import awkward as ak
    >>> import graphed
    >>> from graphed.awkward import from_awkward, gnano
    >>> data = ak.Array({"Jet": [[{"pt": 40.0}, {"pt": 12.0}], [{"pt": 55.0}]]})
    >>> events = gnano.events(from_awkward("events", data))
    >>> graphed.materialize(events.Jet.pt).tolist()
    [[40.0, 12.0], [55.0]]
    """
    return EventContext(root.session, root, is_data=is_data)
