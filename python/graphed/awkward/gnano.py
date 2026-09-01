"""The nanoevents-flavored event-context constructor (§2.6).

The context MECHANISM — lineage, the ambient weight registry, the fill-inference seam — is neutral
and lives in `graphed` proper; this constructor is awkward-idiom, so it lives here (the §2.1
factorization rule). `is_data=True` is the explicit flag §2.6d guards on: the survey's universal
data special-casing, made explicit rather than inferred.
"""

from __future__ import annotations

from graphed import Array
from graphed.context import EventContext


def events(root: Array, *, is_data: bool = False) -> EventContext:
    """Wrap a root event record as an event context.

    `root` itself stays context-free — only reads performed THROUGH the returned context carry its
    handle (§2.3e's origination rule).
    """
    return EventContext(root.session, root, is_data=is_data)
