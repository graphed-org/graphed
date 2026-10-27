"""`graphed.vary` moved to `graphed.systematics.verb` (m57); this re-exports it.

Every name is the same object as the one `graphed.systematics` defines, so an `isinstance` check or
an import through either path means exactly the same thing.
"""

from __future__ import annotations

from .systematics.registration import (  # isort: skip
    DEFAULT_MAX_UNIVERSES as DEFAULT_MAX_UNIVERSES,
    AmbientCarrier as AmbientCarrier,
    check_family as check_family,
    check_members as check_members,
    gather_members as gather_members,
    record_labels as record_labels,
    register as register,
    stamp_labels as stamp_labels,
)
from .systematics.verb import vary as vary  # isort: skip

# the names `graphed.vary` also carried, from where they are defined
from .systematics import accessors as accessors  # isort: skip
from .systematics.by_label import cone as cone  # isort: skip
from .systematics.varied import (  # isort: skip
    Varied as Varied,
    member_of as member_of,
    rebuild as rebuild,
    registered_points as registered_points,
    session_of as session_of,
)
