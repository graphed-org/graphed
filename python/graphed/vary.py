"""`graphed.vary` moved to `graphed.systematics.registration` (m57); this re-exports it.

Every name is the same object as `graphed.systematics.registration`'s, so an `isinstance` check or an
import through either path means exactly the same thing.
"""

from __future__ import annotations

from .systematics.registration import (  # isort: skip
    DEFAULT_MAX_UNIVERSES as DEFAULT_MAX_UNIVERSES,
    AmbientCarrier as AmbientCarrier,
    _align as _align,
    _bind_points as _bind_points,
    _carrier_nuisances as _carrier_nuisances,
    _carrier_points as _carrier_points,
    _check_reachable as _check_reachable,
    _check_unique as _check_unique,
    _compatible as _compatible,
    _fanout as _fanout,
    _flatten as _flatten,
    _foreign as _foreign,
    _guard as _guard,
    _member_nodes as _member_nodes,
    _mint_defaults as _mint_defaults,
    _parse_points as _parse_points,
    _reachable as _reachable,
    _reads_ambient as _reads_ambient,
    _route as _route,
    _source_ids as _source_ids,
    _vary_loose as _vary_loose,
    check_family as check_family,
    check_members as check_members,
    gather_members as gather_members,
    record_labels as record_labels,
    register as register,
    stamp_labels as stamp_labels,
    vary as vary,
)

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
