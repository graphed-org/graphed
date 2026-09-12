"""`graphed.varied` moved to `graphed.systematics.varied` (m57); this re-exports it.

Every name is the same object as `graphed.systematics.varied`'s, so an `isinstance` check or an
import through either path means exactly the same thing.
"""

from __future__ import annotations

from .systematics.varied import (  # isort: skip
    _BROADCAST_SURFACE as _BROADCAST_SURFACE,
    _VARIED_CLASSES as _VARIED_CLASSES,
    RESERVED as RESERVED,
    SURFACE_DISPOSITIONS as SURFACE_DISPOSITIONS,
    Member as Member,
    Varied as Varied,
    _broadcast_method as _broadcast_method,
    _collect as _collect,
    _refusing_method as _refusing_method,
    boundary_refusal as boundary_refusal,
    broadcasting as broadcasting,
    containers_in as containers_in,
    expand as expand,
    expand_tuple as expand_tuple,
    expanding as expanding,
    install_surface as install_surface,
    labels_of as labels_of,
    member_of as member_of,
    most_derived_context as most_derived_context,
    narrow as narrow,
    point_registry as point_registry,
    rebuild as rebuild,
    refuse_boundary as refuse_boundary,
    refuse_container as refuse_container,
    register_varied as register_varied,
    registered_points as registered_points,
    session_of as session_of,
    union_labels as union_labels,
    union_tags as union_tags,
    universes_of as universes_of,
    varied_class_for as varied_class_for,
)
