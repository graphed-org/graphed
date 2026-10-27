"""`graphed.varied` moved to `graphed.systematics.varied` (m57); this re-exports it.

Every name is the same object as `graphed.systematics.varied`'s, so an `isinstance` check or an
import through either path means exactly the same thing.
"""

from __future__ import annotations

from .systematics.varied import (  # isort: skip
    RESERVED as RESERVED,
    SURFACE_DISPOSITIONS as SURFACE_DISPOSITIONS,
    Member as Member,
    Varied as Varied,
    boundary_refusal as boundary_refusal,
    broadcasting as broadcasting,
    containers_in as containers_in,
    context_of as context_of,
    expand as expand,
    expand_tuple as expand_tuple,
    expanding as expanding,
    install_surface as install_surface,
    key_for as key_for,
    labels_of as labels_of,
    member_exact as member_exact,
    member_nodes as member_nodes,
    member_of as member_of,
    members_of as members_of,
    most_derived_context as most_derived_context,
    narrow as narrow,
    point_registry as point_registry,
    rebuild as rebuild,
    refuse_boundary as refuse_boundary,
    refuse_container as refuse_container,
    register_varied as register_varied,
    registered_points as registered_points,
    session_of as session_of,
    tags_of as tags_of,
    union_labels as union_labels,
    union_tags as union_tags,
    universes_of as universes_of,
    varied_class_for as varied_class_for,
)
