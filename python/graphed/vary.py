"""`graphed.vary`: the one functional verb that registers variations (§2.1).

A neutral module verb — like `join`/`repartition`, never an `Array` method, never gak, never
numpy-idiom — with three overloads distinguished by the target: the loose primitive over an
`Array`/`Varied`, and the weight and shift forms over an event context. It NEVER mutates: the
result is a new object of the target's kind and the target stays valid and unchanged, because
variation history is object lineage (§2.6b).
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from typing import Any

from . import accessors
from ._tags import canonical_tag, numeric_value
from .array import Array
from .errors import GraphedError
from .varied import Varied, rebuild


def vary(
    target: Array | Varied | Any,
    name: str,
    /,
    nominal: Array | Varied | None = None,
    *,
    is_weight: bool = False,
    variations: Mapping[Any, Any] | None = None,
    collections: Mapping[str, Mapping[Any, Any]] | None = None,
    **tags: Any,
) -> Any:
    """Register a variation family `name` on `target`, returning a NEW object (§2.1).

    `**tags` and `variations=` carry tag/member pairs under the §1.1 grammar; the signature's own
    keyword names (`nominal`, `is_weight`, `variations`, `collections`) are legal tags AND legal
    collection names, so one so named arrives through a mapping channel instead.
    """
    if not isinstance(name, str) or not name.isidentifier():
        raise GraphedError(f"a variation name must be a Python identifier, got {name!r}")
    if isinstance(target, Array | Varied):
        return _vary_loose(target, name, nominal, is_weight, variations, collections, tags)
    from .context import EventContext, vary_context  # noqa: PLC0415  (import cycle)

    if isinstance(target, EventContext):
        return vary_context(target, name, nominal, is_weight, variations, collections, tags)
    raise GraphedError(
        f"graphed.vary takes an Array, a Varied or an event context, got {type(target).__name__}"
    )


def _vary_loose(
    target: Array | Varied,
    name: str,
    nominal: object,
    is_weight: bool,
    variations: Mapping[Any, Any] | None,
    collections: Mapping[str, Mapping[Any, Any]] | None,
    tags: Mapping[str, Any],
) -> Varied:
    """Overload (a): the loose primitive. `is_weight=` and `nominal=` have no meaning here — a
    loose weight variation is just a `Varied` used in a `weight=[…]` factor list (§4.2)."""
    if is_weight:
        raise GraphedError(
            "is_weight=True needs an event-context target; a loose weight variation is a Varied "
            "passed in a fill's weight=[...] factor list"
        )
    if nominal is not None:
        raise GraphedError(
            "nominal= has no meaning on an Array or Varied target: the target IS the central universe"
        )
    if collections is not None:
        raise GraphedError("collections= needs an event-context target (the shift form)")
    inherited = target._tags.get(name, ()) if isinstance(target, Varied) else ()
    members = gather_members(name, tags, variations, inherited)
    existing = dict(target._members) if isinstance(target, Varied) else {"nominal": target}
    resolved = {label: central_universe(member) for label, member in members.items()}
    handle = check_members({**existing, **resolved})
    resolved = {label: accessors.reindex_to(member, handle) for label, member in resolved.items()}
    for label in resolved:
        if label in existing:
            raise GraphedError(f"variation label {label!r} is already carried by this container")
    inherited_tags = dict(target._tags) if isinstance(target, Varied) else {}
    inherited_tags[name] = inherited + tuple(label[len(name) + 1 :] for label in resolved)
    return register(rebuild({**existing, **resolved}, tags=inherited_tags, context=handle))


# ---- shared construction machinery (the context overloads use it too) ---------------------
def gather_members(
    name: str,
    tags: Mapping[str, Any],
    variations: Mapping[Any, Any] | None,
    inherited: tuple[str, ...],
) -> dict[str, Any]:
    """The §1.1 tag channels, canonicalized and family-checked, as `{label: member}`.

    Validation is CHANNEL-INDEPENDENT: literal kwarg syntax cannot spell a dotted or digit-leading
    tag, but `**`-unpacking admits any string key, so every channel takes the same rules.
    """
    raw: dict[Any, Any] = dict(tags)
    for tag, member in (variations or {}).items():
        if tag in raw:
            raise GraphedError(f"variation tag {tag!r} was given both as a keyword and in variations=")
        raw[tag] = member
    if not raw:
        raise GraphedError(f"graphed.vary({name!r}) needs at least one tag")
    canonical: dict[str, Any] = {}
    for tag, member in raw.items():
        canonical_form = canonical_tag(tag)
        if canonical_form in canonical:
            raise GraphedError(
                f"tags {tag!r} and one already given canonicalize to {canonical_form!r}: one value "
                "cannot name two universes"
            )
        canonical[canonical_form] = member
    check_family(name, inherited, tuple(canonical))
    return {f"{name}_{tag}": member for tag, member in canonical.items()}


def check_family(name: str, inherited: tuple[str, ...], added: tuple[str, ...]) -> None:
    """§1.1's family rule over the tags one `name` carries on one container, inherited included."""
    for tag in added:
        if tag in inherited:
            raise GraphedError(f"variation tag {tag!r} is already registered under {name!r}")
    seen: dict[Any, str] = {}
    for tag in (*inherited, *added):
        value = numeric_value(tag)
        if value is None:
            continue
        twin = seen.setdefault(value, tag)
        if twin != tag:
            raise GraphedError(
                f"variation tags {twin!r} and {tag!r} in family {name!r} name the same value "
                f"({value}); two labels for one universe would mean two bins and two content hashes"
            )


def central_universe(member: object) -> Any:
    """§2.1's member rule for overloads (a)/(c): a `Varied` member is reduced as SUPPLIED."""
    return accessors.nominal(member) if isinstance(member, Varied) else member


def check_members(labelled: Mapping[str, Any]) -> Any:
    """§2.1's construction checks over `{label: member}`: one Session, compatible forms, one
    source set, one ancestry chain. Returns the container's context handle (§2.3e's most-derived).

    Every rejection names the LABEL, so a §2.5 silent drop becomes a located construction error.
    """
    session = _flatten(labelled["nominal"])[0].session
    reference = session.form(_flatten(labelled["nominal"])[0])
    sources: set[frozenset[int]] = set()
    for label, member in labelled.items():
        for array in _flatten(member):
            if array.session is not session:
                raise GraphedError(f"variation {label}: its member records into another Session")
            sources.add(_source_ids(session, array))
            if len(sources) > 1:
                raise GraphedError(
                    f"variation {label}: its member roots in a different source; one container's "
                    "universes must describe one dataset"
                )
            if not _compatible(session, reference, session.form(array)):
                raise GraphedError(
                    f"variation {label}: its form {session.form(array)} is incompatible with the "
                    f"central universe's {reference}"
                )
    return accessors.unify_contexts(*(accessors.context_of(member) for member in labelled.values()))


def _flatten(member: object) -> list[Array]:
    if isinstance(member, Varied):
        return [array for item in member._members.values() for array in _flatten(item)]
    if isinstance(member, Array):
        return [member]
    raise GraphedError(f"a variation member must be an Array or a Varied, got {type(member).__name__}")


def _compatible(session: Any, reference: Any, form: Any) -> bool:
    """Form compatibility, backend-checked: identical forms pass outright, and otherwise the
    backend's own binary inference decides — so dtype promotion is fine while a record meeting a
    number is not."""
    if str(reference) == str(form):
        return True
    try:
        session.backend.op_form("add", [reference, form], {})
    except Exception:
        return False
    return True


def _source_ids(session: Any, array: Array) -> frozenset[int]:
    seen: set[int] = set()
    found: set[int] = set()
    stack = [array.node_id]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        if node_id in session._sources:
            found.add(node_id)
        elif node_id in session._externals:
            stack.extend(session._externals[node_id][1])
        else:
            stack.extend(session._ops[node_id][2])
    return frozenset(found)


def register(container: Varied) -> Varied:
    """§2.5: each container registers with its Session (weakly) so `compile_ir` can report a label
    that reaches no marked output, and each non-central member carries its label for that walk."""
    session = None
    for label, member in container._members.items():
        if label == "nominal":
            continue
        for array in _flatten(member):
            array._labels = (array._labels or frozenset()) | {label}
            session = array.session
    if session is not None:
        # The LABELS are held by value: a container the analysis discards is exactly the silent-cost
        # case the diagnostic exists to report, so it must outlive the weak reference to it.
        labels = tuple(label for label in container._members if label != "nominal")
        session._varied.append((labels, weakref.ref(container)))
    return container
