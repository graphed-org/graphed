"""`graphed.vary`: the one functional verb that registers variations (§2.1).

A neutral module verb — like `join`/`repartition`, never an `Array` method, never gak, never
numpy-idiom — with three overloads distinguished by the target: the loose primitive over an
`Array`/`Varied`, and the weight and shift forms over an event context. It NEVER mutates: the
result is a new object of the target's kind and the target stays valid and unchanged, because
variation history is object lineage (§2.6b).
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Mapping
from typing import Any

from . import accessors
from ._points import Point, coordinate, default, render
from ._tags import canonical_tag, numeric_value
from .array import Array
from .errors import GraphedError
from .varied import Varied, member_of, rebuild, registered_points, session_of

#: §4's fanout budget: the default grid a plain `vary` mints is bounded by this many universes, above
#: which the guard raises unless the analyst names `points=` or raises `max_universes=`. It sits above
#: the benchmark's legitimate 15 and below the accidental `3^N` chain runaway.
DEFAULT_MAX_UNIVERSES = 64


def vary(
    target: Array | Varied | Any,
    name: str,
    /,
    nominal: Array | Varied | None = None,
    *,
    is_weight: bool = False,
    variations: Mapping[Any, Any] | None = None,
    collections: Mapping[str, Mapping[Any, Any]] | None = None,
    points: Iterable[Mapping[str, Any]] | None = None,
    composes_as_union: bool = False,
    max_universes: int = DEFAULT_MAX_UNIVERSES,
    **tags: Any,
) -> Any:
    """Register a variation family `name` on `target`, returning a NEW object (§2.1).

    `**tags` and `variations=` carry tag/member pairs under the §1.1 grammar; the signature's own
    keyword names (`nominal`, `is_weight`, `variations`, `collections`, `points`) are legal tags AND
    legal collection names, so one so named arrives through a mapping channel instead.

    When a member is computed over another registered nuisance's varied nodes the family fans out to
    the full grid of joint universes automatically (§2). `composes_as_union=True` collapses it back to
    the one-at-a-time datacard union; `points=` is an ITERABLE of `{nuisance: coordinate}` maps that
    PRUNES the auto grid to a named subset (§3); `max_universes=` raises the §4 guard's budget.
    """
    if not isinstance(name, str) or not name.isidentifier():
        raise GraphedError(f"a variation name must be a Python identifier, got {name!r}")
    from .context import EventContext, vary_context  # noqa: PLC0415  (import cycle)

    overload: Any  # the two context overloads take one arg shape; mypy keeps its own narrowing
    if isinstance(target, Array | Varied):
        session = session_of(target)
        overload = _vary_loose
    elif isinstance(target, EventContext):
        session = target._session
        overload = vary_context
    else:
        raise GraphedError(
            f"graphed.vary takes an Array, a Varied or an event context, got {type(target).__name__}"
        )
    # §4.5's TRANSACTIONAL mint: everything that can raise after `gather_members` has minted — the
    # label-collision check, `check_members`, `_align`/`reindex_to` — must leave no binding behind,
    # or one failed call poisons a label for the life of the Session with no escape but a new one.
    saved = dict(session._points)
    try:
        return overload(
            target, name, nominal, is_weight, variations, collections, points,
            composes_as_union, max_universes, tags,
        )
    except BaseException:
        session._points.clear()
        session._points.update(saved)
        raise


def _vary_loose(
    target: Array | Varied,
    name: str,
    nominal: object,
    is_weight: bool,
    variations: Mapping[Any, Any] | None,
    collections: Mapping[str, Mapping[Any, Any]] | None,
    points: Iterable[Mapping[str, Any]] | None,
    composes_as_union: bool,
    max_universes: int,
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
    # the loose form's own carrier for §4.11-4 is the target it registers on
    one_at_a_time, joints = gather_members(
        name, tags, variations, inherited, points, session=session_of(target), carriers=(target,),
        composes_as_union=composes_as_union, max_universes=max_universes,
    )
    existing = dict(target._members) if isinstance(target, Varied) else {"nominal": target}
    # §4.6: registration resolves by the label's OWN point, so a supplied `Varied` contributes the
    # inner universe the point names instead of being flattened to its central one. A machine-minted
    # joint is already that inner cross node, so `member_of` on it is the identity.
    resolved = {
        label: member_of(member, label) for label, member in {**one_at_a_time, **joints}.items()
    }
    # BEFORE the row-space maps: a colliding label shadows its existing member in the merged
    # dict, so `check_members` never sees that member's handle and `_align` would work from a
    # handle the container does not really have.
    for label in resolved:
        if label in existing:
            raise GraphedError(f"variation label {label!r} is already carried by this container")
    handle = check_members({**existing, **resolved})
    existing = {label: _align(member, handle) for label, member in existing.items()}
    resolved = {label: accessors.reindex_to(member, handle) for label, member in resolved.items()}
    inherited_tags = dict(target._tags) if isinstance(target, Varied) else {}
    # a joint is a cross-coordinate, not a tag of this single family, so it stays out of `_tags`
    inherited_tags[name] = inherited + tuple(label[len(name) + 1 :] for label in one_at_a_time)
    return register(rebuild({**existing, **resolved}, tags=inherited_tags, context=handle))


def _align(member: Any, handle: Any) -> Any:
    """§2.1's one-row-space rule for overload (a)'s INHERITED members, the target included.

    Only across a `mask` or `project` link — the two kinds `_follow` acts on. A `vary` link is
    the identity in both row space and content, so re-indexing across it would do nothing but
    re-stamp the handle, losing the parent identity §2.3e pins on the member.
    """
    src = accessors.context_of(member)
    if src is None or src is handle:
        return member
    if not any(kind in ("mask", "project") for kind, _payload in handle._links_below(src)):
        return member
    return accessors.reindex_to(member, handle)


# ---- shared construction machinery (the context overloads use it too) ---------------------
def gather_members(
    name: str,
    tags: Mapping[str, Any],
    variations: Mapping[Any, Any] | None,
    inherited: tuple[str, ...],
    points: Iterable[Mapping[str, Any]] | None = None,
    *,
    session: Any,
    carriers: tuple[Any, ...] = (),
    composes_as_union: bool = False,
    max_universes: int = DEFAULT_MAX_UNIVERSES,
    composed: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The §1.1 tag channels as `(one_at_a_time, joints)`, both `{label: member}` (§2).

    Validation is CHANNEL-INDEPENDENT: literal kwarg syntax cannot spell a dotted or digit-leading
    tag, but `**`-unpacking admits any string key, so every channel takes the same rules.

    §4.5: this is the ONE place a label is minted, so it is also the one place a POINT is minted —
    the one-at-a-time default points and, when a member depends on a foreign nuisance's varied
    nodes, the joint points the fanout derives (§2).
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
    one_at_a_time = {f"{name}_{tag}": member for tag, member in canonical.items()}

    points_list = None if points is None else list(points)
    if composes_as_union:
        # §2.5: collapse every foreign coordinate to nominal — the pre-m53 union. There is nothing
        # for a `points=` selection to keep once the joints are gone, so pairing them is an error.
        if points_list:
            raise GraphedError(
                f"graphed.vary({name!r}) got both composes_as_union=True and a points= selection; "
                "the union collapses every joint away, so there is no joint for points= to keep"
            )
        _mint_defaults(name, tuple(canonical), session)
        return one_at_a_time, {}

    joints, joint_points = _fanout(name, canonical, carriers, composed)
    _mint_defaults(name, tuple(canonical), session)
    if joint_points:
        _check_unique(joint_points, session._points)
        session._points.update(joint_points)
    return one_at_a_time, joints


def _mint_defaults(name: str, tags: tuple[str, ...], session: Any) -> None:
    """§4.4: give every one-at-a-time label its default point `{name: tag}` and register it."""
    minted = {f"{name}_{tag}": default(name, tag) for tag in tags}
    _check_unique(minted, session._points)
    session._points.update(minted)


def _foreign(
    name: str, member: Any, carrier_nuisances: frozenset[str], composed: frozenset[str]
) -> dict[str, Point]:
    """§1: the foreign universes `member` genuinely depends on — `{foreign label: point}` over the
    member's own registered points, dropping the three cases that are NOT a dependency to fan out.

    A member that is not a `Varied`, or one whose only registered coordinates are its own family's,
    is INDEPENDENT (the union path, byte-identical to pre-m53). Beyond that, a foreign nuisance is
    dropped when it is:

    * **stacked** (`composed`): carried by the ambient weight AS A WEIGHT, so the weight form's
      label-aligned composition already resolves it into the union — fanning it out would
      double-count it through `_two_level(old, ...)`; or
    * a **spectator**: the carrier the family registers on is itself varied but NOT by this nuisance,
      so the member's foreign coordinate is incidental and collapses to nominal (the §2.1 stacking
      case). A nuisance the carrier DOES carry, or any nuisance when the carrier is unvaried, is the
      genuine cross-term the fanout mints.
    """
    if not isinstance(member, Varied):
        return {}
    out: dict[str, Point] = {}
    for label, point in registered_points(member).items():
        if label == "nominal":
            continue
        nuisances = frozenset(nuisance for nuisance, _ in point)
        if name in nuisances or nuisances & composed:
            continue
        if carrier_nuisances and not nuisances <= carrier_nuisances:
            continue
        out[label] = point
    return out


def _carrier_nuisances(carriers: tuple[Any, ...]) -> frozenset[str]:
    """The foreign nuisances the carriers a family registers on already vary (§1's spectator gate)."""
    return frozenset(
        nuisance
        for carrier in carriers
        if isinstance(carrier, Varied)
        for point in registered_points(carrier).values()
        for nuisance, _ in point
    )


def _fanout(
    name: str, canonical: Mapping[str, Any], carriers: tuple[Any, ...], composed: frozenset[str]
) -> tuple[dict[str, Any], dict[str, Point]]:
    """§2: the full grid of joint universes a dependent family mints — `(joints, joint_points)`.

    For each dependent tag `t` (in this call's canonical order) and each foreign universe `(fl, fp)`
    (in the member's own label order), the machine-minted joint label `f"{name}_{t}__{fl}"` binds the
    real cross node `member._members[fl]` and carries the merged point `{name: t, **fp}`. Both loops
    are over insertion-ordered dicts, so the sequence is a pure function of registration order.
    """
    carrier_nuisances = _carrier_nuisances(carriers)
    joints: dict[str, Any] = {}
    joint_points: dict[str, Point] = {}
    for tag, member in canonical.items():
        for fl, fp in _foreign(name, member, carrier_nuisances, composed).items():
            joint_label = f"{name}_{tag}__{fl}"
            joints[joint_label] = member._members[fl]
            joint_points[joint_label] = Point({**dict(default(name, tag)), **dict(fp)})
    return joints, joint_points


def _reachable(name: str, tags: tuple[str, ...], carriers: tuple[Any, ...]) -> dict[str, set[str]]:
    """§4.11-4's carrier walk: `{nuisance: {coordinate}}` over the family being registered in this
    call plus the carriers' own labels.

    Read through the REGISTRY's points over those labels, never through the carriers' `_tags` — a
    per-family map that legitimately omits inherited families, so a shift-then-weight ambient
    weight carries `jes_up` while its `_tags` has no `jes` key at all (§8-g).
    """
    found: dict[str, set[str]] = {name: {coordinate(tag) for tag in tags}}
    for carrier in carriers:
        if isinstance(carrier, Varied):
            for point in registered_points(carrier).values():
                for nuisance, value in point:
                    found.setdefault(nuisance, set()).add(value)
    return found


def _check_reachable(name: str, point: Point, reachable: Mapping[str, set[str]]) -> None:
    """§4.11-4: a TYPED coordinate names a real universe or the call fails naming what does.

    This is what stops `{"jes": 1}` typed against a family registered `up` from silently returning
    nominal kinematics, and a joint point registered before its axis exists from producing a
    one-axis universe wearing a joint name. INHERITED labels keep the silent fallback (§4.7):
    partial coverage is a legitimate pattern, a typed coordinate is not.
    """
    for nuisance, value in point:
        registered = reachable.get(nuisance)
        if registered is None:
            raise GraphedError(
                f"points= on graphed.vary({name!r}): nuisance {nuisance!r} is registered nowhere "
                f"this call can see; the registered nuisances are {sorted(reachable)}"
            )
        if value not in registered:
            raise GraphedError(
                f"points= on graphed.vary({name!r}): {value!r} is not a registered tag of nuisance "
                f"{nuisance!r}, whose tags are {sorted(registered)}"
            )


def _check_unique(minted: Mapping[str, Point], registry: Mapping[str, Point]) -> None:
    """§4.11-1/2: within a Session a label names one point and a point wears one label.

    Minting the same label with the same point — two independent containers each registering
    `vary(., "jes", up=.)` — is idempotent and stays legal.
    """
    for label, point in minted.items():
        seen = registry.get(label)
        if seen is not None and seen != point:
            raise GraphedError(
                f"variation label {label!r} already names the point {render(seen)} in this "
                f"Session, and this call names {render(point)}; one label names one universe"
            )
        for other, other_point in (*registry.items(), *minted.items()):
            if other != label and other_point == point:
                raise GraphedError(
                    f"point {render(point)} is already registered under label {other!r}, so label "
                    f"{label!r} would be a second name for one universe — two slots, two "
                    "StrCategory bins and two content hashes"
                )


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
