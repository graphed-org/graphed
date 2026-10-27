"""`graphed.vary`: the one functional verb that registers variations (§2.1).

A neutral module verb — like `join`/`repartition`, never an `Array` method, never gak, never
numpy-idiom — with three overloads distinguished by the target: the loose primitive over an
`Array`/`Varied` (`registration.vary_loose`), and the weight and shift forms over an event context
(`graphed.context.vary_context`). It NEVER mutates: the result is a new object of the target's
kind and the target stays valid and unchanged, because variation history is object lineage
(§2.6b). This module is the target dispatch and the transactional mint, above both overloads.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..array import Array
from ..context import EventContext, vary_context
from ..errors import GraphedError, GraphedTypeError
from ..provenance import capture
from .lineage import context_session
from .registration import DEFAULT_MAX_UNIVERSES, vary_loose
from .varied import Varied, session_of


def vary(
    target: Array | Varied | Any,
    name: str,
    /,
    nominal: Array | Varied | None = None,
    *,
    is_weight: bool = False,
    points: Mapping[Any, Any] | Iterable[Any] | None = None,
    collections: Mapping[str, Mapping[Any, Any] | Varied] | None = None,
    composes_as_union: bool = False,
    max_universes: int = DEFAULT_MAX_UNIVERSES,
    **tags: Any,
) -> Any:
    """Register a variation family `name` on `target`, returning a NEW object (§2.1).

    `**tags` and `points=` carry tag/member pairs under the §1.1 grammar; the signature's own
    keyword names (`nominal`, `is_weight`, `points`, `collections`) are legal tags AND legal
    collection names, so one so named arrives through a mapping channel instead.

    `points=` is either a `Mapping[tag, member]` (declares only, the common case) or an iterable
    mixing 2-`tuple` `(tag, member)` declares with `{nuisance: coordinate}` placement mappings that
    SELECT a derived joint (dependent member) or RE-POINT a label off-grid (independent member)
    (§2/§3). When a member is computed over another registered nuisance's varied nodes the family
    fans out to the full grid of joint universes automatically (§2). `composes_as_union=True`
    collapses it back to the one-at-a-time datacard union; `max_universes=` raises the §4 guard's
    budget.
    """
    if not isinstance(name, str) or not name.isidentifier():
        raise GraphedError(f"a variation name must be a Python identifier, got {name!r}")
    declares, placements = _parse_points(points)
    overload: Any  # the two context overloads take one arg shape; mypy keeps its own narrowing
    if isinstance(target, Array | Varied):
        session = session_of(target)
        overload = vary_loose
    elif isinstance(target, EventContext):
        session = context_session(target)
        overload = vary_context
    else:
        raise GraphedError(
            "graphed.vary takes an Array, a Varied or an event context, got "
            f"{type(target).__name__}"
        )
    # §4.5's TRANSACTIONAL mint: everything that can raise after `gather_members` has minted — the
    # label-collision check, `check_members`, `_align`/`reindex_to` — must leave no binding behind
    with session.systematics.transaction():
        return overload(
            target,
            name,
            nominal,
            is_weight,
            declares,
            collections,
            placements,
            composes_as_union,
            max_universes,
            tags,
        )


def _parse_points(
    points: Mapping[Any, Any] | Iterable[Any] | None,
) -> tuple[Mapping[Any, Any] | None, list[Mapping[str, Any]] | None]:
    """Split the unified `points=` surface into the internal (declares, placements) channels.

    A `Mapping` (or `None`) is declares-only — today's path, passed straight through. A non-Mapping
    iterable mixes 2-`tuple` `(tag, member)` declares with `{nuisance: coordinate}` placement
    mappings; an entry that is neither is ill-typed input (`GraphedTypeError`). The result feeds
    the old two-parameter seam unchanged, so `gather_members`/`_route` stay byte-identical.
    """
    if points is None or isinstance(points, Mapping):
        return points, None
    declares: dict[Any, Any] = {}
    placements: list[Mapping[str, Any]] = []
    for entry in points:
        if isinstance(entry, Mapping):
            placements.append(entry)
        elif isinstance(entry, tuple) and len(entry) == 2:
            tag, member = entry
            declares[tag] = member
        else:
            raise GraphedTypeError(
                "vary",
                capture(),
                f"a points= entry must be a (tag, member) declare 2-tuple or a "
                f"{{nuisance: coordinate}} placement mapping, got {entry!r}",
            )
    return (declares or None), (placements or None)
