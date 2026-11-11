"""`graphed.vary`: the one functional verb that registers variations (§2.1).

A neutral module verb — like `join`/`repartition`, never an `Array` method, never gak, never
numpy-idiom — with overloads distinguished by the target: the loose primitive over an
`Array`/`Varied` (`registration.vary_loose`), the weight and shift forms over an event context
(`graphed.context.vary_context`), and a flat `ScalarFactor` over a number. It NEVER mutates: the
result is a new object of the target's kind and the target stays valid and unchanged, because
variation history is object lineage (§2.6b). This module is the target dispatch and the
transactional mint, above the overloads.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite
from numbers import Real
from typing import Any

from ..array import Array
from ..context import EventContext, vary_context
from ..errors import GraphedError, GraphedTypeError
from ..provenance import capture
from .lineage import context_session
from .registration import DEFAULT_MAX_UNIVERSES, vary_loose
from .varied import Varied, member_of, session_of


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


class ScalarFactor:
    """A flat factor: `array * factor` registers its family on `array`.

    It holds numbers, no node and no Session, so one factor serves every Session it meets. Its
    members scale the array's nominal alone, which keeps the family independent of whatever else
    varies there. Each number is an absolute value of the factor.
    """

    __slots__ = ("_members", "_name", "_nominal")

    def __init__(self, nominal: Any, name: str, members: Mapping[str, Any]) -> None:
        """Hold the factor's ``nominal`` value and its ``members`` by tag, all finite numbers."""
        if not members or not all(map(_is_number, (nominal, *members.values()))):
            raise GraphedError(
                f"the scalar factor {name!r} needs number members, as in "
                f"graphed.vary(1.0, {name!r}, up=1.025, down=0.975), got {dict(members)!r}"
            )
        self._nominal = float(nominal)
        self._name = name
        self._members = {tag: float(value) for tag, value in members.items()}
        if not all(map(isfinite, (self._nominal, *self._members.values()))):
            raise GraphedError(f"the scalar factor {name!r} needs finite numbers, got {self!r}")

    def __repr__(self) -> str:
        """Show what `graphed.vary` was given."""
        members = "".join(f", {tag}={value!r}" for tag, value in self._members.items())
        return f"graphed.vary({self._nominal!r}, {self._name!r}{members})"

    def __getattr__(self, name: str) -> Any:
        """Refuse what only an array answers, naming the way to one."""
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(
            f"the scalar factor {self._name!r} is no array and has no {name!r}; multiply it "
            "into one"
        )

    def __mul__(self, other: object) -> Any:
        """Register the family on ``other``, an array or a `Varied`."""
        if not isinstance(other, Array | Varied):
            raise GraphedError(
                f"the scalar factor {self._name!r} multiplies an Array or a Varied, got "
                f"{type(other).__name__}"
            )
        nominal = member_of(other, "nominal")
        points = {tag: nominal * value for tag, value in self._members.items()}
        target: Any = other
        if self._nominal != 1.0:
            target = target * self._nominal
        return vary(target, self._name, points=points)

    __rmul__ = __mul__


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
    """Register a variation family `name` on `target`, returning a NEW object.

    `**tags` and `points=` carry tag/member pairs under the variation-tag grammar; the signature's
    own keyword names (`nominal`, `is_weight`, `points`, `collections`) are legal tags AND legal
    collection names, so one so named arrives through a mapping channel instead.

    `points=` is either a `Mapping[tag, member]` (declares only, the common case) or an iterable
    mixing 2-`tuple` `(tag, member)` declares with `{nuisance: coordinate}` placement mappings that
    SELECT a derived joint (dependent member) or RE-POINT a label off-grid (independent member).
    When a member is computed over another registered nuisance's varied nodes the family fans out
    to the full grid of joint universes automatically. `composes_as_union=True` collapses it back
    to the one-at-a-time datacard union; `max_universes=` raises the universe-count guard's budget.

    A number target, `graphed.vary(1.0, "lumi", up=1.025, down=0.975)`, is a flat factor to
    multiply into an array, returned as a `ScalarFactor`; its members are numbers.

    >>> import awkward as ak
    >>> import graphed
    >>> from graphed.awkward import from_awkward, gak, gnano
    >>> data = ak.Array({"Jet": [[{"pt": 40.0}, {"pt": 12.0}], [{"pt": 55.0}]]})
    >>> events = gnano.events(from_awkward("events", data))
    >>> events = graphed.vary(events, "jes", collections={"Jet": {
    ...     "up": gak.with_field(events.Jet, events.Jet.pt * 1.05, "pt"),
    ...     "down": gak.with_field(events.Jet, events.Jet.pt * 0.95, "pt")}})
    >>> lead = gak.max(events.Jet.pt, axis=1)
    >>> graphed.systematics.labels(lead)
    ('nominal', 'jes_up', 'jes_down')
    >>> graphed.materialize(graphed.systematics.universe(lead, "jes_up")).tolist()
    [42.0, 57.75]
    """
    if not isinstance(name, str) or not name.isidentifier():
        raise GraphedError(f"a variation name must be a Python identifier, got {name!r}")
    if _is_number(target):
        options = nominal, points, collections
        if (
            any(option is not None for option in options)
            or is_weight
            or composes_as_union
            or max_universes != DEFAULT_MAX_UNIVERSES
        ):
            raise GraphedError(f"the scalar factor {name!r} takes only its members, up=1.025")
        return ScalarFactor(target, name, tags)
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
            "graphed.vary takes an Array, a Varied, an event context or a number, got "
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
