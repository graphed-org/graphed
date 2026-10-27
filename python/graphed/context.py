"""The §2.6 event context: systematics attach to `events`, functionally.

A frontend wrapper over the root event record carrying (a) the collections and (b) an ambient
event-weight registry. Pure sugar over §§2.1-2.5 — no IR change, §3.1 intact.

Three properties do the work. Contexts are IMMUTABLE, so `graphed.vary` returns a new one and a
fill from a pre-`vary` context is unaffected by later calls by construction. They RESERVE NO
NAMES, so a tree branch called `weights` or `vary` stays reachable and every graphed operation on
a context is a module function. And a read performed THROUGH a context yields THAT context's row
space, which is what makes `sel.Jet` mean `events.Jet` re-indexed by `sel`'s derivation mask,
label-aligned when that mask is `Varied`.

The neutral mechanism lives here; the nanoevents-flavored constructor is awkward-idiom and lives
in `graphed.awkward.gnano` (the §2.1 factorization rule).

A context's lineage is `graphed.systematics.lineage`'s `ContextNode`, and its ambient weight is an
`AmbientRecord` that `graphed.systematics.ambient` builds and decides over; this module builds the
contexts. `graphed.explain` is `graphed.systematics.explain`. Both are re-exported here, which is
where the probes and the extra suites read them from.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from typing import Any

from .array import Array
from .errors import GraphedError
from .provenance import capture
from .systematics import accessors

# The ambient-weight machinery lives in `graphed.systematics` (m57). Every name is re-exported
# under its own name because `graphed.context` is where the probes, the extra suites and the docs
# read the machinery from.
from .systematics.ambient import (  # isort: skip
    AmbientRecord,
    Memo,
    context_labels,
    _SLOT as _SLOT,
    Rider as Rider,
    _all_settled as _all_settled,
    _compose as _compose,
    _covering_overlay as _covering_overlay,
    _extension as _extension,
    _lineage_factors as _lineage_factors,
    _lineage_reads as _lineage_reads,
    _live_factors as _live_factors,
    _live_riders as _live_riders,
    _live_slots as _live_slots,
    _marks as _marks,
    _owned_projection as _owned_projection,
    _placed_elsewhere as _placed_elsewhere,
    _project_riders as _project_riders,
    _Read as _Read,
    _Registration as _Registration,
    _same_node as _same_node,
    _two_level as _two_level,
    _union as _union,
    _union_nominal as _union_nominal,
    adopt,
    ambient_entries as ambient_entries,
    fold,
    seed,
    shift_form,
    weight_form,
    weight_of,
)
from .systematics.by_label import cone as cone  # isort: skip
from .systematics.lineage import (  # isort: skip
    ContextNode,
    Link as Link,
    ambient_of,
    collections_of,
    context_session,
)
from .systematics.explain import Explanation as Explanation, explain as explain  # isort: skip
from .systematics.registration import stamp_labels as stamp_labels  # isort: skip
from .systematics.varied import (  # isort: skip
    Varied as Varied,
    expand as expand,
    labels_of as labels_of,
    member_nodes as member_nodes,
    member_of as member_of,
    members_of as members_of,
    point_registry as point_registry,
    rebuild as rebuild,
    tags_of as tags_of,
)

#: contexts are compared by IDENTITY (§2.6b), so a divergence error must name two distinguishable
#: objects; the serial plus the user line where the context was built is that name
_SERIAL = itertools.count()


class EventContext(ContextNode):
    """An event context (§2.6). Built by an idiom constructor, never directly.

    Its RECORDED state — collections, selection, the registered weight factors and the label union
    over them — never changes after construction, which is what makes `graphed.vary` return a new
    context and a fill from a pre-`vary` one unaffected. The ambient weight is composed from that
    recorded state on demand (`graphed.systematics.ambient`), and its memo is a pure CACHE stamped
    with the Session's mint epoch: a mint landing after a composition makes the next read remake
    it, so what a factor contributes at a label is what the registry says AS OF THE READ.
    """

    __slots__ = ("_derived", "_is_data", "_projected", "_provenance", "_record", "_serial")

    def __init__(
        self,
        session: Any,
        record: Array | Varied,
        *,
        is_data: bool = False,
        parent: EventContext | None = None,
        link: Link | None = None,
        collections: Mapping[str, Any] | None = None,
        weight: Varied | None = None,
    ) -> None:
        """Wrap ``record`` as an event context, derived from ``parent`` by ``link``."""
        super().__init__(session, parent, link, collections)
        self._record = record
        self._is_data = is_data
        self._ambient = seed(self, weight, parent)
        self._serial = next(_SERIAL)
        self._provenance = capture()
        self._derived: dict[tuple[tuple[str, int], ...], EventContext] = {}
        self._projected: dict[str, EventContext] = {}

    # the extra suites read these three of the ambient record
    @property
    def _factors(self) -> list[Any]:
        return ambient_of(self).factors

    @property
    def _recorded(self) -> tuple[str, ...]:
        return ambient_of(self).recorded

    @property
    def _memo(self) -> Memo | None:
        return ambient_of(self).memo

    def __repr__(self) -> str:
        """Show the context's serial and where it was made."""
        return f"EventContext(#{self._serial} from {self._provenance})"

    # ---- tree content (§2.6a: the context reserves NO names) --------------------
    def __getattr__(self, name: str) -> Any:
        """Read field ``name`` of the event record."""
        if name.startswith("_"):
            raise AttributeError(name)
        return self._read(name)

    def __getitem__(self, key: object) -> Any:
        """Read a field or field list, or derive the context a mask selects."""
        if isinstance(key, str):
            return self._read(key)
        if isinstance(key, list) and key and all(isinstance(field, str) for field in key):
            return self._stamp(expand(lambda record: record[key], (self._record,), {}))
        if isinstance(key, Array | Varied):
            return self._derive(key)
        raise GraphedError(
            f"an event context takes a field name, a list of field names, or an Array/Varied mask "
            f"as its subscript, not {key!r}"
        )

    def __iter__(self) -> Any:
        """Refuse iteration over an event context."""
        raise TypeError(
            "an event context is not iterable; subscript it with a mask or a field name"
        )

    def _read(self, name: str) -> Any:
        value = collections_of(self).get(name)
        if value is None:
            value = expand(lambda record: getattr(record, name), (self._record,), {})
        return self._stamp(value)

    def _stamp(self, value: Any) -> Any:
        """§2.3e's ORIGINATION rule.

        Everything a context produces carries THAT context's handle, overriding whatever the input
        merge would yield (a derived context's reads go through the same root wrapper, so the merge
        alone would answer with the parent's handle).
        """
        return accessors.with_context(value, self)

    # ---- derivation (§2.6c: scoping is lineage) ---------------------------------
    def _derive(self, mask: Array | Varied) -> EventContext:
        """`ctx[mask]`.

        PURE DERIVATIONS ARE CANONICAL — the same mask answers with the same object, memoised here,
        or two reads of one universe would falsely trip §2.3e's divergence rule.
        """
        key = _mask_key(mask)
        memo = self._derived.get(key)
        if memo is not None:
            return memo
        child = EventContext(
            context_session(self),
            expand(lambda record, on: record[on], (self._record, mask), {}),
            is_data=self._is_data,
            parent=self,
            link=("mask", mask),
        )
        # every member re-indexed by THAT label's own mask, label-aligned per §2.4
        child._collections = {
            name: child._stamp(expand(lambda value, on: value[on], (collection, mask), {}))
            for name, collection in collections_of(self).items()
        }
        ambient = weight_of(self)
        if ambient is not None:
            selected = expand(lambda value, on: value[on], (ambient, mask), {})
            adopt(child, self, child._stamp(selected))
        child._record = child._stamp(child._record)
        self._derived[key] = child
        return child

    def _vary_child(self, record: AmbientRecord, replaced: Mapping[str, Any]) -> EventContext:
        """Return a `vary` link: the row space unchanged, `replaced` collections, `record` (§6.1d).

        Stamping records no node, so the varied collections are stamped here, after the form built
        them.
        """
        child = EventContext(
            context_session(self),
            self._record,
            is_data=self._is_data,
            parent=self,
            link=("vary", None),
            collections=collections_of(self),
        )
        child._ambient = record
        if replaced:
            child._collections = {
                **collections_of(self),
                **{name: child._stamp(value) for name, value in replaced.items()},
            }
        child._record = child._stamp(child._record)
        return child

    def _project(self, label: str) -> EventContext:
        """`graphed.universe(ctx, L)` / `graphed.nominal(ctx)`.

        A CHILD context carrying that label's collections and ambient weight (§2.2).
        """
        carried = context_labels(self)
        if label not in carried:
            raise KeyError(
                f"unknown variation label {label!r}; this context carries {list(carried)}"
            )
        memo = self._projected.get(label)
        if memo is not None:
            return memo
        child = EventContext(
            context_session(self),
            member_of(self._record, label),
            is_data=self._is_data,
            parent=self,
            link=("project", label),
        )
        child._collections = {
            name: child._stamp(member_of(collection, label))
            for name, collection in collections_of(self).items()
        }
        ambient = weight_of(self)
        adopt(
            child,
            self,
            None if ambient is None else child._stamp(member_of(ambient, label)),
            projected=label,
        )
        child._record = child._stamp(child._record)
        self._projected[label] = child
        return child


def project(ctx: EventContext, label: str) -> EventContext:
    """Return `ctx`'s child context for universe ``label`` (`graphed.universe`/`nominal`)."""
    return ctx._project(label)


def _mask_key(mask: Array | Varied) -> tuple[tuple[str, int], ...]:
    """Return a derivation's identity: its per-label node ids.

    A REBUILT mask interns to the same ids, which is the binding condition for
    `ctx[rebuilt] is ctx[mask]`.
    """
    if isinstance(mask, Varied):
        return tuple((label, member.node_id) for label, member in members_of(mask).items())
    return (("nominal", mask.node_id),)


# ---- the two context overloads of `graphed.vary` -------------------------------------------
def vary_context(
    ctx: EventContext,
    name: str,
    nominal: object,
    is_weight: bool,
    variations: Mapping[Any, Any] | None,
    collections: Mapping[str, Mapping[Any, Any] | Varied] | None,
    points: Iterable[Mapping[str, Any]] | None,
    composes_as_union: bool,
    max_universes: int,
    tags: Mapping[str, Any],
) -> EventContext:
    """Register variation ``name`` on ``ctx`` and return the context it derives."""
    if ctx._is_data:
        raise GraphedError(
            f"variation {name!r} cannot be registered on a data context: data fills nominal-only, "
            "and accepting a registration whose labels the fill then drops would be a silent drop"
        )
    if is_weight:
        record, operands = weight_form(
            ctx,
            name,
            nominal,
            variations,
            collections,
            points,
            composes_as_union,
            max_universes,
            tags,
        )
        child = ctx._vary_child(record, {})
        if operands is not None:
            fold(child, operands)
        return child
    replaced, record = shift_form(
        ctx, name, nominal, variations, collections, points, composes_as_union, max_universes, tags
    )
    return ctx._vary_child(record, replaced)
