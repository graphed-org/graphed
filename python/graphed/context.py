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

The context KEEPS the ambient weight's state — the operation list, its slots, their riders, the
reads and the memo — and every rule that decides over it is `graphed.systematics.ambient`;
`graphed.explain` is `graphed.systematics.explain`. Both are re-exported here, which is where the
probes and the extra suites read them from.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .array import Array
from .errors import GraphedError
from .provenance import capture
from .systematics import accessors

# The ambient-weight machinery lives in `graphed.systematics` (m57); the context KEEPS its state and
# calls into it. Every name is re-exported under its own name because `graphed.context` is where the
# probes, the extra suites and the docs read the machinery from.
from .systematics.ambient import (  # isort: skip
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
    _vary_shift as _vary_shift,
    _vary_weight as _vary_weight,
    ambient_entries as ambient_entries,
)
from .systematics.by_label import cone as cone  # isort: skip
from .systematics.explain import Explanation as Explanation, explain as explain  # isort: skip
from .systematics.registration import _member_nodes as _member_nodes, stamp_labels as stamp_labels  # isort: skip
from .systematics.varied import (  # isort: skip
    Varied as Varied,
    expand as expand,
    labels_of as labels_of,
    member_of as member_of,
    point_registry as point_registry,
    rebuild as rebuild,
)

#: contexts are compared by IDENTITY (§2.6b), so a divergence error must name two distinguishable
#: objects; the serial plus the user line where the context was built is that name
_SERIAL = itertools.count()

#: ("mask", mask) derives rows, ("vary", None) only registrations, ("project", label) narrows to
#: one universe — §6.1d's three link kinds, in one place
Link = tuple[str, Any]


class EventContext:
    """An event context (§2.6). Built by an idiom constructor, never directly.

    Its RECORDED state — collections, selection, the registered weight factors and the label union
    over them — never changes after construction, which is what makes `graphed.vary` return a new
    context and a fill from a pre-`vary` one unaffected. The ambient weight is composed from that
    recorded state on demand, and `_memo` is a pure CACHE of it stamped with the Session's mint
    epoch: a mint landing after a composition makes the next read remake it from the same original
    factors, so what a factor contributes at a label is what the registry says AS OF THE READ.
    """

    __slots__ = (
        "_adopted", "_collections", "_derived", "_factors", "_gens", "_head", "_is_data",
        "_link", "_memo", "_origin", "_overlays", "_parent", "_projected",
        "_provenance", "_reads", "_record", "_recorded", "_registration", "_riders", "_serial",
        "_session", "_slots", "_weight_tags",
    )  # fmt: skip

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
        self._session = session
        self._record = record
        self._is_data = is_data
        self._parent = parent
        self._link = link
        self._collections: dict[str, Any] = dict(collections or {})
        # §2.6's ambient weight is a DEFERRED composition (§4): the registered factors in
        # registration order, the label union RECORDED at each registration, and the composed
        # container once something reads it. Composing at every registration is quadratic in the
        # families registered whatever the association, because each one rewrites every label the
        # ambient already carries.
        self._factors: list[Any] = [] if weight is None else [weight]
        #: the SLOT of each entry, minted at append and kept by a join, so a handle recorded over
        #: `(pu, hf)` still names those operations after `lf` joined the `hf` container — which
        #: `id()` could not, the join replacing the object the read was recorded over.
        self._slots: list[int] = [] if weight is None else [next(_SLOT)]
        #: whether `_factors[0]` is the composed container a row-space change ADOPTED, and so
        #: stands for the parent's own factors (`_live_factors` walks back through it)
        self._adopted = False
        #: §2.3: what that adopted head STANDS FOR — the parent's live factor slots and their
        #: generations at the adoption, the same tuple a read records. A record from across the
        #: change names the head when it equals this, and names nothing the child can anchor
        #: otherwise (a prefix of the parent's list, or a read from after that list grew).
        self._head: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        #: §2.1's ORDERED operations: a SLOT named here holds an OVERLAY — a relative-delta
        #: family whose members are the whole ambient, so the composition REPLACES the running
        #: value at its labels instead of multiplying.
        self._overlays: frozenset[int] = frozenset()
        #: §2.1's record of what a `graphed.weight()` READ handed out: `(the container's member
        #: nodes, the live FACTOR slots it composed, their generations)`. A central naming one of
        #: these is that composition, and the overlay lands right after the factors it was read
        #: over. The key is the WHOLE member map, not the nominal: a join leaves the nominal node
        #: untouched, so two handles read either side of one are told apart by nothing else.
        #: Overlays are not in the key: they never move the nominal, so inserting one among the
        #: factors leaves every earlier handle naming the same composition. SHARED down a `vary`
        #: link (one row space, so every id means the same value), replaced by a row-space change.
        self._reads: list[_Read] = []
        #: §2.1's staleness stamp: `slot -> generation`, bumped by a join whose union ADDS
        #: universes to that slot's nominal member. A handle read before such a join is no longer
        #: the composition it names, and a re-index (which changes every node but no universe)
        #: must not look like one — which is why this counts unions and not nodes.
        self._gens: dict[int, int] = {}
        #: §2.3's RIDERS: `slot -> Rider`, the provenance of each operation this context's list
        #: introduced, beside the slot and never on the entry. The ancestors' stay reachable
        #: through the adoption, exactly as their entries do (`_live_riders`).
        self._riders: dict[int, Rider] = (
            {}
            if weight is None
            else {
                self._slots[0]: Rider(
                    "factor",
                    dict(getattr(weight, "_tags", None) or {}),
                    priors=(_two_level(weight, "nominal"),),
                    home=self,
                )
            }
        )
        #: the weight families REGISTERED on this lineage, `{name: tags}` — the record `variations`
        #: and a same-name registration read, as opposed to the ambient container's tag map, which
        #: a row-space change widens with every shift the mask carries
        self._weight_tags: dict[str, tuple[str, ...]] = {} if parent is None else dict(parent._weight_tags)
        self._recorded: tuple[str, ...] = _union(("nominal",), labels_of(weight))
        #: the cache: `(mint epoch, factors it covers, composed container, settled)`, assigned as
        #: one immutable tuple so a concurrent reader sees either the old state or the new, both
        #: valid. `settled` says no mint can move a resolution it made, which is what lets a later
        #: registration fold onto it instead of remaking.
        self._memo: tuple[int, int, Varied, bool] | None = (
            None if weight is None else (session._mint_epoch, 1, weight, True)
        )
        # §2.3e's ORIGINATION handle for the ambient: the context that last CHANGED the factor
        # list, which is NOT the one that happens to read it first. Stamping the reader would make
        # a `graphed.weight(parent)` call observable in a later divergence check.
        #: §2.7: the `graphed.vary` call that produced THIS context, `None` for every other one.
        #: `explain` walks the lineage for them, which is what puts the families in registration
        #: order and names the context each was registered on.
        self._registration: _Registration | None = None
        self._origin: EventContext = self
        self._serial = next(_SERIAL)
        self._provenance = capture()
        self._derived: dict[tuple[tuple[str, int], ...], EventContext] = {}
        self._projected: dict[str, EventContext] = {}

    def __repr__(self) -> str:
        return f"EventContext(#{self._serial} from {self._provenance})"

    # ---- tree content (§2.6a: the context reserves NO names) --------------------
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._read(name)

    def __getitem__(self, key: object) -> Any:
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
        raise TypeError("an event context is not iterable; subscript it with a mask or a field name")

    def _read(self, name: str) -> Any:
        value = self._collections.get(name)
        if value is None:
            value = expand(lambda record: getattr(record, name), (self._record,), {})
        return self._stamp(value)

    def _stamp(self, value: Any) -> Any:
        """§2.3e's ORIGINATION rule: everything a context produces carries THAT context's handle,
        overriding whatever the input merge would yield (a derived context's reads go through the
        same root wrapper, so the merge alone would answer with the parent's handle)."""
        return accessors.with_context(value, self)

    # ---- lineage (§2.6b: variation history is object lineage) -------------------
    def _is_ancestor_of(self, other: EventContext) -> bool:
        node: EventContext | None = other
        while node is not None:
            if node is self:
                return True
            node = node._parent
        return False

    def _links_below(self, ancestor: EventContext) -> tuple[Link, ...]:
        """The links from just below `ancestor` down to `self`, parent-to-child (§6.1d)."""
        links: list[Link] = []
        node: EventContext | None = self
        while node is not None and node is not ancestor:
            if node._link is not None:
                links.append(node._link)
            node = node._parent
        return tuple(reversed(links))

    def _selection(self) -> Any:
        """§9.1's selection: the mask that derived this context, skipping `vary` IDENTITY links
        and answering as of the first non-identity link (`None` at a root or across a projection,
        which is one label's unvaried member and contributes no labels)."""
        node: EventContext | None = self
        while node is not None and node._link is not None:
            kind, payload = node._link
            if kind == "mask":
                return payload
            if kind == "project":
                return None
            node = node._parent
        return None

    def _selection_bridge(self) -> Any:
        """§9.1's `graphed.selection(ctx)` — the FULL three-case bridge, NOT `_selection()`.

        Case 1 (mask-derived): the mask that derived this context, skipping `vary` identity links.
        Case 2 (universe/nominal-derived): that label's member of the PARENT's selection — an
        unvaried `Array` in the GRANDparent's row space, `None` when the parent is root (where
        `_selection()` returns `None`, the row-space/context reason §6.4a's universe/nominal REFUSE
        control relies on). Case 3 (root): `None`.
        """
        node: EventContext | None = self
        while node is not None and node._link is not None:
            kind, payload = node._link
            if kind == "mask":
                return payload
            if kind == "project":
                parent_selection = node._parent._selection() if node._parent is not None else None
                return None if parent_selection is None else member_of(parent_selection, payload)
            node = node._parent  # `vary` identity link: skip
        return None

    def _ambient_weight(self) -> Varied | Array | None:
        """The composed ambient weight, `None` while nothing is registered; a projection's is the
        one member it adopted (§2.2).

        A cache read: the memo answers when it still stands at the Session's mint epoch, and any
        mint since it was stamped makes this REMAKE the composition from the original factors, so
        every factor is resolved against the registry as of this read. The factor list is never
        replaced, which is what makes the remake possible and what makes inserting or removing a
        `graphed.weight()` read change no composed value.
        """
        if not self._factors:
            return None
        if len(self._factors) == 1 and not isinstance(self._factors[0], Varied):
            # §2.2: a projection adopted ONE resolved member and nothing registered after it, so
            # there is no universe left to resolve — composing would only wrap it in a one-label
            # container, and the read's type would then depend on what minted since the projection.
            # The adoption recorded this member as the child's read (§2.3), so handing it out again
            # records nothing new.
            adopted: Array = self._factors[0]
            return adopted
        epoch = self._session._mint_epoch
        memo = self._memo
        if memo is not None and memo[0] == epoch and memo[1] == len(self._factors):
            self._record_read(memo[2])
            return memo[2]
        composed, settled = self._materialise(self._ambient_operands())
        self._memo = (epoch, len(self._factors), composed, settled)
        self._record_read(composed)
        return composed

    def _record_read(self, composed: Any) -> None:
        """§2.1: the READ is what lets a later central name this composition. What is recorded is
        what a later registration compares — the member nodes the handle carries, the live factor
        slots it stands for and their generations — never the container, which a mint may remake.
        """
        slots = _live_slots(self)
        entry = (_member_nodes(composed), slots, tuple(self._gens.get(slot, 0) for slot in slots))
        if entry not in self._reads:
            self._reads.append(entry)

    def _overlay_ids(self) -> frozenset[int]:
        """The overlay slots as the OBJECT ids `_compose` tests, so the composition walk keeps
        working on the fold base, where the head is a composed container in no slot."""
        return _marks(self._factors, self._slots, self._overlays)

    def _fixed_ids(self) -> frozenset[int]:
        """The same, for the overlays a projection into their own universe FIXED (§2.3): there
        every value is that universe, so the composition replaces with them at every label."""
        return _marks(
            self._factors,
            self._slots,
            frozenset(slot for slot, rider in self._riders.items() if rider.fixed),
        )

    def _foldable(self) -> tuple[int, int, Varied, bool] | None:
        """The memo the next composition may build ON, `None` when it must remake from the
        original factors.

        One predicate for both readers — the operand list below and `_vary_weight`'s fold — so the
        record-time check walks exactly what the read will compose, in every memo state. A memo a
        mint has passed is stale; an UNSETTLED one is a composition a later mint can still move an
        operand of, and folding onto it would freeze a resolution the read remakes.
        """
        memo = self._memo
        if memo is None or memo[0] != self._session._mint_epoch or not memo[3]:
            return None
        return memo

    def _ambient_operands(self) -> list[Any]:
        """What the next composition multiplies: the composed container plus whatever registered
        after it while the memo may be built on, and the original factors otherwise.
        """
        memo = self._foldable()
        return list(self._factors) if memo is None else [memo[2], *self._factors[memo[1] :]]

    def _materialise(self, operands: Sequence[Any]) -> tuple[Varied, bool]:
        """Compose `operands` over the recorded union, with the verdict on whether a later mint
        could still move any resolution it just made (`_settled` over every operand and label)."""
        composed = rebuild(
            _compose(operands, self._recorded, overlays=self._overlay_ids(), fixed=self._fixed_ids()),
            tags=self._ambient_tags(),
            context=self._origin,
        )
        stamped = accessors.with_context(stamp_labels(composed), self._origin)
        return stamped, _all_settled(self._session, operands, self._recorded)

    def _ambient_tags(self) -> dict[str, tuple[str, ...]]:
        """The ambient's §1.1 tag map, accumulated over the factors in registration order — what
        today's composed container carries, read without composing. A projection's factor is a
        bare member carrying no map, which is how a projection still drops it."""
        merged: dict[str, tuple[str, ...]] = {}
        for factor in self._factors:
            merged.update(getattr(factor, "_tags", None) or {})
        return merged

    def _context_labels(self) -> tuple[str, ...]:
        """§2.2's §2.4-ordered union: (a) the ambient registry's labels, (b) the labels of the
        `Varied` collections this context CARRIES, (c) the labels of its selection."""
        out: dict[str, None] = dict.fromkeys(self._recorded)
        for source in (*self._collections.values(), self._selection()):
            for label in labels_of(source):
                out.setdefault(label, None)
        return tuple(out)

    # ---- derivation (§2.6c: scoping is lineage) ---------------------------------
    def _derive(self, mask: Array | Varied) -> EventContext:
        """`ctx[mask]`. PURE DERIVATIONS ARE CANONICAL — the same mask answers with the same
        object, memoised here, or two reads of one universe would falsely trip §2.3e's divergence
        rule."""
        key = _mask_key(mask)
        memo = self._derived.get(key)
        if memo is not None:
            return memo
        child = EventContext(
            self._session,
            expand(lambda record, on: record[on], (self._record, mask), {}),
            is_data=self._is_data,
            parent=self,
            link=("mask", mask),
        )
        # every member re-indexed by THAT label's own mask, label-aligned per §2.4
        child._collections = {
            name: child._stamp(expand(lambda value, on: value[on], (collection, mask), {}))
            for name, collection in self._collections.items()
        }
        if self._factors:
            ambient = self._ambient_weight()
            child._adopt_ambient(child._stamp(expand(lambda value, on: value[on], (ambient, mask), {})))
        child._record = child._stamp(child._record)
        self._derived[key] = child
        return child

    def _project(self, label: str) -> EventContext:
        """`graphed.universe(ctx, L)` / `graphed.nominal(ctx)`: a CHILD context carrying that
        label's collections and ambient weight (§2.2)."""
        if label not in self._context_labels():
            raise KeyError(
                f"unknown variation label {label!r}; this context carries {list(self._context_labels())}"
            )
        memo = self._projected.get(label)
        if memo is not None:
            return memo
        child = EventContext(
            self._session,
            member_of(self._record, label),
            is_data=self._is_data,
            parent=self,
            link=("project", label),
        )
        child._collections = {
            name: child._stamp(member_of(collection, label)) for name, collection in self._collections.items()
        }
        if self._factors:
            child._adopt_ambient(child._stamp(member_of(self._ambient_weight(), label)))
            if label != "nominal":
                child._riders = _project_riders(self, label)
        child._weight_tags = {}  # a projection drops the registry (§2.2)
        child._record = child._stamp(child._record)
        self._projected[label] = child
        return child

    def _adopt_ambient(self, composed: Any) -> None:
        """A row-space change MATERIALISES: the child starts from the one composed container, which
        is also its running form state. Re-indexing every factor instead would cost factors x
        labels per derivation — at the sizes this composition exists for, the same order as the
        whole quadratic it removes.
        """
        self._factors = [composed]
        self._slots = [next(_SLOT)]
        self._adopted = True
        self._overlays = frozenset()
        # the child's OWN reads start empty; the ancestors' stay reachable through the adoption
        # (`_lineage_reads`), because the head is exactly the composition they recorded (§2.3)
        self._reads = []
        parent = self._parent
        slots = _live_slots(parent) if parent is not None else ()
        gens = parent._gens if parent is not None else {}
        self._head = (slots, tuple(gens.get(slot, 0) for slot in slots))
        # the live list is the adopting parent's, so its stamps come along
        self._gens = dict(gens)
        self._memo = (self._session._mint_epoch, 1, composed, True)
        self._recorded = _union(("nominal",), labels_of(composed))
        self._origin = self
        # §2.1/§2.3: the adoption HANDS OUT the composed member, exactly as a read does, so a family
        # whose nominal is that node decides the same way whether or not anything read here first —
        # which is what makes `graphed.weight()` and `graphed.explain()` observations, not inputs.
        self._record_read(composed)


def _mask_key(mask: Array | Varied) -> tuple[tuple[str, int], ...]:
    """A derivation's identity: its per-label node ids. A REBUILT mask interns to the same ids,
    which is the binding condition for `ctx[rebuilt] is ctx[mask]`."""
    if isinstance(mask, Varied):
        return tuple((label, member.node_id) for label, member in mask._members.items())
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
    if ctx._is_data:
        raise GraphedError(
            f"variation {name!r} cannot be registered on a data context: data fills nominal-only, "
            "and accepting a registration whose labels the fill then drops would be a silent drop"
        )
    if is_weight:
        return _vary_weight(
            ctx, name, nominal, variations, collections, points, composes_as_union, max_universes, tags
        )
    return _vary_shift(
        ctx, name, nominal, variations, collections, points, composes_as_union, max_universes, tags
    )


def _child_of(ctx: EventContext) -> EventContext:
    """A `vary` link: the row space is unchanged, only registrations differ (§6.1d kind (2)).

    So the ambient state is COPIED, never composed: seeding the child with the composed parent
    would make every registration a two-element fold and reproduce the quadratic exactly.
    """
    child = EventContext(
        ctx._session,
        ctx._record,
        is_data=ctx._is_data,
        parent=ctx,
        link=("vary", None),
        collections=ctx._collections,
    )
    child._factors = list(ctx._factors)
    child._slots = list(ctx._slots)
    child._adopted = ctx._adopted
    child._head = ctx._head
    child._overlays = ctx._overlays
    # the same list object, not a copy: one row space, so a read at either end names the same
    # values, and a handle read from the parent after the child was built still decides here
    child._reads = ctx._reads
    child._gens = dict(ctx._gens)
    child._riders = dict(ctx._riders)
    child._recorded = ctx._recorded
    child._memo = ctx._memo
    child._origin = ctx._origin
    child._record = child._stamp(child._record)
    return child
