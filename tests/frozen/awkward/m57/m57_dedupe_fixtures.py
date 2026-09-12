"""Fixtures for the m57 weight-dedupe suite (awkward backend).

The toy is §1's chain in miniature: a pure-weight `pu`, a b-tag SF over the jets whose central node
two families (`hf`, `lf`) re-use, a `trig` weight, and a relative-delta `mu` whose nominal is a
`graphed.weight(ctx)` handle. Every per-event value is a DYADIC rational with a small denominator —
the SF counts jets over a pT threshold (powers of two), the systematic tables and the deltas are
exact binary fractions — so every product of operations is exact whatever association the
composition picks, and an oracle folded here compares bit-for-bit against the ambient's own tree.

Every m57-new outcome (the join, the overlay, the refusals, `ambient_entries`, `explain`) is reached
only inside test bodies, so the tree COLLECTS against a pre-m57 tree and fails at RUN time. The
`m57_` prefix is load-bearing: pytest's prepend import mode publishes these helpers under a global
name.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import awkward as ak
from graphed_corpus import make_events

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.context import EventContext

#: one synthetic dataset for the whole tree; six events keep every materialised list readable
EVENTS = make_events(n_events=6, seed=57)

#: the SF counts jets above this pT, so a jes shift MOVES it (§4's "a weight the shift moves") while
#: every central value stays a power of two
PT_CUT = 20.0

#: the two-tag up/down shape every family in the tree has
TAGS = ("up", "down")

#: the systematic tables: each scales the SF's per-jet step, so the member is `(1 + table)**n_jets`
HF_TABLE = {"up": 1.5, "down": 0.5}
LF_TABLE = {"up": 0.75, "down": 0.25}
#: further tables, for legs that need families whose universes cannot coincide
MF_TABLE = {"up": 0.375, "down": 0.125}
NF_TABLE = {"up": 0.1875, "down": 0.0625}

#: the shifts' pT scale factors — dyadic, and far enough to move jets across `PT_CUT`
JES = {"up": 1.25, "down": 0.75}
JER = {"up": 1.5, "down": 0.5}

#: the relative-delta families' rescaling of the whole ambient
MU = {"up": 1.25, "down": 0.75}
NU = {"up": 1.125, "down": 0.875}
#: the pure-weight factors' rescaling of their own central
PU = {"up": 1.5, "down": 0.5}
TRIG = {"up": 2.0, "down": 0.25}

#: the mask cuts for the row-space legs; each keeps part of the six-event sample
MET_CUT = 10.0
HT_CUT = 90.0


# ---- the toy's per-event weights -------------------------------------------------------------
def m57_sf(jets: Any, table: float = 1.0) -> Any:
    """The b-tag SF over `jets`: one step per jet above `PT_CUT`, scaled by the systematic table.

    `table=1.0` is the central. A pT shift moves which jets pass the cut, so wherever `jets` is
    varied the central itself carries that nuisance's universes.
    """
    return gak.prod(1.0 + table * gak.where(jets.pt > PT_CUT, 1.0, 0.0), axis=1)


def m57_pu(jets: Any) -> Any:
    """A pure-weight pileup factor off the jet MULTIPLICITY, which no pT shift moves."""
    return 1.0 + 0.25 * gak.num(jets)


def m57_trig(met: Any) -> Any:
    """A second pure-weight factor, off the MET — the factor the tree registers after an overlay."""
    return gak.where(met.pt > MET_CUT, 1.5, 1.25)


def m57_scaled(value: Any, scale: Mapping[str, float]) -> dict[str, Any]:
    """`{tag: value * scale[tag]}` — the members of a family that rescales one node."""
    return {tag: value * factor for tag, factor in scale.items()}


def m57_table_members(jets: Any, table: Mapping[str, float]) -> dict[str, Any]:
    """The systematic members of a family whose central is `m57_sf(jets)`."""
    return {tag: m57_sf(jets, factor) for tag, factor in table.items()}


def m57_scale(record: Any, factor: float) -> Any:
    """A record whose `pt` is scaled — the toy's one shift shape, for jets and MET alike."""
    return gak.with_field(record, record.pt * factor, "pt")


# ---- the toy's contexts ----------------------------------------------------------------------
def m57_base() -> tuple[Session, Any]:
    """A fresh Session and a root context carrying `Jet` and `MET`, nothing registered."""
    session = Session(AwkwardBackend())
    root = from_awkward(session, "events", EVENTS)
    ctx = EventContext(session, root, collections={"Jet": root.Jet, "MET": root.MET})
    return session, ctx


def m57_shifted(ctx: Any, name: str, scale: Mapping[str, float]) -> Any:
    """Register `name` as a SHIFT of `Jet` and of `MET`, so the shift is visible on both."""
    jets, met = ctx["Jet"], ctx["MET"]
    return graphed.vary(
        ctx,
        name,
        collections={
            "Jet": {tag: m57_scale(jets, factor) for tag, factor in scale.items()},
            "MET": {tag: m57_scale(met, factor) for tag, factor in scale.items()},
        },
    )


def m57_weight(ctx: Any, name: str, central: Any, members: Mapping[str, Any], **kw: Any) -> Any:
    """Register the weight family `name` with `central` and its declared `members`."""
    return graphed.vary(ctx, name, central, is_weight=True, points=dict(members), **kw)


def m57_delta(ctx: Any, name: str, handle: Any, scale: Mapping[str, float] = MU) -> Any:
    """The relative-delta idiom: an ambient read as the nominal, the whole ambient rescaled."""
    return m57_weight(ctx, name, handle, m57_scaled(handle, scale))


# ---- the oracle ------------------------------------------------------------------------------
def m57_values(session: Session, value: Any) -> list[Any]:
    """`value` materialised as a Python list — the comparison every value leg makes."""
    return ak.to_list(session.materialize(value))


@dataclass(frozen=True)
class Op:
    """One expected operation of the ambient, built by `m57_factor` / `m57_overlay`."""

    #: a factor's nominal member; `None` for an overlay, whose nominal IS the running product
    nominal: Any
    #: `{label: the member this operation contributes there}`
    members: Mapping[str, Any]
    overlay: bool = False


def m57_by_label(name: str, members: Mapping[str, Any]) -> dict[str, Any]:
    """A family's `{tag: member}` re-keyed to the `{label: member}` form the oracle reads."""
    return {f"{name}_{tag}": member for tag, member in members.items()}


def m57_factor(name: str, nominal: Any, members: Mapping[str, Any], **labels: Any) -> Op:
    """A factor: it multiplies its member at each label it declares, its nominal everywhere else.

    `labels` carries the entries whose key is a label rather than a tag — a machine-minted cross
    member, a placed universe, or a second family's members on a joined container.
    """
    return Op(nominal, {**m57_by_label(name, members), **labels})


def m57_overlay(name: str, members: Mapping[str, Any], **labels: Any) -> Op:
    """An overlay: it REPLACES the running product at the labels its family covers and contributes
    nothing anywhere else, its nominal being that product already."""
    return Op(None, {**m57_by_label(name, members), **labels}, overlay=True)


def m57_reindexed(op: Op, ctx: Any) -> Op:
    """`op` with its nominal and every member re-indexed into `ctx`'s row space — what a row-space
    change does to an entry the child's list inherits."""
    return Op(
        None if op.nominal is None else graphed.reindex_to(op.nominal, ctx),
        {label: graphed.reindex_to(member, ctx) for label, member in op.members.items()},
        op.overlay,
    )


def m57_joined(op: Op, name: str, members: Mapping[str, Any], **labels: Any) -> Op:
    """`op` with a second family's members on its container — what a join adds to a factor."""
    return Op(op.nominal, {**op.members, **m57_by_label(name, members), **labels}, op.overlay)


def m57_at(value: Any, label: str) -> Any:
    """`value` as an operation contributes it at `label`: its own member there, then that member's
    own — a member built over shifted objects carries its dependence one level down."""
    return graphed.member_of(graphed.member_of(value, label), label)


def m57_ambient(ops: Sequence[Op], label: str) -> Any:
    """The one-SF oracle: the ambient at `label`, folded from the operands the test itself holds.

    A left fold, which is exact for this toy's dyadic values whatever association the composition
    chooses.
    """
    product: Any = None
    for op in ops:
        if op.overlay:
            if label in op.members:
                product = m57_at(op.members[label], label)
            continue
        member = m57_at(op.members.get(label, op.nominal), label)
        product = member if product is None else product * member
    return product


def m57_oracle_values(session: Session, ops: Sequence[Op], labels: Sequence[str]) -> dict[str, list[Any]]:
    """`{label: the one-SF oracle's value there}` over `labels`."""
    return {label: m57_values(session, m57_ambient(ops, label)) for label in labels}


def m57_ambient_values(session: Session, weight: Any) -> dict[str, list[Any]]:
    """`{label: the ambient's value there}` for every label the ambient carries."""
    return {label: m57_values(session, graphed.member_of(weight, label)) for label in graphed.labels(weight)}


def m57_node(weight: Any, label: str = "nominal") -> int:
    """The node id the ambient resolves to at `label` — the node legs' instrument."""
    return int(graphed.member_of(weight, label).node_id)


def m57_ambient_nodes(weight: Any) -> dict[str, int]:
    """`{label: the node id the ambient resolves to there}` over every label it carries."""
    return {label: m57_node(weight, label) for label in graphed.labels(weight)}


# ---- the shared programs ---------------------------------------------------------------------
@dataclass(frozen=True)
class Base:
    """The two-factor base most legs start from: `pu`, then `hf` over the SF central `sf`."""

    session: Session
    ctx: Any
    jets: Any
    met: Any
    sf: Any
    pu: Any
    pu_members: Mapping[str, Any]
    hf_members: Mapping[str, Any]

    @property
    def factors(self) -> list[Op]:
        """The two registered factors' expected operations, in registration order."""
        return [
            m57_factor("pu", self.pu, self.pu_members),
            m57_factor("hf", self.sf, self.hf_members),
        ]


def m57_two_factors() -> Base:
    """`pu` then `hf` (on the SF central) on a root context — two live factors, no ambient read."""
    session, ctx = m57_base()
    jets, met = ctx["Jet"], ctx["MET"]
    sf, pu = m57_sf(jets), m57_pu(jets)
    pu_members, hf_members = m57_scaled(pu, PU), m57_table_members(jets, HF_TABLE)
    after_pu = m57_weight(ctx, "pu", pu, pu_members)
    return Base(
        session=session,
        ctx=m57_weight(after_pu, "hf", sf, hf_members),
        jets=jets,
        met=met,
        sf=sf,
        pu=pu,
        pu_members=pu_members,
        hf_members=hf_members,
    )


@dataclass(frozen=True)
class Tour:
    """§1's chain: `pu`, `hf` on the SF central, the overlay `mu` read over both, `lf` on the same
    central behind the overlay, and `lf`'s diagonal placement at `{hf: up, mu: up}`."""

    session: Session
    ctx: Any
    jets: Any
    sf: Any
    pu: Any
    handle: Any
    pu_members: Mapping[str, Any]
    hf_members: Mapping[str, Any]
    lf_members: Mapping[str, Any]
    mu_members: Mapping[str, Any]

    @property
    def ops(self) -> list[Op]:
        """The expected operation list: `pu`, the container `hf` and `lf` share, the overlay."""
        return [
            m57_factor("pu", self.pu, self.pu_members),
            m57_factor(
                "hf",
                self.sf,
                self.hf_members,
                **m57_by_label("lf", self.lf_members),
            ),
            m57_overlay("mu", self.mu_members),
        ]


def m57_tour(*, place: bool = True, offset: int = 0) -> Tour:
    """§1's chain: `pu`, `hf` on `sf`, `mu` read over both, then `lf` on `sf` behind the overlay.

    `lf` also PLACES its `up` universe at the diagonal point carrying `hf`'s and `mu`'s `up`
    coordinates — the tour's `scale_upup`. `place=False` drops the placement, so a leg that must see
    `lf_up` on its own axis can have it. `offset` mints that many unrelated nodes before the program,
    so two Sessions' node ids for the same program cannot coincide.
    """
    session, ctx = m57_base()
    jets = ctx["Jet"]
    for step in range(offset):
        gak.num(jets) * float(step + 3)
    sf, pu = m57_sf(jets), m57_pu(jets)
    pu_members = m57_scaled(pu, PU)
    hf_members, lf_members = m57_table_members(jets, HF_TABLE), m57_table_members(jets, LF_TABLE)
    after = m57_weight(ctx, "pu", pu, pu_members)
    after = m57_weight(after, "hf", sf, hf_members)
    handle = graphed.weight(after)
    mu_members = m57_scaled(handle, MU)
    after = m57_delta(after, "mu", handle)
    lf_points: Any = [*lf_members.items(), {"lf": "up", "hf": "up", "mu": "up"}] if place else lf_members
    after = graphed.vary(after, "lf", sf, is_weight=True, points=lf_points)
    return Tour(
        session=session,
        ctx=after,
        jets=jets,
        sf=sf,
        pu=pu,
        handle=handle,
        pu_members=pu_members,
        hf_members=hf_members,
        lf_members=lf_members,
        mu_members=mu_members,
    )


# ---- instruments -----------------------------------------------------------------------------
def m57_node_count(session: Session) -> int:
    """How many nodes the Session's store holds — the mint instrument the refusal legs read."""
    return int(session._store.node_count())


class m57_OpSpy:
    """Counts CALLS to `Session.record_op` — the cost-class leg's instrument.

    Minted-node counts are blind to a walk that re-derives an interned product, so the work a
    composition does is counted at the record call instead. Installed on the instance under test,
    so the patch dies with it.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._inner = session.record_op
        self.calls = 0

    def __enter__(self) -> m57_OpSpy:
        def counting(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            return self._inner(*args, **kwargs)

        self._session.record_op = counting  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: object) -> None:
        del self._session.record_op  # type: ignore[attr-defined]


def m57_explain(ctx: Any) -> Any:
    """`graphed.systematics.explain(ctx)`, imported HERE so the tree still collects on a pre-m57
    tree: the package does not exist there and the call raises at run time."""
    from graphed.systematics import explain  # noqa: PLC0415

    return explain(ctx)


def m57_entries(ctx: Any) -> Any:
    """`graphed.systematics.ambient_entries(ctx)` — the `(slot, rider, entry)` records in the order
    the composition applies them. Imported inside the call, as `m57_explain` is."""
    from graphed.systematics import ambient_entries  # noqa: PLC0415

    return ambient_entries(ctx)


def m57_ir(session: Session, weight: Any) -> bytes:
    """The serialized IR of every universe the ambient carries — the determinism legs' bytes."""
    outputs = [graphed.member_of(weight, label) for label in graphed.labels(weight)]
    return bytes(session.serialized_ir(*outputs))
