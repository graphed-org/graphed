"""Fixtures for the m56 both-kind fan-out suite (awkward backend).

The toy is the tour capstone in miniature: `jes` MOVES the jets (a shift of the `Jet`/`MET`
collections) and SWAPS the b-tag SF table by name identity (a weight factor), so
`graphed.variations(ctx)["jes"]` reports `Kind.WEIGHT | Kind.SHIFT`. A second family registered over
those jets reaches `jes_up` through the shifted jets, never through the ambient weight, so its
`jes` coordinate is a dependency and not weight composition.

Every m56-new outcome — the `jes` joints, the placement that names one, the guard's wider bound — is
reached only inside the functions the tests call, never at import, so the tree COLLECTS against a
pre-m56 tree and fails at RUN time (TEST_SANITY). The `m56_` prefix is load-bearing: the pytest
`pythonpath` publishes cross-dir helpers under a global name.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from graphed_corpus import make_events

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.context import EventContext

#: one synthetic dataset for the whole tree
EVENTS = make_events(n_events=40, seed=56)

#: every family in the tree is a two-tag up/down family, so a fan-out mints FOUR joints
TAGS = ("up", "down")

JES = {"up": 1.05, "down": 0.95}
JER = {"up": 1.02, "down": 0.98}
#: the SF table each `jes` WEIGHT member swaps to, and the second family's own table. Their
#: NOMINALS differ, so the two families' centrals are two nodes and the two stay two factors: a
#: shared nominal makes one central name the other's factor and the ambient composes one entry.
JES_SF = {"nominal": 1.0, "up": 1.2, "down": 0.8}
HF_SF = {"nominal": 1.1, "up": 1.3, "down": 0.7}
#: the probe families' table, distinct from both so no two universes coincide
PROBE_SF = {"nominal": 1.0, "up": 1.4, "down": 0.6}

#: how much of the jets' pT sum the MET recoils against, so a jet shift MOVES the MET
MET_COUPLING = 0.1
#: cuts that keep part of the sample for the mask-derived legs
SEED_CUT = 1.02
HT_CUT = 60.0
MET_CUT = 20.0

#: the capstone's seven one-at-a-time labels, `nominal` first
UNION_LABELS = ("nominal", "jes_up", "jes_down", "jer_up", "jer_down", "hf_up", "hf_down")

#: the guard's pre-m56 bound on the capstone's `hf` registration is jer(3) x hf(3)
PRE_M56_BOUND = 9
#: the design's bound is the product of family sizes, nominal included: jer(3) x jes(3) x hf(3)
DESIGN_BOUND = 27


def m56_scale(record: Any, factor: float) -> Any:
    """A record whose `pt` is scaled — the toy's one shift, for jets and MET alike."""
    return gak.with_field(record, record.pt * factor, "pt")


def m56_met_of(jets: Any, raw: Any) -> Any:
    """The toy Type-1 MET: the raw MET recoiling against the jets' pT sum, phi untouched."""
    return gak.with_field(raw, raw.pt - MET_COUPLING * gak.sum(jets.pt, axis=1), "pt")


def m56_sf(jets: Any, table: float) -> Any:
    """A per-event b-tag scale factor off `jets`; `table` selects the systematic table."""
    return gak.prod(0.95 + 0.1 * jets.btag * table, axis=1)


def m56_joints(name: str, foreign: str) -> set[str]:
    """The four joint labels `name`(up/down) x `foreign`(up/down)."""
    return {f"{name}_{tag}__{foreign}_{other}" for tag in TAGS for other in TAGS}


def m56_minted(weight: Any, name: str) -> set[str]:
    """The joint labels `name`'s registration minted — the family's own name comes first."""
    return {label for label in graphed.labels(weight) if label.startswith(f"{name}_") and "__" in label}


def m56_base(*, seed: bool = False) -> tuple[Session, Any, Any]:
    """A fresh Session and a context carrying `Jet` and `MET`, with an optional seed event weight
    (`EventContext(..., weight=...)`) — a factor whose member is its nominal at every label."""
    session = Session(AwkwardBackend())
    root = from_awkward(session, "events", EVENTS)
    genweight = 1.0 + 0.001 * root.MET.pt if seed else None
    ctx = EventContext(session, root, collections={"Jet": root.Jet, "MET": root.MET}, weight=genweight)
    return session, ctx, genweight


def m56_weight_family(ctx: Any, name: str, member: Any, **vary_kwargs: Any) -> Any:
    """Register `name` as a weight family whose three members scale `member` by the `PROBE_SF`
    table. Extra keywords pass straight through to `graphed.vary`."""
    return graphed.vary(
        ctx,
        name,
        member * PROBE_SF["nominal"],
        is_weight=True,
        points={tag: member * PROBE_SF[tag] for tag in TAGS},
        **vary_kwargs,
    )


def m56_both_kind(ctx: Any) -> tuple[Any, Any]:
    """`jes` registered as a SHIFT of the jets and as a WEIGHT by name identity; returns the
    context and the varied jets the weight factor's members are computed on."""
    jets = ctx.Jet
    shifted = graphed.vary(ctx, "jes", collections={"Jet": {tag: m56_scale(jets, JES[tag]) for tag in TAGS}})
    sjets = shifted["Jet"]
    registered = graphed.vary(
        shifted,
        "jes",
        m56_sf(sjets, JES_SF["nominal"]),
        is_weight=True,
        points={tag: m56_sf(sjets, JES_SF[tag]) for tag in TAGS},
    )
    return registered, sjets


def _m56_shift(ctx: Any, name: str, members: Mapping[str, Any], raw: Any) -> Any:
    """Register `name` as a shift of `Jet` and of the MET propagated from those jets."""
    return graphed.vary(
        ctx,
        name,
        collections={
            "Jet": dict(members),
            "MET": {tag: m56_met_of(member, raw) for tag, member in members.items()},
        },
    )


@dataclass
class Capstone:
    session: Session
    ctx: Any
    weight: Any
    #: {label: the jet record that label's universe reads} — the oracle's operands
    jets: dict[str, Any]


def m56_capstone(
    *,
    weight_first: bool = True,
    placements: Iterable[Mapping[str, Any]] | None = None,
    **vary_kwargs: Any,
) -> Capstone:
    """The headline program: `jes` shifts `Jet`/`MET` AND swaps the SF table by name identity;
    `jer` shifts them too, its members built on the NOMINAL jets so it stays shift-only; `hf` is a
    weight family over the jets both families vary.

    `weight_first` picks the registration order of the two weight families (§2 item 2);
    `placements` and the extra keywords land on the `hf` registration.
    """
    session, ctx, _seed = m56_base()
    jets, raw = ctx.Jet, ctx.MET
    jes_members = {tag: m56_scale(jets, JES[tag]) for tag in TAGS}
    jer_members = {tag: m56_scale(jets, JER[tag]) for tag in TAGS}
    after_jes = _m56_shift(ctx, "jes", jes_members, raw)
    jes_jets = after_jes["Jet"]  # the jets as `jes` alone varies them: the weight factor's operand
    context = _m56_shift(after_jes, "jer", jer_members, raw)
    sjets = context["Jet"]
    varied: dict[str, Any] = {
        "nominal": jets,
        **{f"jes_{tag}": member for tag, member in jes_members.items()},
        **{f"jer_{tag}": member for tag, member in jer_members.items()},
    }

    def jes_weight(target: Any) -> Any:
        return graphed.vary(
            target,
            "jes",
            m56_sf(jes_jets, JES_SF["nominal"]),
            is_weight=True,
            points={tag: m56_sf(jes_jets, JES_SF[tag]) for tag in TAGS},
        )

    def hf_weight(target: Any) -> Any:
        declares = {tag: m56_sf(sjets, HF_SF[tag]) for tag in TAGS}
        points: Any = declares if placements is None else [*declares.items(), *placements]
        return graphed.vary(
            target, "hf", m56_sf(sjets, HF_SF["nominal"]), is_weight=True, points=points, **vary_kwargs
        )

    steps = (jes_weight, hf_weight) if weight_first else (hf_weight, jes_weight)
    for step in steps:
        context = step(context)
    return Capstone(session, context, graphed.weight(context), varied)


def m56_incidental(leg: str) -> tuple[Session, Any]:
    """§2 item 4's INCLUSION legs: the probe family's member also reads a factor whose member at the
    label is its NOMINAL node — the context's seed weight (`"seed"`), or an unrelated weight
    family's central member computed from the unshifted jets (`"central"`) — beside a bare
    `SF(varied jets)` sibling registered in the same program."""
    session, ctx, seed = m56_base(seed=True)
    context, sjets = m56_both_kind(ctx)
    context = m56_weight_family(context, "bare", m56_sf(sjets, 1.0))
    if leg == "seed":
        return session, m56_weight_family(context, "probe", seed * m56_sf(sjets, 1.1))
    pu_central = 1.0 + 0.002 * gak.num(ctx.Jet)  # the UNSHIFTED jets
    context = m56_weight_family(context, "pu", pu_central)
    return session, m56_weight_family(context, "probe", m56_sf(sjets, 1.1) * pu_central)


def m56_cut_child(cut: str) -> tuple[Session, Any]:
    """A family registered on a mask-derived child: the mask reads the SEED weight (`"seed"`) or the
    varied jets (`"kinematic"`, the control). Both masks read a factor whose member at every label
    is its nominal, or no factor at all."""
    session, ctx, seed = m56_base(seed=True)
    context, sjets = m56_both_kind(ctx)
    mask = seed > SEED_CUT if cut == "seed" else gak.sum(sjets.pt, axis=1) > HT_CUT
    child = context[mask]
    return session, m56_weight_family(child, "cut", m56_sf(child["Jet"], 1.0))


def m56_two_both_kind() -> tuple[Session, Any, Any]:
    """`jes` and `jer` BOTH registered as a shift and a weight, so every coordinate the ambient
    carries is composition. `jer`'s weight members read the jes-varied jets, so the ambient itself
    carries joint labels."""
    session, ctx, _seed = m56_base()
    jets = ctx.Jet
    after_jes = graphed.vary(
        ctx, "jes", collections={"Jet": {tag: m56_scale(jets, JES[tag]) for tag in TAGS}}
    )
    jes_jets = after_jes["Jet"]
    context = graphed.vary(
        after_jes, "jer", collections={"Jet": {tag: m56_scale(jets, JER[tag]) for tag in TAGS}}
    )
    sjets = context["Jet"]
    context = graphed.vary(
        context,
        "jes",
        m56_sf(jes_jets, JES_SF["nominal"]),
        is_weight=True,
        points={tag: m56_sf(jes_jets, JES_SF[tag]) for tag in TAGS},
    )
    context = graphed.vary(
        context,
        "jer",
        m56_sf(sjets, HF_SF["nominal"]),
        is_weight=True,
        points={tag: m56_sf(sjets, HF_SF[tag]) for tag in TAGS},
    )
    return session, context, sjets


def m56_pure_weight() -> tuple[Session, Any, Any]:
    """A PURE-weight `pu` whose members are computed from the unshifted jets, beside a PLAIN `jes`
    shift: at a `jes` label no factor's member differs from its nominal."""
    session, ctx, _seed = m56_base()
    jets = ctx.Jet
    context = graphed.vary(ctx, "jes", collections={"Jet": {tag: m56_scale(jets, JES[tag]) for tag in TAGS}})
    sjets = context["Jet"]
    return session, m56_weight_family(context, "pu", 1.0 + 0.002 * gak.num(jets)), sjets


def m56_masked_child() -> tuple[Session, Any, Any]:
    """A both-kind `jes`, the PARENT's ambient weight captured before the row-space change, and the
    mask-derived child the probe family registers on."""
    session, ctx, _seed = m56_base()
    context, _sjets = m56_both_kind(ctx)
    parent_ambient = graphed.weight(context)
    return session, context[ctx.MET.pt > MET_CUT], parent_ambient


def m56_spectator() -> tuple[Session, Any, Any, Any]:
    """A both-kind `jes` and a LOOSE `inner` family the ambient carrier does not carry — a spectator
    coordinate on any member that reads it."""
    session, ctx, _seed = m56_base()
    context, sjets = m56_both_kind(ctx)
    loose = m56_sf(graphed.nominal(sjets), 1.0)
    inner = graphed.vary(loose, "inner", hi=loose * 1.5, lo=loose * 0.5)
    return session, context, sjets, inner
