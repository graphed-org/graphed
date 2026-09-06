"""Fixtures for the m53 projection-resolution suite.

This tree runs in its own pytest process (`scripts/run-tests.sh` splits `tests/frozen/awkward` per
milestone directory), but the `m52_` prefix is still required: the pytest `pythonpath` publishes
cross-dir helpers, so a bare helper name is a global name.

The b-tag SF members are computed on the jes-varied jets, so a plain `graphed.vary` mints the jes x
btag joints automatically. The auto-fanout is reached only inside these functions, never at import,
so the suite COLLECTS against a tree with no m53 implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graphed_corpus import make_events

import graphed
import graphed.awkward as ga
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak

#: one synthetic dataset for the whole tree
EVENTS = make_events(n_events=60, seed=52)

JES_UP, JES_DOWN = 1.10, 0.90
JER_UP = 1.05
JET_PT_CUT = 25.0

#: the machine-minted joint label the projection tests drive, and the one-at-a-time label that must
#: NOT follow it (`f"{name}_{tag}__{fl}"`, §2)
JOINT_LABEL = "btag_hf_up__jes_up"
ONE_AT_A_TIME_LABEL = "btag_hf_up"


def events_context() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    return session, ga.gnano.events(from_awkward(session, "events", EVENTS))


def _shifted_jets(jets: Any, factor: float) -> Any:
    return gak.with_field(jets, jets.pt * factor, "pt")


def jes_shifted_context() -> tuple[Session, Any]:
    """`jes` as a lockstep shift of the Jet collection — the analysis-side JES nuisance."""
    session, ctx = events_context()
    jets = ctx.Jet
    shifted = graphed.vary(
        ctx,
        "jes",
        collections={
            "Jet": {"up": _shifted_jets(jets, JES_UP), "down": _shifted_jets(jets, JES_DOWN)}
        },
    )
    return session, shifted


def btag_scale_factors(jets: Any) -> tuple[Any, Any, Any]:
    """A pT-DEPENDENT per-event b-tag SF product and its two systematic siblings, evaluated on the
    jets of whatever universe `jets` carries — R1-a propagation. The pT dependence is what makes the
    joint universe differ from the factorized product."""
    relative = 0.01 + 0.05 * gak.where(jets.pt < 100.0, jets.pt / 100.0, 1.0)
    central = gak.prod(1.0 + 0.0 * relative, axis=1)
    return central, gak.prod(1.0 + relative, axis=1), gak.prod(1.0 - relative, axis=1)


@dataclass
class JointProgram:
    session: Session
    context: Any
    #: the ambient weight of the registered context — carries every joint label
    ambient: Any
    #: the factor container's members, keyed by label, each itself `Varied` over `jes`
    factor_members: dict[str, Any]
    #: a `jes`-only container: one axis, so a joint point restricts onto it
    jes_only: Any
    #: a `jes` + `jer` container, so a union has labels new to the second operand
    two_axis: Any


def joint_weight_program() -> JointProgram:
    """The m53 auto-fanout spelling: a plain weight `vary` over two jes-dependent b-tag SF members.
    Each joint the fanout mints binds the SAME expression object as its one-at-a-time sibling, and
    the POINT selects which inner universe each label reads."""
    session, shifted = jes_shifted_context()
    jets = shifted.Jet
    central, sf_hf_up, sf_hf_down = btag_scale_factors(jets)

    registered = graphed.vary(
        shifted,
        "btag",
        central,
        is_weight=True,
        variations={"hf_up": sf_hf_up, "hf_down": sf_hf_down},
    )

    # the joint labels bind the SAME SF source as their one-at-a-time sibling; two-level resolution
    # then reads the inner jes universe named by each joint's point
    sources = {"hf_up": sf_hf_up, "hf_down": sf_hf_down}
    factor_members = {"nominal": central, "btag_hf_up": sf_hf_up, "btag_hf_down": sf_hf_down}
    for side, source in sources.items():
        for jes in ("jes_up", "jes_down"):
            factor_members[f"btag_{side}__{jes}"] = source

    jes_only = gak.sum(jets.pt, axis=1)
    met = shifted.MET.pt
    two_axis = graphed.vary(jes_only * met, "jer", up=jes_only * met * JER_UP)
    return JointProgram(
        session=session,
        context=registered,
        ambient=graphed.weight(registered),
        factor_members=factor_members,
        jes_only=jes_only,
        two_axis=two_axis,
    )


def selection_program() -> tuple[Session, Any, Any, Any]:
    """A `jes`-varied event mask, the context it selects, and a parent-read weight carrying the
    joint label — the `reindex_to` / `_follow` row-space shape."""
    session, shifted = jes_shifted_context()
    jets = shifted.Jet
    mask = gak.num(jets[jets.pt > JET_PT_CUT]) >= 4
    selected = shifted[mask]

    central, sf_hf_up, _sf_hf_down = btag_scale_factors(jets)
    # `sf_hf_up` is jes-dependent, so the loose fanout mints `btag_hf_up__jes_up` alongside the
    # one-at-a-time `btag_hf_up`
    carrier = graphed.vary(central, "btag", variations={"hf_up": sf_hf_up * 1.5})
    return session, mask, selected, carrier
