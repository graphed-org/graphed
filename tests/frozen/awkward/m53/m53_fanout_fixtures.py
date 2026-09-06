"""Fixtures for the m53 dependency-driven fanout suite (awkward backend).

The b-tag SF members are computed on the jes-varied jets, so each member DEPENDS on the jes nuisance
and a plain `graphed.vary` mints the full jes(3) x btag(5) grid automatically — no `points=`, no
hand-named joint tags. The joint labels are machine-generated `f"{name}_{tag}__{fl}"`, e.g.
`btag_hf_up__jes_up`.

The auto-fanout is reached only inside the functions the tests call, never at import, so the tree
COLLECTS against a tree with no m53 implementation and fails at RUN time (TEST_SANITY).
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
EVENTS = make_events(n_events=60, seed=53)

JES_FACTOR = {"up": 1.05, "down": 0.95}

#: the four b-tag SF tags, each computed on the jes-varied jets → all four depend on `jes`
BTAG_TAGS = ("hf_up", "hf_down", "lf_up", "lf_down")

#: every label the auto-fanout mints -> its point: nominal, the leaked jes pair, four one-at-a-time
#: b-tag points and the eight machine-named joints = the full jes(3) x btag(5) grid
GRID_POINTS: dict[str, dict[str, str]] = {
    "nominal": {},
    "jes_up": {"jes": "up"},
    "jes_down": {"jes": "down"},
    **{f"btag_{tag}": {"btag": tag} for tag in BTAG_TAGS},
    **{
        f"btag_{tag}__jes_{d}": {"btag": tag, "jes": d}
        for tag in BTAG_TAGS
        for d in ("up", "down")
    },
}

#: the eight joint labels, and the pre-m53 union (the seven the collapse produced)
JOINT_LABELS = tuple(
    f"btag_{tag}__jes_{d}" for tag in BTAG_TAGS for d in ("up", "down")
)
UNION_LABELS = ("nominal", "jes_up", "jes_down", *(f"btag_{tag}" for tag in BTAG_TAGS))

#: {joint label: (btag source key, the inner jes universe it must read)}
JOINT_SOURCES: dict[str, tuple[str, str]] = {
    f"btag_{tag}__jes_{d}": (tag, f"jes_{d}") for tag in BTAG_TAGS for d in ("up", "down")
}


@dataclass
class FanoutProgram:
    session: Session
    context: Any
    weight: Any
    #: the four SF sources, each itself `Varied` over `jes`
    members: dict[str, Any]
    #: the observable (HT), also jes-`Varied`, for per-universe materialization
    observable: Any


def _jes_context() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    ctx = ga.gnano.events(from_awkward(session, "events", EVENTS))
    jets = ctx.Jet
    shifted = graphed.vary(
        ctx,
        "jes",
        Jet={tag: gak.with_field(jets, jets.pt * f, "pt") for tag, f in JES_FACTOR.items()},
    )
    return session, shifted


def _btag_members(jets: Any) -> tuple[Any, dict[str, Any]]:
    """A pT-dependent central SF and four systematic siblings on `jets`. Heavy- and light-flavour
    use distinct pT dependences, so no two universes coincide."""
    rel_hf = 0.01 + 0.05 * gak.where(jets.pt < 100.0, jets.pt / 100.0, 1.0)
    rel_lf = 0.02 + 0.04 * gak.where(jets.pt < 60.0, jets.pt / 60.0, 1.0)
    sf = 0.95 + 0.1 * jets.btag
    central = gak.prod(sf, axis=1)
    members = {
        "hf_up": gak.prod(sf * (1.0 + rel_hf), axis=1),
        "hf_down": gak.prod(sf * (1.0 - rel_hf), axis=1),
        "lf_up": gak.prod(sf * (1.0 + rel_lf), axis=1),
        "lf_down": gak.prod(sf * (1.0 - rel_lf), axis=1),
    }
    return central, members


def fanout_weight(**vary_kwargs: Any) -> FanoutProgram:
    """A dependent b-tag weight family over jes-varied jets. Extra keywords pass straight through to
    `graphed.vary` — `composes_as_union=`, `points=`, `max_universes=` — so one builder drives every
    escalation rung."""
    session, shifted = _jes_context()
    jets = shifted.Jet
    good = jets[jets.pt > 25.0]
    sel = shifted[(gak.num(good) >= 4) & (gak.sum(good.btag > 0.7, axis=1) == 1)]
    sgood = sel.Jet[sel.Jet.pt > 25.0]
    observable = gak.sum(sgood.pt, axis=1)

    central, members = _btag_members(sgood)
    registered = graphed.vary(
        sel, "btag", central, is_weight=True, variations=members, **vary_kwargs
    )
    return FanoutProgram(
        session=session,
        context=registered,
        weight=graphed.weight(registered),
        members=members,
        observable=observable,
    )


def independent_program() -> tuple[Session, Any]:
    """jes x jer where the `jer` members are built from NOMINAL pt, so `jer` is INDEPENDENT of `jes`
    and the combination stays the union — the bound m53 preserves (no fanout)."""
    session = Session(AwkwardBackend())
    ctx = ga.gnano.events(from_awkward(session, "events", EVENTS))
    pt = ctx.Jet.pt
    jes = graphed.vary(pt, "jes", up=pt * 1.05, down=pt * 0.95)
    jer = graphed.vary(pt, "jer", hi=pt * 1.02, lo=pt * 0.98)  # from NOMINAL pt → independent
    return session, jes * jer
