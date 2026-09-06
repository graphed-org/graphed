"""Fixtures for the m53 corpus tree — the arc's end-to-end NUMERIC acceptance (design §4.4, §6-C6).

Two halves that must not be confused:

* the **eager reference** lives in `graphed_corpus` (`btag_sf_rel_uncertainty`,
  `ttbar_joint_reference`) and knows nothing about `graphed` — that independence is what makes the
  joint-vs-reference comparison non-circular;
* the **auto-fanout program** lives here, because `graphed_corpus` is framework-free.

Under m53 the b-tag SF is computed over the jes-varied jets, so its members DEPEND on the jes
nuisance and a plain `graphed.vary` mints the full jes x btag grid automatically — no `points=`, no
hand-named joint tags. The joint labels are machine-generated `f"{name}_{tag}__{fl}"`, e.g.
`btag_hf_up__jes_up`.

The histogram producer is eager `hist.Hist`, never `graphed_histogram`: `graphed`'s CI does not
install it, so a tree reaching for it would `skip` and this arc's only physics check would contribute
no frozen coverage.

No `conftest.py`: three `corpus/m05` tests do `from conftest import ...` and the whole
`tests/frozen/corpus` tree collects in ONE process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from graphed_corpus import make_events
from graphed_corpus.analyses import systematics
from hist import Hist

import graphed
import graphed.awkward as ga
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak

#: the canonical corpus dataset, the one `corpus/m05`'s stored references are built on
EVENTS = make_events()

#: the region whose observable is HT; its selection and binning are `ttbar_region`'s verbatim
REGION = "4j1b"

#: `graphed_corpus.analyses.systematics`' own JES shift, as a jet-pT scale
JES_FACTOR = {"up": 1.05, "down": 0.95}

#: the four b-tag SF tags the family carries: heavy- and light-flavour, both directions. Each is
#: computed over the jes-varied jets, so all four members depend on the jes nuisance.
BTAG_TAGS = ("hf_up", "hf_down", "lf_up", "lf_down")

#: every label the auto-fanout mints -> its point (§4.10): nominal, the JES pair leaked from the
#: foreign family, the four one-at-a-time b-tag points, and the eight machine-named joint points =
#: the full jes(3) x btag(5) grid
EXPECTED_POINTS = {
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

#: the joint universe the acceptance is measured at, and the eager reference's coordinates for it
JOINT_LABEL = "btag_hf_up__jes_up"
JOINT_COORD = {"jes": "jes_up", "btag": "btag_up"}

#: the one-at-a-time universe the joint would COLLAPSE onto if resolution took nominal for JES
BTAG_ONLY_LABEL = "btag_hf_up"

#: {machine joint label: (btag source key, jes universe)} — the heavy-flavour cross terms whose
#: member-sharing `test_joint_member_sharing` pins; each reads its SF source's inner jes universe
HF_JOINTS = {
    f"btag_hf_{s}__jes_{d}": (f"hf_{s}", f"jes_{d}")
    for s in ("up", "down")
    for d in ("up", "down")
}


@dataclass(frozen=True)
class JointProgram:
    """The recorded R1-c program, plus the two SF containers its joint tags share."""

    session: Session
    observable: Any
    weight: Any
    sf_hf_up: Any
    sf_hf_down: Any


def joint_program() -> JointProgram:
    """The m53 auto-fanout spelling over the corpus ttbar analysis.

    A plain `graphed.vary` over the four b-tag SF members: each is computed on the jes-varied jets,
    so the family depends on `jes` and the full jes x btag grid is minted automatically — no
    `points=`, no hand-named joint tags. The auto-fanout is m53-new, so this is reachable only from
    a test body (TEST_SANITY §5-1).
    """
    session = Session(AwkwardBackend())
    ctx = ga.gnano.events(from_awkward(session, "events", EVENTS))
    jets = ctx.Jet
    ctx = graphed.vary(
        ctx,
        "jes",
        Jet={tag: gak.with_field(jets, jets.pt * f, "pt") for tag, f in JES_FACTOR.items()},
    )
    good = ctx.Jet[ctx.Jet.pt > 25.0]
    sel = ctx[(gak.num(good) >= 4) & (gak.sum(good.btag > 0.7, axis=1) == 1)]

    sgood = sel.Jet[sel.Jet.pt > 25.0]
    observable = gak.sum(sgood.pt, axis=1)
    sf = 0.95 + 0.1 * sgood.btag
    # `gak` has no element-wise minimum; `where` is the reference's arithmetic, `pt == 100` included
    rel_hf = 0.01 + 0.05 * gak.where(sgood.pt < 100.0, sgood.pt / 100.0, 1.0)
    # a light-flavour component with a distinct pT dependence, so its universes never coincide with
    # the heavy-flavour ones
    rel_lf = 0.02 + 0.04 * gak.where(sgood.pt < 60.0, sgood.pt / 60.0, 1.0)
    hf_up = gak.prod(sf * (1.0 + rel_hf), axis=1)
    hf_down = gak.prod(sf * (1.0 - rel_hf), axis=1)
    lf_up = gak.prod(sf * (1.0 + rel_lf), axis=1)
    lf_down = gak.prod(sf * (1.0 - rel_lf), axis=1)

    sel = graphed.vary(
        sel,
        "btag",
        gak.prod(sf, axis=1),
        is_weight=True,
        points={"hf_up": hf_up, "hf_down": hf_down, "lf_up": lf_up, "lf_down": lf_down},
    )
    return JointProgram(session, observable, graphed.weight(sel), hf_up, hf_down)


def universe_hist(program: JointProgram, label: str) -> Hist:
    """One graphed universe, materialized and filled into the region's own eager histogram."""
    values = program.session.materialize(graphed.member_of(program.observable, label))
    weights = program.session.materialize(graphed.member_of(program.weight, label))
    h = Hist.new.Reg(40, 0, 800, name="ht").Double()
    h.fill(np.asarray(values), weight=np.asarray(weights))
    return h


def reference_hist(*, jes: str, btag: str, **kwargs: bool) -> Hist:
    """The eager joint reference at one `(jes, btag)` coordinate pair.

    Reached as a module ATTRIBUTE, so this file still imports against a corpus without it.
    """
    return systematics.ttbar_joint_reference(EVENTS, region=REGION, jes=jes, btag=btag, **kwargs)


def integral(h: Hist) -> float:
    """The acceptance is on the INTEGRAL: per bin, JES migrates events and even the flat,
    frozen-selection leg is nonzero, which would void the machine-zero control."""
    return float(h.sum(flow=True))


def factorized(*, jes_only: float, btag_only: float, nominal: float) -> float:
    """The multiplicative composition of the two one-at-a-time universes."""
    return jes_only * btag_only / nominal


def reldiff(joint: float, product: float) -> float:
    return abs(joint - product) / abs(joint)


def reference_reldiff(**kwargs: bool) -> float:
    """joint-vs-factorized, read off the eager reference alone — no graphed resolution in it."""

    def integ(jes: str, btag: str) -> float:
        return integral(reference_hist(jes=jes, btag=btag, **kwargs))

    return reldiff(
        integ(JOINT_COORD["jes"], JOINT_COORD["btag"]),
        factorized(
            jes_only=integ(JOINT_COORD["jes"], "nominal"),
            btag_only=integ("nominal", JOINT_COORD["btag"]),
            nominal=integ("nominal", "nominal"),
        ),
    )
