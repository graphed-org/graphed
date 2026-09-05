"""Fixtures for the m52 corpus tree — the arc's end-to-end NUMERIC acceptance (design §4.4, §6-C6).

Two halves that must not be confused:

* the **eager reference** lives in `graphed_corpus` (`btag_sf_rel_uncertainty`,
  `ttbar_joint_reference`) and knows nothing about `graphed` — that independence is what makes the
  joint-vs-reference comparison non-circular;
* the **`points=` program** lives here, because `graphed_corpus` is framework-free.

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

#: §4.4's R1-c joint tags: b-tag heavy-flavour SF ⊗ JES, both directions on both axes
JOINT_TAGS = tuple(f"jes{d}_hf_{s}" for d in ("up", "dn") for s in ("up", "down"))

#: {tag: point} — the coordinate map §4.4's R1-c comprehension registers for those tags
JOINT_POINTS = {
    f"jes{d}_hf_{s}": {"jes": full, "btag": f"hf_{s}"}
    for d, full in (("up", "up"), ("dn", "down"))
    for s in ("up", "down")
}

#: every label the program mints -> its point (§4.10), the two one-at-a-time b-tag points and the
#: JES pair carrying the default point `{name: tag}` beside the four registered joint points
EXPECTED_POINTS = {
    "nominal": {},
    "jes_up": {"jes": "up"},
    "jes_down": {"jes": "down"},
    "btag_hf_up": {"btag": "hf_up"},
    "btag_hf_down": {"btag": "hf_down"},
    "btag_jesup_hf_up": {"btag": "hf_up", "jes": "up"},
    "btag_jesup_hf_down": {"btag": "hf_down", "jes": "up"},
    "btag_jesdn_hf_up": {"btag": "hf_up", "jes": "down"},
    "btag_jesdn_hf_down": {"btag": "hf_down", "jes": "down"},
}

#: the joint universe the acceptance is measured at, and the eager reference's coordinates for it
JOINT_LABEL = "btag_jesup_hf_up"
JOINT_COORD = {"jes": "jes_up", "btag": "btag_up"}

#: the one-at-a-time universe the joint would COLLAPSE onto if resolution took nominal for JES
BTAG_ONLY_LABEL = "btag_hf_up"


@dataclass(frozen=True)
class JointProgram:
    """The recorded R1-c program, plus the two SF containers its joint tags share."""

    session: Session
    observable: Any
    weight: Any
    sf_hf_up: Any
    sf_hf_down: Any


def joint_program() -> JointProgram:
    """§4.4's R1-c spelling over the corpus ttbar analysis.

    `points=` is m52-new, so this is reachable only from a test body (TEST_SANITY §5-1).
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
    rel = 0.01 + 0.05 * gak.where(sgood.pt < 100.0, sgood.pt / 100.0, 1.0)
    hf_up = gak.prod(sf * (1.0 + rel), axis=1)
    hf_down = gak.prod(sf * (1.0 - rel), axis=1)

    # the four joint tags take the SAME two expression objects as the one-at-a-time pair; the POINT
    # is what selects which inner JES universe each label reads (§4.4)
    variations: dict[str, Any] = {"hf_up": hf_up, "hf_down": hf_down}
    variations.update({t: (hf_up if t.endswith("_hf_up") else hf_down) for t in JOINT_TAGS})
    sel = graphed.vary(
        sel,
        "btag",
        gak.prod(sf, axis=1),
        is_weight=True,
        variations=variations,
        points=JOINT_POINTS,
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
