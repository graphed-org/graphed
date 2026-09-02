"""§2.5's shift-after-weight report describes ONE COMPILED PROGRAM, not the Session that built it.

`Session._shift_after_weight` accumulates for the Session's lifetime, so without a per-compile
filter two correct programs inherit a third one's violation. Both directions here: the violating
program still reports its pair (the instrument is live), while a sibling program that never applies
the offending weight reports `()` — whether it is compiled from the same session afterwards, or
built on a branch the weight was never registered on at all.
"""

from __future__ import annotations

from typing import Any

from graphed_corpus import make_events

import graphed
import graphed.awkward as ga
from graphed import Array, Session, compile_ir
from graphed.awkward import AwkwardBackend, from_awkward, gak

EVENTS = make_events(n_events=50, seed=4949)


def _context() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    return session, ga.gnano.events(from_awkward(session, "events", EVENTS))


def _btag_weight(source: Any, scale: float = 1.0) -> Array:
    """A per-event weight factor whose cone reaches the Jet collection."""
    return gak.prod(1.0 + source.Jet.btag * scale, axis=1)


def _weighted(ctx: Any) -> Any:
    return graphed.vary(
        ctx, "btag", _btag_weight(ctx), is_weight=True, up=_btag_weight(ctx, 1.1), down=_btag_weight(ctx, 0.9)
    )


def _jes(source: Any) -> dict[str, dict[str, Array]]:
    jets = source.Jet
    return {
        "Jet": {
            "up": gak.with_field(jets, jets.pt * 1.05, "pt"),
            "down": gak.with_field(jets, jets.pt * 0.95, "pt"),
        }
    }


def _weight_universes(ctx: Any) -> list[Array]:
    ambient = graphed.weight(ctx)
    return [graphed.universe(ambient, label) for label in graphed.labels(ambient)]


def _jet_pt_universes(ctx: Any) -> list[Array]:
    """One marked output per label of the (varied) Jet collection — no weight is applied."""
    jets = ctx.Jet
    return [gak.sum(graphed.universe(jets, label).pt, axis=1) for label in graphed.labels(jets)]


def test_an_unrelated_program_on_the_same_session_does_not_inherit_the_violation() -> None:
    session, ctx = _context()
    weighted = _weighted(ctx)
    violating = graphed.vary(weighted, "jes", collections=_jes(weighted))

    assert compile_ir(session, *_weight_universes(violating)).shift_after_weight == (("btag", "Jet"),)
    # the sibling program applies no weight, so the pair is not a fact about it
    assert compile_ir(session, *_jet_pt_universes(violating)).shift_after_weight == ()


def test_a_shift_on_a_branch_the_weight_was_never_registered_on_reports_nothing() -> None:
    session, ctx = _context()
    _weighted(ctx)  # branch A: registers `btag` on the session
    branch_b = graphed.vary(ctx, "jes", collections=_jes(ctx))  # branch B: no ambient weight

    assert graphed.weight(branch_b) is None
    assert compile_ir(session, *_jet_pt_universes(branch_b)).shift_after_weight == ()
