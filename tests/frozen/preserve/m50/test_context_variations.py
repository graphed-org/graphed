"""m50 §9.1 ``graphed.variations`` + §6.2(i-bis) narrowing helpers over an axis-mode histogram.

``graphed.variations(ctx)`` reports the context's registry as ``{name: {tag: (kind, value | None)}}``
with the two-word kind vocabulary ``"weight"``/``"shift"``, the numeric tag parsed under BOTH the
canonical e-form and the datacard p-form, and a non-numeric tag carrying no value (not raising).

The §6.2(i-bis) helpers read a variation axis off a bare histogram OBJECT: the axis is recognised by
``axis.__dict__.get("name") == "variation"`` (the ``StrCategory(..., name=)`` kwarg is a ``TypeError``,
so the fixture sets the name through ``__dict__``). ``graphed.labels`` reorders ``"nominal"``-first
then axis order while the STORED order stays lexicographic; ``universe``/``nominal`` slice that axis.
These live here — not in graphed-histogram — so G1's ``accessors.py`` edit is covered by graphed's own
frozen suite (no ``graphed_histogram`` import: graphed CI does not install it).

Freeze-time spelling this suite pins: ``graphed.variations`` (name settled by §9.1) is homed in
``accessors.py`` beside ``labels``/``universe``/``nominal``/``weight``.
"""

from __future__ import annotations

from typing import Any

import boost_histogram as bh
import numpy as np
from graphed_corpus import make_events

import graphed
import graphed.awkward as ga
from graphed import Session, vary
from graphed.awkward import AwkwardBackend, from_awkward, gak


def _ctx_with_families() -> Any:
    """A context carrying ONE weight family and ONE shift family, over tags that exercise both
    numeric parsers (``5em1``/``m15em1`` e-form, ``2p5`` p-form) and a non-numeric tag (``up``)."""
    events = make_events(n_events=100, seed=51)
    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", events)
    ctx = ga.gnano.events(ev)
    w = ev.MET.pt
    ctx = vary(
        ctx, "sf", w, is_weight=True,
        variations={"5em1": w, "m15em1": w * 1.1, "2p5": w * 1.2, "up": w * 1.3},
    )  # fmt: skip
    jets = ctx.Jet
    ctx = vary(
        ctx, "jes",
        collections={"Jet": {"up": gak.with_field(jets, jets.pt * 1.05, "pt"),
                             "down": gak.with_field(jets, jets.pt * 0.95, "pt")}},
    )  # fmt: skip
    return ctx


def test_variations_reports_kind_and_parsed_value_per_tag() -> None:
    v = graphed.variations(_ctx_with_families())
    # weight family: both parsers, and the non-numeric tag carries no value (does not raise)
    assert v["sf"]["5em1"] == ("weight", 0.5)
    assert v["sf"]["m15em1"] == ("weight", -1.5)
    assert v["sf"]["2p5"] == ("weight", 2.5)
    assert v["sf"]["up"] == ("weight", None)
    # shift family: the OTHER kind of the two-word vocabulary; identifier tags carry no value
    assert v["jes"]["up"] == ("shift", None)
    assert v["jes"]["down"] == ("shift", None)
    assert {kind for fam in v.values() for kind, _ in fam.values()} == {"weight", "shift"}


# ---- §6.2(i-bis): narrowing helpers over a bare axis-mode histogram ------------------------------
_STORED_ORDER = ("jes_down", "jes_up", "nominal")  # lexicographic; its first bin is NOT "nominal"


def _axis_mode_hist() -> bh.Histogram:
    h = bh.Histogram(bh.axis.Regular(5, 0.0, 1.0), bh.axis.StrCategory(list(_STORED_ORDER), growth=True))
    h.axes[1].__dict__["name"] = "variation"  # the kwarg form is a TypeError (§6.2 i-bis)
    h.fill([0.1, 0.2], ["jes_up", "jes_up"])
    h.fill([0.3], ["jes_down"])
    h.fill([0.4, 0.5, 0.6], ["nominal", "nominal", "nominal"])
    return h


def test_labels_reorders_nominal_first_over_the_variation_axis() -> None:
    h = _axis_mode_hist()
    assert tuple(h.axes[1]) == _STORED_ORDER  # stored order stays lexicographic
    # reordered "nominal"-first then axis order — NOT the ("nominal",) the unvaried arm gives today
    assert graphed.labels(h) == ("nominal", "jes_down", "jes_up")


def test_universe_and_nominal_slice_the_variation_axis() -> None:
    h = _axis_mode_hist()
    assert np.array_equal(graphed.universe(h, "jes_up").view(), h[{1: bh.loc("jes_up")}].view())
    # nominal is the SLICE, not the whole histogram, and it is a DIFFERENT slice from jes_up
    assert np.array_equal(graphed.nominal(h).view(), h[{1: bh.loc("nominal")}].view())
    assert not np.array_equal(graphed.nominal(h).view(), graphed.universe(h, "jes_up").view())
