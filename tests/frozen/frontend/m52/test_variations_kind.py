"""C4 / design §4.10: `variations()` keeps its m50-frozen `{name: {tag: (kind, value|None)}}` shape
and gains a third kind word.

Today a family registered as both a shift and a weight reports only `"shift"` — the collections pass
`update()`s over the weight pass — so the verb lies about exactly the mechanism §4.8 tells analysts
to use for R1-b.
"""

from __future__ import annotations

from fractions import Fraction

from m52_point_fixtures import dual_registered_family, shift_then_weight_context

import graphed


def test_a_family_registered_as_both_shift_and_weight_reports_both() -> None:
    _session, ctx = dual_registered_family()

    assert graphed.variations(ctx)["jes"] == {"up": ("both", None), "down": ("shift", None)}


def test_the_dual_kind_keeps_the_frozen_value_half() -> None:
    """The shape is m50-frozen: the second element stays `_tags.numeric_value(tag)`."""
    _session, ctx = dual_registered_family(numeric=True)

    assert graphed.variations(ctx)["jes"] == {
        "1": ("both", Fraction(1, 1)),
        "m1": ("shift", Fraction(-1, 1)),
    }


def test_disjoint_families_report_exactly_the_two_m50_kinds() -> None:
    """The m50 anchor guard: on a program with no dual registration the kind vocabulary is
    unchanged, so an implementation reporting `"both"` for a whole family fails here."""
    _session, ctx = shift_then_weight_context()

    reported = graphed.variations(ctx)
    kinds = {kind for family in reported.values() for kind, _value in family.values()}
    assert kinds == {"weight", "shift"}
    assert reported["jes"] == {"up": ("shift", None), "down": ("shift", None)}
    assert reported["btag"] == {"up": ("weight", None), "down": ("weight", None)}
