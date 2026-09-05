"""vary-m52-C6 — design §4.4 / §6-C6: the joint universe is not the factorized product.

Three legs of ONE reldiff instrument, all measured here so no absolute constant crosses the
isolation boundary: A over the graphed universes (pT-binned SF, live selection), B off the eager
reference with a pT-INDEPENDENT SF, C off the reference with the SF flat AND the selection frozen at
nominal kinematics. C removes every cross term and reads machine zero; B is the real JES
selection-migration cross term and is A's denominator.
"""

from __future__ import annotations

from graphed_corpus.histograms import bin_values
from m52_corpus_fixtures import (
    BTAG_ONLY_LABEL,
    JOINT_COORD,
    JOINT_LABEL,
    factorized,
    integral,
    joint_program,
    reference_hist,
    reference_reldiff,
    reldiff,
    universe_hist,
)

#: the pT-dependence of the SF must contribute at least this multiple of the selection-migration
#: cross term alone; both sides regenerate on every run, so drift fails the ordering, not a constant
K = 3.0

#: `C` is a difference of two identical sums, so it reads exact zero; the bound is float slack only
MACHINE_ZERO = 1e-8


def test_the_joint_universe_is_not_the_factorized_product_and_equals_the_direct_reference() -> None:
    b_flat_live = reference_reldiff(pt_dependent=False)
    c_flat_frozen = reference_reldiff(pt_dependent=False, freeze_selection=True)

    program = joint_program()
    hists = {
        label: universe_hist(program, label)
        for label in ("nominal", "jes_up", BTAG_ONLY_LABEL, JOINT_LABEL)
    }
    integrals = {label: integral(h) for label, h in hists.items()}
    a_graphed = reldiff(
        integrals[JOINT_LABEL],
        factorized(
            jes_only=integrals["jes_up"],
            btag_only=integrals[BTAG_ONLY_LABEL],
            nominal=integrals["nominal"],
        ),
    )

    assert c_flat_frozen < MACHINE_ZERO, (
        f"a flat SF with the selection frozen leaves no cross term, but the instrument read "
        f"{c_flat_frozen!r}"
    )
    assert b_flat_live > 0.0, (
        f"JES moves jets across pt > 25 and events across n_good >= 4, so the flat/live leg is a "
        f"real cross term, but the instrument read {b_flat_live!r}"
    )

    assert bin_values(hists[JOINT_LABEL]) == bin_values(reference_hist(**JOINT_COORD)), (
        f"the joint universe {JOINT_LABEL!r} does not reproduce the eager reference at "
        f"{JOINT_COORD}: integral {integrals[JOINT_LABEL]!r} vs "
        f"{integral(reference_hist(**JOINT_COORD))!r}"
    )
    assert bin_values(hists[JOINT_LABEL]) != bin_values(hists[BTAG_ONLY_LABEL]), (
        f"{JOINT_LABEL!r} equals {BTAG_ONLY_LABEL!r}: resolution took nominal for the JES coordinate"
    )

    assert a_graphed > K * b_flat_live, (
        f"pT-binned/live {a_graphed!r} is not {K}x the flat/live {b_flat_live!r} "
        f"(flat/frozen-selection control {c_flat_frozen!r})"
    )
