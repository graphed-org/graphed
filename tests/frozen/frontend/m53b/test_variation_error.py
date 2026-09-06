"""vary-m53u / design §4: the unified-grammar misuse situations raise ``graphed.PointError``
— a ``GraphedError`` subclass carrying a ``situation`` discriminator — not a bare ``GraphedError``.

Each situation is a CLASS with a member at each end: one entry the situation MUST refuse (asserting
``PointError`` AND ``GraphedError``, the ``situation`` string §4 pins, and the message substring
the pre-unification raise carried) and a neighbouring valid entry it MUST admit, so the class is
neither over- nor under-broad. A malformed entry (neither declare-tuple nor placement-Mapping) is an
ill-typed input → ``GraphedTypeError``, not a situation.

Numpy-idiom, awkward-free (the required free-threaded frontend job collects with only pytest +
numpy). COLLECTION safety: ``PointError`` is reached only as ``graphed.PointError`` inside
test bodies (the repo bans in-body imports, PLC0415), and the list-form ``points=`` likewise —
so the file collects against a tree with neither and fails at RUN time for feature-absence:
``AttributeError: module 'graphed' has no attribute 'PointError'`` for the situations, and
``'list' object has no attribute 'items'`` for the malformed / list-form entries.
"""

from __future__ import annotations

from typing import Any

import pytest
from m53b_offgrid_fixtures import (
    VECTOR,
    independent_weight,
    triple_numeric,
    two_axis_context,
)

import graphed
from graphed import Session
from graphed.errors import GraphedError, GraphedTypeError
from graphed.numpy import NumpyBackend, from_record


def _dependent() -> tuple[Any, Any]:
    """A jes-dependent ``corr`` member (fans out over jes) on a jes+jer context."""
    _session, ctx = two_axis_context()
    return ctx, ctx["pt"] * 0.5  # read off the jes-varied pt


def _independent() -> tuple[Any, Any]:
    """An independent ``corr`` member (off the nominal) on a jes+jer context — re-pointed additively."""
    _session, ctx = two_axis_context()
    return ctx, independent_weight(ctx)


def _two_containers() -> tuple[Any, Any]:
    """Two independent containers of one Session, for a declare-side label collision."""
    session = Session(NumpyBackend())
    record = from_record(session, "ev", pt=VECTOR, eta=VECTOR / 10.0)
    return record["pt"], record["eta"]


def test_the_declare_channel_is_live_positive_control() -> None:
    """Positive control — no unified-grammar feature is touched: a pure-declare dict ``vary`` mints
    and reports its point on the current tree, so a green line here proves this file's fixtures and
    ``graphed`` are live, distinguishing the feature-absence failures below from a dead instrument."""
    ctx, ind = _independent()
    registered = graphed.vary(ctx, "corr", ind, is_weight=True, points={"a": ind * 3.0})
    reported = graphed.points(graphed.weight(registered))
    assert reported["nominal"] == {}
    assert reported["corr_a"] == {"corr": "a"}


def test_e1_no_derived_joint_is_unresolved() -> None:
    """E1: a placement over a DEPENDENT member naming a foreign axis its fanout does not derive."""
    ctx, dep = _dependent()  # jes-dependent → the grid is over jes only
    with pytest.raises(graphed.PointError) as caught:
        graphed.vary(
            ctx, "corr", dep, is_weight=True,
            points=[("a", dep * 1.3), {"corr": "a", "jer": "up"}],  # jer is not in the jes grid
        )
    exc = caught.value
    assert isinstance(exc, graphed.PointError)
    assert isinstance(exc, GraphedError)
    assert exc.situation == "unresolved"
    assert "joint" in str(exc)  # names that no joint is derived

    # admitted: the same member, a coordinate the fanout DOES derive
    ctx2, dep2 = _dependent()
    graphed.vary(
        ctx2, "corr", dep2, is_weight=True,
        points=[("a", dep2 * 1.3), {"corr": "a", "jes": "up"}],
    )


def test_e3_unknown_nuisance_is_unresolved() -> None:
    """E3: a placement naming a nuisance registered nowhere this call can see."""
    ctx, dep = _dependent()
    with pytest.raises(graphed.PointError) as caught:
        graphed.vary(
            ctx, "corr", dep, is_weight=True,
            points=[("a", dep * 3.0), {"corr": "a", "nosuch": "up"}],
        )
    exc = caught.value
    assert isinstance(exc, graphed.PointError)
    assert isinstance(exc, GraphedError)
    assert exc.situation == "unresolved"
    assert "nosuch" in str(exc)  # names the offending nuisance

    # admitted: a reachable foreign nuisance
    ctx2, dep2 = _dependent()
    graphed.vary(
        ctx2, "corr", dep2, is_weight=True,
        points=[("a", dep2 * 3.0), {"corr": "a", "jes": "up"}],
    )


def test_e3_unknown_own_tag_is_unresolved() -> None:
    """E3: a placement whose OWN-family coordinate is not a declared tag of this call."""
    ctx, dep = _dependent()
    with pytest.raises(graphed.PointError) as caught:
        graphed.vary(
            ctx, "corr", dep, is_weight=True,
            points=[("up", dep * 3.0), {"corr": "down", "jes": "up"}],  # 'down' is not declared
        )
    exc = caught.value
    assert isinstance(exc, graphed.PointError)
    assert isinstance(exc, GraphedError)
    assert exc.situation == "unresolved"
    assert "down" in str(exc)  # names the offending own tag

    # admitted: the own coordinate IS a declared tag
    ctx2, dep2 = _dependent()
    graphed.vary(
        ctx2, "corr", dep2, is_weight=True,
        points=[("up", dep2 * 3.0), {"corr": "up", "jes": "up"}],
    )


def test_e4_unreachable_foreign_tag_is_unreachable() -> None:
    """E4: an off-grid additive pin whose foreign coordinate is not a registered tag of its axis."""
    ctx, ind = _independent()
    with pytest.raises(graphed.PointError) as caught:
        graphed.vary(
            ctx, "corr", ind, is_weight=True,
            points=[("a", ind * 3.0), {"corr": "a", "jes": "up", "jer": "sideways"}],
        )
    exc = caught.value
    assert isinstance(exc, graphed.PointError)
    assert isinstance(exc, GraphedError)
    assert exc.situation == "unreachable"
    assert "sideways" in str(exc)  # the unreachable coordinate
    assert "jer" in str(exc)  # the axis whose registered tags it fails to name

    # admitted: the same shape with a reachable jer tag re-points cleanly
    ctx2, ind2 = _independent()
    graphed.vary(
        ctx2, "corr", ind2, is_weight=True,
        points=[("a", ind2 * 3.0), {"corr": "a", "jes": "up", "jer": "up"}],
    )


def test_e5_duplicate_label_is_duplicate() -> None:
    """E5: two different points rendering ONE label within a Session (``_check_unique``)."""
    x, y = _two_containers()
    graphed.vary(x, "jes", btag_up=x * 5.0)  # label jes_btag_up at point {jes: btag_up}
    with pytest.raises(graphed.PointError) as caught:
        graphed.vary(y, "jes_btag", up=y * 7.0)  # same label, different point {jes_btag: up}
    exc = caught.value
    assert isinstance(exc, graphed.PointError)
    assert isinstance(exc, GraphedError)
    assert exc.situation == "duplicate"
    assert "jes_btag_up" in str(exc)  # names the colliding label

    # admitted (adversarial): the SAME label at the SAME point, minted twice on independent
    # containers of one Session, is idempotent — a blanket "refuse every re-mint" wrongly rejects it
    x2, y2 = _two_containers()
    graphed.vary(x2, "jes", up=x2 * 1.1)
    graphed.vary(y2, "jes", up=y2 * 1.1)


def test_e6_empty_own_only_is_empty() -> None:
    """E6: a placement carrying only its own-name coordinate (or foreign coordinates all at 0)."""
    ctx, ind = _independent()
    with pytest.raises(graphed.PointError) as caught:
        graphed.vary(
            ctx, "corr", ind, is_weight=True,
            points=[("a", ind * 3.0), {"corr": "a"}],  # only its own coordinate — empty
        )
    exc = caught.value
    assert isinstance(exc, graphed.PointError)
    assert isinstance(exc, GraphedError)
    assert exc.situation == "empty"
    assert "central" in str(exc) or "nominal" in str(exc)  # why empty is nominal, already present

    # admitted (adversarial): a zero foreign coordinate DROPS beside two live ones, leaving a legal
    # two-coordinate point — the class must still admit it (empty must not swallow this)
    _session, num_ctx, pt = triple_numeric()
    base = pt * 0.5  # independent
    graphed.vary(
        num_ctx, "corr", base, is_weight=True,
        points=[("a", base * 1.1), {"corr": "a", "jes": 1, "btag": -1, "jer": 0}],
    )


def test_ec_union_plus_placement_is_conflict() -> None:
    """Ec: ``composes_as_union=True`` collapses every joint away, so a placement cannot coexist."""
    ctx, dep = _dependent()
    with pytest.raises(graphed.PointError) as caught:
        graphed.vary(
            ctx, "corr", dep, is_weight=True,
            points=[("a", dep * 1.3), {"corr": "a", "jes": "up"}],
            composes_as_union=True,
        )
    exc = caught.value
    assert isinstance(exc, graphed.PointError)
    assert isinstance(exc, GraphedError)
    assert exc.situation == "conflict"
    assert "composes_as_union" in str(exc)  # names the incompatible combination, not a bad member

    # admitted #1: composes_as_union WITHOUT a placement collapses cleanly (pure-declare dict form)
    ctx2, dep2 = _dependent()
    graphed.vary(ctx2, "corr", dep2, is_weight=True, points={"a": dep2 * 1.3}, composes_as_union=True)

    # admitted #2: the SAME placement WITHOUT composes_as_union is a valid select — the conflict is
    # exactly the intersection, neither leg alone
    ctx3, dep3 = _dependent()
    graphed.vary(
        ctx3, "corr", dep3, is_weight=True,
        points=[("a", dep3 * 1.3), {"corr": "a", "jes": "up"}],
    )


def test_a_malformed_entry_raises_graphed_type_error() -> None:
    """A list entry that is neither a 2-tuple declare nor a Mapping placement is ill-typed input."""
    ctx, dep = _dependent()
    with pytest.raises(GraphedTypeError):
        graphed.vary(
            ctx, "corr", dep, is_weight=True,
            points=[("a", dep * 1.3), 42],  # 42 is neither a declare-tuple nor a placement
        )

    # admitted: a well-formed declare tuple beside a valid placement does not raise
    ctx2, dep2 = _dependent()
    graphed.vary(
        ctx2, "corr", dep2, is_weight=True,
        points=[("a", dep2 * 1.3), {"corr": "a", "jes": "up"}],
    )
