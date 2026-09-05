"""vary-m53b §4, §5, plan §7.3: a PRESCRIBED off-diagonal ``{jes: 1, btag: -1}`` — a reparameterized
direction an analyst names by hand — re-points an independent member to a two-axis point over two
DIFFERENT foreign families. Both coordinates are validated by the SAME carrier-reachability walk;
numeric coordinates canonicalize ``1`` → ``"1"`` and ``-1`` → ``"m1"`` (probed on the prune path).

This is a class: the registered-carrier member re-points and passes; the same call against a context
that registered neither ``jes`` nor ``btag`` is refused by carrier-reachability, naming what IS
registered. Non-vacuity: the passing member is refused today for feature-absence ("names no joint the
fanout of 'corr' derives; the derived joints are []").
"""

from __future__ import annotations

import pytest
from m53b_offgrid_fixtures import offdiag_axes, unregistered_context

import graphed
from graphed.errors import GraphedError


def test_a_prescribed_off_diagonal_repoints_an_independent_member() -> None:
    """The class, paired in one call: with ``jes`` / ``btag`` registered the point re-points and
    passes; against a context that registered neither, carrier-reachability refuses it rather than
    silently resolving to nominal, naming what IS registered."""
    _session, btag_ctx, pt = offdiag_axes()  # jes, btag registered as carriers (tags 1 / -1)
    base = pt * 0.5  # independent

    registered = graphed.vary(
        btag_ctx,
        "corr",
        base,
        is_weight=True,
        variations={"off": base * 1.1},
        points=[{"corr": "off", "jes": 1, "btag": -1}],
    )

    point = graphed.points(graphed.weight(registered))["corr_off"]
    assert point == {"jes": "1", "btag": "m1"}  # both axes, canonicalized; own axis dropped
    assert "corr" not in point
    assert "corr_off" in graphed.labels(graphed.weight(registered))  # label kept

    # the refused member of the class: neither axis is a carrier
    _s2, bare_ctx, bare_pt = unregistered_context()
    bare = bare_pt * 0.5
    with pytest.raises(GraphedError) as caught:
        graphed.vary(
            bare_ctx,
            "corr",
            bare,
            is_weight=True,
            variations={"off": bare * 1.1},
            points=[{"corr": "off", "jes": 1, "btag": -1}],
        )
    message = str(caught.value)
    assert "btag" in message  # names the unreachable axis
    assert "corr" in message  # and what the call can see
