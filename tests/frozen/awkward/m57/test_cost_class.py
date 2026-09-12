"""m57 §3: an overlay leaves the composition's COST CLASS.

The composition stays a balanced tree over the product runs between overlays, with the overlay applied
by replacement at the labels its family covers — not a walk that re-multiplies the list per label,
which would put the read back in the quadratic this design exists to avoid. Interning makes
minted-node counts blind to such a walk, so the work is counted at the record call; the minted nodes
are pinned only as a ceiling.
"""

from __future__ import annotations

from typing import Any

from m57_dedupe_fixtures import (
    MU,
    PU,
    m57_ambient_values,
    m57_base,
    m57_delta,
    m57_factor,
    m57_node_count,
    m57_OpSpy,
    m57_oracle_values,
    m57_overlay,
    m57_pu,
    m57_scaled,
    m57_weight,
)

import graphed
from graphed.awkward import gak

#: the factor counts the cost class is measured across — wide enough that a per-label re-multiply
#: walk leaves a constant factor of the tree
WIDTHS = (2, 4, 8, 16)
#: the constant factor the overlay read must stay within
BUDGET = 4


def _program(width: int, *, overlay: bool) -> tuple[Any, Any, list[Any]]:
    """`width` pure-weight factors, optionally with one overlay inserted halfway; the memo is made
    stale before returning, so the caller's read composes from the list in both shapes."""
    session, ctx = m57_base()
    jets = ctx["Jet"]
    after: Any = ctx
    ops: list[Any] = []
    for index in range(width):
        central = m57_pu(jets) * (2.0**-index)
        members = m57_scaled(central, PU)
        after = m57_weight(after, f"f{index}", central, members)
        ops.append(m57_factor(f"f{index}", central, members))
        if overlay and index == width // 2 - 1:
            handle = graphed.weight(after)
            after = m57_delta(after, "mu", handle)
            ops.append(m57_overlay("mu", m57_scaled(handle, MU)))
    stale = gak.num(jets) * 1.0
    graphed.vary(stale, "cold", up=stale * 2.0)  # a point mint: every composition memo goes stale
    return session, after, ops


def _read(session: Any, ctx: Any) -> tuple[int, int, Any]:
    """One cold ambient read, with the operations it records and the nodes it mints."""
    before = m57_node_count(session)
    with m57_OpSpy(session) as spy:
        weight = graphed.weight(ctx)
    return spy.calls, m57_node_count(session) - before, weight


def test_an_overlay_leaves_the_compositions_cost_class() -> None:
    """Across a range of factor counts the overlay read stays within a constant factor of the plain
    read's work and within the plain read's nodes plus one product per label plus its own members —
    and every universe is the tree's product with the overlay's replacement applied."""
    for width in WIDTHS:
        plain_session, plain_ctx, _plain_ops = _program(width, overlay=False)
        plain_calls, plain_nodes, plain_weight = _read(plain_session, plain_ctx)
        session, ctx, ops = _program(width, overlay=True)
        calls, nodes, weight = _read(session, ctx)
        labels = graphed.labels(weight)

        assert plain_calls > 0 and calls > 0, width  # the spy really saw the compositions
        assert len(labels) == 2 * width + 3, width
        assert m57_ambient_values(session, weight) == m57_oracle_values(session, ops, labels), width
        assert calls <= BUDGET * plain_calls, (width, calls, plain_calls)
        assert nodes <= plain_nodes + len(labels) + len(MU), (width, nodes, plain_nodes)
        assert len(graphed.labels(plain_weight)) == 2 * width + 1, width
