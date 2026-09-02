"""Two context-program laws the m48 frozen suite leaves unstated, so the loose-`vary` anchors stay
green while the context path answers wrongly.

`_mask_key` (§2.6c): a derivation's identity is its PER-LABEL node ids. Keyed on the nominal member
alone, two different varied selections whose nominal universes coincide share one memoised derived
context — silent corruption, since the second selection's own universes are never applied.

`EventContext._project` (§2.2/§6.1d kind (3)): projecting to a universe stamps the projected context
onto every value it carries, and that stamp must carry the variation labels §2.5's unreached-label
walk reads. Dropped, the label a program actually filled is reported as reaching no output.
"""

from __future__ import annotations

from m49_ctx_fixtures import events_context, jes_collections

import graphed
from graphed import compile_ir
from graphed.awkward import gak


def test_two_selections_sharing_a_nominal_universe_derive_distinct_contexts() -> None:
    session, ctx = events_context()
    njets = gak.num(ctx.Jet)
    shared_nominal = njets >= 2
    loose = graphed.vary(shared_nominal, "sel", up=(njets >= 3))
    tight = graphed.vary(shared_nominal, "sel", up=(njets >= 5))
    assert graphed.nominal(loose).node_id == graphed.nominal(tight).node_id, "the memo key's decoy"

    first = ctx[loose]
    second = ctx[tight]
    assert first is not second

    counts = [len(session.materialize(gak.num(graphed.universe(ctx_, "sel_up").Jet))) for ctx_ in (first, second)]
    assert counts[0] != counts[1], "the second selection's own universe decides its own row set"


def test_a_label_read_through_a_projected_context_still_reaches_the_output() -> None:
    session, ctx = events_context()
    shifted = graphed.vary(ctx, "jes", collections=jes_collections(ctx))
    up = graphed.universe(shifted, "jes_up")

    compiled = compile_ir(session, gak.sum(up.Jet.pt, axis=1))

    assert compiled.unreached_labels == ("jes_down",)
