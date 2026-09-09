"""m56 §2 item 1, the two precisions the label set alone does not pin: the member is judged by its
node AT the label, not by every universe it carries, and the factor is resolved by the two-level,
point-restricted read at that label, not by a one-level read of the container.
"""

from __future__ import annotations

from m56_fanout_fixtures import (
    HF_SF,
    JES_SF,
    PROBE_SF,
    m56_base,
    m56_both_kind,
    m56_joints,
    m56_minted,
    m56_sf,
    m56_two_both_kind,
    m56_weight_family,
)

import graphed


def m56_per_universe_member(tainted: str) -> tuple[object, object, object]:
    """A `jes`-varied member whose universes are built one at a time, so a factor read can be placed
    in ONE of them. `tainted` picks which universe reads a `jes` weight member the design does not
    resolve at that label: the member's `"nominal"`, or its `jes_up` reading the factor's `jes_down`.

    Returns the registered context, the member, and the factor node the taint reads.
    """
    _session, ctx, _seed = m56_base()
    registered, sjets = m56_both_kind(ctx)
    up_jets = graphed.member_of(sjets, "jes_up")
    down_jets = graphed.member_of(sjets, "jes_down")
    read = graphed.member_of(
        m56_sf(sjets, JES_SF["up" if tainted == "nominal" else "down"]),
        "jes_up" if tainted == "nominal" else "jes_down",
    )
    central = m56_sf(graphed.nominal(sjets), 1.0)
    universes = {"up": m56_sf(up_jets, PROBE_SF["up"]), "down": m56_sf(down_jets, PROBE_SF["up"])}
    if tainted == "nominal":
        central = central + 0.0 * read
    else:
        universes["up"] = universes["up"] + 0.0 * read
    return registered, graphed.vary(central, "jes", **universes), read


def test_a_factor_read_in_the_nominal_universe_alone_fans_out_at_every_label() -> None:
    registered, member, read = m56_per_universe_member("nominal")

    # the taint is a real varied member, not the factor's central: the read is worth excluding
    assert read.node_id != graphed.member_of(member, "jes_up").node_id
    weight = graphed.weight(m56_weight_family(registered, "probe", member))
    assert m56_minted(weight, "probe") == m56_joints("probe", "jes")


def test_reading_the_factors_other_universe_is_not_composition_at_this_label() -> None:
    registered, member, read = m56_per_universe_member("jes_up")

    assert read.node_id != graphed.member_of(member, "jes_down").node_id
    weight = graphed.weight(m56_weight_family(registered, "probe", member))
    assert m56_minted(weight, "probe") == m56_joints("probe", "jes")


def test_a_member_that_is_a_factors_two_level_member_is_composed_at_that_label_alone() -> None:
    _session, ctx, sjets = m56_two_both_kind()

    # `jer`'s central expression: at a `jes` label it two-level-resolves to this member's own node
    weight = graphed.weight(m56_weight_family(ctx, "probe", m56_sf(sjets, HF_SF["nominal"])))
    assert m56_minted(weight, "probe") == m56_joints("probe", "jer")
