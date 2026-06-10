"""M16.2: the M12 structural rule reaches the awkward reducers (parity plan P0).

In awkward semantics axis=0 is the event (partitioned) axis: reducing over it — or over
everything (axis=None) — crosses partitions and records a BOUNDARY reduction node for the M7
tree reduction; an inner-axis reduction (axis>=1, the per-event reductions HEP lives on) is
partition-local and records a FUSIBLE op. The old name-based classification made EVERY ak.sum a
boundary, needlessly blocking fusion of per-event work — the witness pins both directions.
"""

from __future__ import annotations

import pytest
from m16_helpers import recorded, session_events

from graphed_awkward import gak

REDUCER_KINDS = [
    "sum",
    "any",
    "all",
    "count",
    "count_nonzero",
    "min",
    "max",
    "prod",
    "mean",
    "std",
    "var",
    "ptp",
]


@pytest.mark.parametrize("kind", REDUCER_KINDS)
def test_inner_axis_reductions_are_fusible_ops(kind: str) -> None:
    s, g, _ = session_events()
    out = getattr(gak, kind)(g.Jet.pt, axis=1)
    node = recorded(s, out)
    assert node["kind"] == "op", f"gak.{kind}(axis=1) is per-event work and must stay fusible"
    assert node["params"]["axis"] == 1  # type: ignore[index]


@pytest.mark.parametrize("kind", REDUCER_KINDS)
def test_global_reductions_are_boundary_nodes(kind: str) -> None:
    s, g, _ = session_events()
    out = getattr(gak, kind)(g.met, axis=None)
    assert recorded(s, out)["kind"] == "reduction"


def test_axis_zero_is_a_boundary_too() -> None:
    s, g, _ = session_events()
    assert recorded(s, gak.sum(g.met, axis=0))["kind"] == "reduction"
    assert recorded(s, gak.max(g.met, axis=0))["kind"] == "reduction"


def test_softmax_is_always_partition_local() -> None:
    s, g, _ = session_events()
    assert recorded(s, gak.softmax(g.Jet.pt, axis=1))["kind"] == "op"


def test_two_array_reducers_follow_the_same_rule() -> None:
    s, g, _ = session_events()
    assert recorded(s, gak.corr(g.met, g.met, axis=None))["kind"] == "reduction"
    assert recorded(s, gak.covar(g.Jet.pt, g.Jet.eta, axis=1))["kind"] == "op"
    assert recorded(s, gak.linear_fit(g.Jet.pt, g.Jet.eta, axis=1))["kind"] == "op"


def test_argminmax_and_count_keep_their_m3_shape() -> None:
    # M3 recorded these with axis params already; the rule classifies them too
    s, g, _ = session_events()
    assert recorded(s, gak.argmin(g.Jet.pt, axis=1))["kind"] == "op"
    assert recorded(s, gak.num(g.Jet, axis=1))["kind"] == "op"
