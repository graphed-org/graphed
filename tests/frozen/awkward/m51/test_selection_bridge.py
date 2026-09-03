"""m51 anchor C — the `graphed.selection(ctx)` bridge (§6.4a, §9.1).

Contexts expose no mask accessor (§2.2/§9.1 list `labels`/`universe`/`nominal`/`weight`/`variations`
only), so a §2.6-context skim reaches `select=` through `graphed.selection(ctx)`, which implements
§9.1's FULL three-case contract:

  * ROOT context           → `None` (the sink is then reachable only from the loose §2.1a style);
  * mask-derived context   → the `Varied`/Array mask that derived it from its parent (case 1);
  * `vary`-derived context → skips any number of `vary` IDENTITY links to the first non-identity
    link (case: the walk answers with the mask below the `vary`);
  * universe/nominal-derived context → that label's member of the argument's OWN selection: an
    UNVARIED Array in the GRANDparent's row space, `None` when the argument is a root context
    (case 2 — NOT a thin wrapper over the private `_selection()`, which returns `None` on a project
    link, so this half fails until the real case-2 walk exists).

Bridge writes round-trip identically to the same skim written with the mask passed by hand. The
`vary`-link discriminator and the re-recorded-mask control pin §6.4a(2a)'s admission rule against the
two obvious wrong implementations (bare handle `is`, and Array/object `is`).
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import pytest
from m51_write_fixtures import as_list, events_context, raw_events

import graphed
import graphed.awkward as ga
from graphed.awkward import gak
from graphed.errors import GraphedError

pytest.importorskip("pyarrow")


def _event_mask(events: Any) -> Any:
    """A flat per-event predicate read THROUGH the context (carries the context's handle)."""
    return gak.num(events.Jet) >= 2


def _eager_jets_selected() -> ak.Array:
    raw = raw_events()
    return raw.Jet[ak.num(raw.Jet, axis=1) >= 2]


def _skim(record: Any, select: Any, dest: str) -> dict[str, ak.Array]:
    paths = ga.to_parquet(record, dest, select=select)  # type: ignore[call-arg]
    assert len(paths) == 1
    return ga.read_varied(paths[0])  # type: ignore[attr-defined]


def test_selection_on_a_root_context_is_none() -> None:
    _session, events = events_context()
    assert graphed.selection(events) is None  # type: ignore[attr-defined]


def test_bridge_write_roundtrips_identically_to_the_hand_written_mask(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _session, events = events_context()
    mask = _event_mask(events)
    sel = events[mask]

    bridge = _skim(events.Jet, graphed.selection(sel), str(tmp_path / "bridge"))  # type: ignore[attr-defined]
    by_hand = _skim(events.Jet, mask, str(tmp_path / "hand"))

    assert list(bridge) == list(by_hand) == ["nominal"]
    assert as_list(bridge["nominal"]) == as_list(by_hand["nominal"])
    assert as_list(bridge["nominal"]) == as_list(_eager_jets_selected())


def test_bridge_walks_vary_identity_links_and_roundtrips_the_prevary_spelling(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _session, events = events_context()
    mask = _event_mask(events)
    sel = events[mask]
    w = gak.prod(1.0 + sel.Jet.btag, axis=1)
    sel2 = graphed.vary(sel, "btag", w, is_weight=True, up=w * 1.1, down=w * 0.9)

    pre = _skim(events.Jet, graphed.selection(sel), str(tmp_path / "pre"))  # type: ignore[attr-defined]
    post = _skim(events.Jet, graphed.selection(sel2), str(tmp_path / "post"))  # type: ignore[attr-defined]

    assert list(pre) == list(post) == ["nominal"]
    assert as_list(pre["nominal"]) == as_list(post["nominal"])


def test_bridge_admits_a_vary_link_between_record_and_mask_handles(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The discriminator §6.4a(2a) needs: record handle `E1`, mask handle `E2`, joined ONLY by a
    `vary` identity link. A bare `context_of(mask) is context_of(record)` implementation REFUSES this
    legal configuration; the vary-link walk ACCEPTS it. Both spellings write the same rows."""
    _session, e1 = events_context()
    w = gak.prod(1.0 + e1.Jet.btag, axis=1)
    e2 = graphed.vary(
        e1, "pu", w, is_weight=True,
        up=gak.prod(1.0 + e1.Jet.btag * 1.1, axis=1), down=gak.prod(1.0 + e1.Jet.btag * 0.9, axis=1),
    )
    mask = gak.num(e2.Jet) >= 2
    sel = e2[mask]

    across_link = _skim(e1.Jet, graphed.selection(sel), str(tmp_path / "e1"))  # type: ignore[attr-defined]
    from_e2 = _skim(e2.Jet, graphed.selection(sel), str(tmp_path / "e2"))  # type: ignore[attr-defined]

    assert list(across_link) == list(from_e2) == ["nominal"]
    assert as_list(across_link["nominal"]) == as_list(from_e2["nominal"])


def test_bridge_accepts_a_re_recorded_equal_mask_expression(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """§6.4a(2a) is handle equality, not Array/object identity: a mask RE-RECORDED as a distinct
    Python object (same context handle, same interned node id) must be accepted."""
    _session, events = events_context()
    sel = events[_event_mask(events)]
    re_recorded = _event_mask(events)  # a fresh recording; hash-consing gives it the same node id
    assert re_recorded is not graphed.selection(sel)  # type: ignore[attr-defined]  # distinct objects

    out = _skim(events.Jet, re_recorded, str(tmp_path / "rr"))
    assert as_list(out["nominal"]) == as_list(_eager_jets_selected())


def test_selection_on_a_universe_nominal_context_is_a_grandparent_array_and_refuses_downstream(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Case 2: `graphed.selection(graphed.nominal(sel))` returns that label's member of `sel`'s own
    selection — an UNVARIED Array in the GRANDparent's row space (here `sel`'s parent selection mask),
    `None` for a root argument. Passing it as `select=` for a record read from `sel` is REFUSED by
    (2a): a universe/nominal projection link is not admitted, so the mask lives one row space up."""
    _session, events = events_context()
    mask = _event_mask(events)
    sel = events[mask]

    projected = graphed.selection(graphed.nominal(sel))  # type: ignore[attr-defined]
    assert not isinstance(projected, graphed.Varied)  # an unvaried Array, not a container
    assert as_list(projected) == as_list(mask)  # the argument's selection, in the grandparent space
    assert graphed.selection(graphed.nominal(events)) is None  # type: ignore[attr-defined]  # root argument

    # (b): the projected mask lives one row space above a record read from `sel` -> record-time refuse
    with pytest.raises(GraphedError):
        ga.to_parquet(sel.Jet, str(tmp_path / "nope"), select=projected)  # type: ignore[call-arg]
