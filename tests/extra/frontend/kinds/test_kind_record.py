"""`graphed.Kind` is a union, and `variations` reads the lineage's registration record: a row-space
change must not change what a nuisance is reported as, nor refuse a same-name registration the
parent accepts."""

from __future__ import annotations

import sys
from pathlib import Path

import graphed
from graphed import Kind

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "frozen" / "awkward" / "m48"))
from test_vary_stacking import _corpus_shaped_selection, pu_weight


def _kinds(ctx: object) -> dict[str, dict[str, Kind]]:
    return {
        name: {tag: kind for tag, (kind, _) in tags.items()} for name, tags in graphed.variations(ctx).items()
    }


def test_a_kind_is_a_union() -> None:
    both = Kind.WEIGHT | Kind.SHIFT
    assert Kind.WEIGHT in both and Kind.SHIFT in both
    assert Kind.WEIGHT not in Kind.SHIFT
    assert repr(both) == "Kind.WEIGHT|SHIFT" and repr(Kind.SHIFT) == "Kind.SHIFT"


def test_a_masked_child_reports_the_parents_kinds() -> None:
    _session, events, _mask, sel = _corpus_shaped_selection()
    assert _kinds(events) == {
        "pu": {"up": Kind.WEIGHT, "down": Kind.WEIGHT},
        "jes": {"up": Kind.SHIFT, "down": Kind.SHIFT},
    }
    assert _kinds(sel) == _kinds(events)


def test_name_identity_is_accepted_at_a_masked_child_as_at_the_parent() -> None:
    _session, events, _mask, sel = _corpus_shaped_selection()
    for ctx in (events, sel):
        joined = graphed.vary(
            ctx, "jes", pu_weight(ctx, 1.0), is_weight=True, up=pu_weight(ctx, 1.1), down=pu_weight(ctx, 0.9)
        )
        assert _kinds(joined)["jes"] == {"up": Kind.WEIGHT | Kind.SHIFT, "down": Kind.WEIGHT | Kind.SHIFT}
