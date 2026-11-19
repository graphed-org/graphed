"""``ak.mask`` and ``.mask[...]`` on a deferred array and on an event context (graphed#259)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak, gnano, to_parquet
from graphed.systematics import universe

EAGER = ak.Array(
    {
        "run": [1, 2, 3],
        "MET": [{"pt": 5.0}, {"pt": 15.0}, {"pt": 25.0}],
        "Jet": [[{"pt": 30.0}], [], [{"pt": 10.0}, {"pt": 40.0}]],
    }
)
WANT = EAGER.mask[EAGER.MET.pt > 10]


def _context(*, is_data: bool = True) -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    return session, gnano.events(from_awkward("ev", EAGER, session=session), is_data=is_data)


def _as_eager(value: ak.Array) -> tuple[str, list[Any], str]:
    return str(value.type), value.tolist(), type(value.layout).__name__


def _same(session: Session, got: Any, want: ak.Array) -> None:
    assert _as_eager(ak.Array(session.materialize(got))) == _as_eager(want)


@pytest.mark.parametrize(
    "masked",
    [
        lambda x: x.mask[x.MET.pt > 10],
        lambda x: ak.mask(x, x.MET.pt > 10),
        lambda x: gak.mask(x, x.MET.pt > 10),
    ],
    ids=["accessor", "ak.mask", "gak.mask"],
)
@pytest.mark.parametrize("on_context", [False, True], ids=["array", "context"])
def test_every_form_masks_as_eager_does(masked: Callable[[Any], Any], on_context: bool) -> None:
    session, ctx = _context()
    x = ctx if on_context else from_awkward("ev2", EAGER, session=session)
    _same(session, masked(x).MET.pt, WANT.MET.pt)


def test_a_masked_context_is_a_context_reading_masked_collections() -> None:
    session, ctx = _context()
    masked = ctx.mask[ctx.MET.pt > 10]
    assert isinstance(masked, type(ctx))
    _same(session, masked.Jet.pt, WANT.Jet.pt)
    _same(session, masked.run, WANT.run)


def test_one_condition_answers_with_one_masked_context() -> None:
    _session, ctx = _context()
    cut = ctx.MET.pt > 10
    assert ctx.mask[cut] is ak.mask(ctx, cut)
    assert ak.mask(ctx, cut) is not ak.mask(ctx, cut, valid_when=False)


def test_valid_when_false_masks_where_the_condition_holds() -> None:
    session, ctx = _context()
    _same(
        session,
        ak.mask(ctx, ctx.MET.pt > 10, valid_when=False).MET.pt,
        ak.mask(EAGER, EAGER.MET.pt > 10, valid_when=False).MET.pt,
    )


def test_an_assignment_then_a_read_on_a_masked_context() -> None:
    session, ctx = _context()
    masked = ctx.mask[ctx.MET.pt > 10]
    masked["ht"] = gak.sum(masked.Jet.pt, axis=1)
    _same(session, masked.ht, ak.sum(WANT.Jet.pt, axis=1))


def test_a_value_read_before_the_mask_combines_with_a_masked_one() -> None:
    session, ctx = _context()
    met = ctx.MET.pt
    masked = ctx.mask[met > 10]
    _same(session, masked.MET.pt + met, WANT.MET.pt + EAGER.MET.pt)


def test_a_filter_on_a_masked_context() -> None:
    session, ctx = _context()
    masked = ctx.mask[ctx.MET.pt > 10]
    _same(session, masked[masked.run > 1].Jet.pt, WANT[WANT.run > 1].Jet.pt)


def test_a_masked_context_writes_the_bytes_eager_writes(tmp_path: Path) -> None:
    _session, ctx = _context()
    masked = ctx.mask[ctx.MET.pt > 10]
    masked["ht"] = gak.sum(masked.Jet.pt, axis=1)
    (part,) = to_parquet(masked[["run", "ht"]], str(tmp_path / "g"))
    want = EAGER.mask[EAGER.MET.pt > 10]
    want["ht"] = ak.sum(want.Jet.pt, axis=1)
    ak.to_parquet(want[["run", "ht"]], tmp_path / "eager.parquet")
    got, eager = ak.from_parquet(part), ak.from_parquet(tmp_path / "eager.parquet")
    assert {f: _as_eager(got[f]) for f in got.fields} == {
        f: _as_eager(eager[f]) for f in eager.fields
    }
    assert Path(part).read_bytes() == (tmp_path / "eager.parquet").read_bytes()


def test_each_universe_is_masked_by_its_own_condition() -> None:
    session, ctx = _context(is_data=False)
    jets = ctx.Jet
    events = graphed.vary(
        ctx,
        "jes",
        collections={"Jet": {"up": gak.with_field(jets, jets.pt * 2, "pt"), "down": jets}},
    )
    cut = gak.max(events.Jet.pt, axis=1) > 35
    masked = events.mask[cut]
    for label, scale in (("nominal", 1), ("jes_up", 2), ("jes_down", 1)):
        jets_at = ak.with_field(EAGER.Jet, EAGER.Jet.pt * scale, "pt")
        eager = ak.mask(ak.with_field(EAGER, jets_at, "Jet"), ak.max(jets_at.pt, axis=1) > 35)
        _same(session, universe(masked.Jet.pt, label), eager.Jet.pt)


def test_a_field_named_mask_stays_reachable_on_a_context() -> None:
    session = Session(AwkwardBackend())
    ctx = gnano.events(from_awkward("ev", ak.Array({"mask": [1, 2]}), session=session))
    assert ak.to_list(session.materialize(ctx.mask)) == [1, 2]


def test_a_field_named_mask_stays_a_field_on_an_array() -> None:
    session = Session(AwkwardBackend())
    ev = from_awkward("ev", ak.Array({"mask": [1, 2]}), session=session)
    assert ak.to_list(session.materialize(ev.mask)) == [1, 2]


def test_a_context_masked_by_a_varied_condition_carries_its_universes() -> None:
    session, ctx = _context(is_data=False)
    jets = ctx.Jet
    varied = graphed.vary(
        ctx,
        "jes",
        collections={"Jet": {"up": gak.with_field(jets, jets.pt * 2, "pt"), "down": jets}},
    )
    masked = ctx.mask[gak.max(varied.Jet.pt, axis=1) > 35]
    masked["met"] = masked.MET.pt
    jets_up = ak.with_field(EAGER.Jet, EAGER.Jet.pt * 2, "pt")
    _same(
        session, universe(masked, "jes_up").met, EAGER.mask[ak.max(jets_up.pt, axis=1) > 35].MET.pt
    )


def test_a_context_refuses_a_condition_that_is_not_deferred() -> None:
    _session, ctx = _context()
    with pytest.raises(graphed.GraphedError, match="Array/Varied condition"):
        ctx.mask[[True, False, True]]


def test_a_value_read_through_the_parent_is_masked_when_assigned() -> None:
    session, ctx = _context()
    masked = ctx.mask[ctx.MET.pt > 10]
    masked["x"] = ctx.MET.pt
    want = EAGER.mask[EAGER.MET.pt > 10]
    want["x"] = EAGER.MET.pt
    got = ak.Array(session.materialize(masked.x))
    assert (str(got.type), got.tolist()) == (str(want.x.type), want.x.tolist())


def test_a_masked_context_varies_a_collection_built_from_its_parent() -> None:
    session, ctx = _context(is_data=False)
    masked = ctx.mask[ctx.MET.pt > 10]
    jets = ctx.Jet
    events = graphed.vary(
        masked, "jes", collections={"Jet": {"up": gak.with_field(jets, jets.pt * 2, "pt")}}
    )
    up = ak.with_field(WANT.Jet, ak.mask(EAGER.Jet.pt * 2, EAGER.MET.pt > 10), "pt")
    got = ak.Array(session.materialize(universe(events.Jet.pt, "jes_up")))
    assert (str(got.type), got.tolist()) == (str(up.pt.type), up.pt.tolist())


def test_inherited_universes_are_masked_when_a_masked_member_joins_them() -> None:
    session, ctx = _context(is_data=False)
    met = ctx.MET.pt
    first = graphed.vary(met, "a", points={"up": met * 2})
    masked = ctx.mask[met > 10]
    both = graphed.vary(first, "b", points={"up": masked.MET.pt})
    want = ak.mask(EAGER.MET.pt * 2, EAGER.MET.pt > 10)
    got = ak.Array(session.materialize(universe(both, "a_up")))
    assert (str(got.type), got.tolist()) == (str(want.type), want.tolist())


class _NoMaskBackend(AwkwardBackend):
    mask = None  # type: ignore[assignment]


def test_a_backend_without_a_mask_seam_refuses_by_name() -> None:
    session = Session(_NoMaskBackend())
    ctx = gnano.events(from_awkward("ev", EAGER, session=session))
    with pytest.raises(graphed.GraphedError, match="without a mask seam"):
        ctx.mask[ctx.MET.pt > 10]
