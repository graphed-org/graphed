"""M17: structure-op parity with dask-awkward (parity plan P1).

Every case records the operation through `gak`, checks the typetracer form exists at record time,
and compares the materialization EXACTLY (ak.array_equal) against the same ak.* call on the raw
array. Multi-output operations (broadcast_arrays, unzip) come back as tuples of deferred arrays.
"""

from __future__ import annotations

import awkward as ak
from m16_helpers import recorded, session_events

from graphed_awkward import gak, project, project_buffers

JAGGED = [[3.0, 1.0], [], [7.0, 2.0, 5.0], [4.0]]
WITH_NONE = [[3.0, None], [], [7.0, None, 5.0], [None]]


def _check(s, deferred, ref) -> None:  # type: ignore[no-untyped-def]
    assert s.form(deferred).is_typetracer
    assert ak.array_equal(ak.Array(s.materialize(deferred)), ref, equal_nan=True)


def test_sort_and_ravel_and_run_lengths() -> None:
    s, g, raw = session_events()
    _check(s, gak.sort(g.Jet.pt, axis=1), ak.sort(raw.Jet.pt, axis=1))
    _check(s, gak.sort(g.Jet.pt, axis=1, ascending=False), ak.sort(raw.Jet.pt, axis=1, ascending=False))
    _check(s, gak.ravel(g.Jet.pt), ak.ravel(raw.Jet.pt))
    _check(s, gak.run_lengths(gak.sort(g.met, axis=0)), ak.run_lengths(ak.sort(raw.met, axis=0)))


def test_mask_is_none_singletons_pad_none() -> None:
    s, g, raw = session_events()
    cond = g.met > 20.0
    _check(s, gak.mask(g.met, cond), ak.mask(raw.met, raw.met > 20.0))
    _check(s, gak.is_none(gak.mask(g.met, cond)), ak.is_none(ak.mask(raw.met, raw.met > 20.0)))
    _check(s, gak.singletons(gak.mask(g.met, cond)), ak.singletons(ak.mask(raw.met, raw.met > 20.0)))
    _check(s, gak.pad_none(g.Jet.pt, 3, axis=1), ak.pad_none(raw.Jet.pt, 3, axis=1))
    _check(
        s,
        gak.pad_none(g.Jet.pt, 2, axis=1, clip=True),
        ak.pad_none(raw.Jet.pt, 2, axis=1, clip=True),
    )


def test_fill_drop_none_roundtrip_with_pad() -> None:
    s, g, raw = session_events()
    padded = gak.pad_none(g.Jet.pt, 3, axis=1)
    _check(s, gak.fill_none(padded, -1.0), ak.fill_none(ak.pad_none(raw.Jet.pt, 3, axis=1), -1.0))


def test_unflatten_and_regular_conversions() -> None:
    s, g, raw = session_events()
    counts = gak.num(g.Jet, axis=1)
    flat = gak.ravel(g.Jet.pt)
    _check(s, gak.unflatten(flat, counts), ak.unflatten(ak.ravel(raw.Jet.pt), ak.num(raw.Jet, axis=1)))
    padded = gak.fill_none(gak.pad_none(g.Jet.pt, 3, axis=1, clip=True), 0.0)
    reg = gak.to_regular(padded, axis=1)
    ref_reg = ak.to_regular(ak.fill_none(ak.pad_none(raw.Jet.pt, 3, axis=1, clip=True), 0.0), axis=1)
    _check(s, reg, ref_reg)
    _check(s, gak.from_regular(reg, axis=1), ak.from_regular(ref_reg, axis=1))


def test_full_like_nan_to_num_isclose() -> None:
    s, g, raw = session_events()
    _check(s, gak.full_like(g.Jet.pt, 9.5), ak.full_like(raw.Jet.pt, 9.5))
    bad = g.Jet.pt / (g.Jet.pt - g.Jet.pt)  # inf/nan factory
    ref_bad = raw.Jet.pt / (raw.Jet.pt - raw.Jet.pt)
    _check(s, gak.nan_to_num(bad), ak.nan_to_num(ref_bad))
    _check(s, gak.isclose(g.Jet.pt, g.Jet.pt + 1e-9), ak.isclose(raw.Jet.pt, raw.Jet.pt + 1e-9))


def test_arg_variants_record_and_match() -> None:
    s, g, raw = session_events()
    _check(s, gak.argcombinations(g.Jet, 2), ak.argcombinations(raw.Jet, 2))
    got = gak.argcartesian([g.Jet.pt, g.Jet.eta])
    ref = ak.argcartesian([raw.Jet.pt, raw.Jet.eta])
    _check(s, got, ref)


def test_without_field_and_values_astype() -> None:
    s, g, raw = session_events()
    slimmed = gak.without_field(g.Jet, "eta")
    ref = ak.without_field(raw.Jet, "eta")
    _check(s, slimmed, ref)
    _check(s, gak.values_astype(g.met, "float32"), ak.values_astype(raw.met, "float32"))


def test_broadcast_arrays_and_unzip_are_tuples_of_deferred_arrays() -> None:
    s, g, raw = session_events()
    a, b = gak.broadcast_arrays(g.met, g.Jet.pt)
    ra, rb = ak.broadcast_arrays(raw.met, raw.Jet.pt)
    _check(s, a, ra)
    _check(s, b, rb)
    pt, eta = gak.unzip(g.Jet)
    _check(s, pt, raw.Jet.pt)
    _check(s, eta, raw.Jet.eta)


def test_fields_subset_getitem_evaluates() -> None:
    s, g, raw = session_events()
    sub = g.Jet[["pt"]]
    node = recorded(s, sub)
    assert node["name"] == "fields"
    got = ak.Array(s.materialize(sub))
    assert got.fields == ["pt"]
    assert ak.array_equal(got.pt, raw.Jet.pt)


def test_to_list_is_eager_sugar() -> None:
    _s, g, raw = session_events()
    assert gak.to_list(g.met) == ak.to_list(raw.met)


def test_structure_ops_stay_fusible() -> None:
    s, g, _ = session_events()
    for arr in (gak.sort(g.Jet.pt, axis=1), gak.pad_none(g.Jet.pt, 3, axis=1), gak.ravel(g.Jet.pt)):
        assert recorded(s, arr)["kind"] == "op"


def test_projection_through_structure_only_touches() -> None:
    _s, g, _ = session_events()
    out = gak.num(gak.pad_none(g.Jet.pt, 3, axis=1), axis=1)
    bufs = project_buffers(out).buffers_for("events")
    assert set(bufs) <= {"Jet", "Jet.pt"}  # nothing outside the Jet branch is touched
    masked = gak.mask(g.Jet.pt, g.Jet.pt > 2.0)
    assert project(masked).columns_for("events") == frozenset({"Jet.pt"})
