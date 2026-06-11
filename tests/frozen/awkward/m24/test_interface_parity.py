"""M24: gak <-> ak interface parity for genuinely missing functionality (user-directed).

Every gak function that mirrors an awkward operation must surface the SAME interface: shared
parameter DEFAULTS are pinned against awkward itself (the anti-drift pin), and the previously
missing, semantics-bearing parameters are witnessed against eager awkward — sequence/tuple zip,
depth_limit (the NanoEvents-of-collections enabler), inline with_name/parameters, combinations
replacement/axis, reducer keepdims/mask_identity/initial, WEIGHTED moments (HEP's bread and
butter), sort stability, drop_none axis, nan_to_num replacements, isclose equal_nan, and
dtype-preserving *_like constructors. Deliberately absent everywhere: highlevel / attrs (eager
concerns) and behavior (backend-owned, M18 — never a per-call kwarg).
"""

from __future__ import annotations

import inspect

import awkward as ak
import numpy as np
from graphed import Session

from graphed_awkward import AwkwardBackend, from_awkward, gak

JAGGED = ak.Array([[1.0, 2.0, 3.0], [], [4.0, 5.0]])
FLAT = ak.Array([3.0, 1.0, 2.0, 5.0, 4.0])


def _session(*arrays: ak.Array):  # type: ignore[no-untyped-def]
    s = Session(AwkwardBackend())
    return s, [from_awkward(s, f"a{i}", arr) for i, arr in enumerate(arrays)]


def _mat(session: Session, node) -> ak.Array:  # type: ignore[no-untyped-def]
    return ak.Array(session.materialize(node))


# ---- the anti-drift pin -------------------------------------------------------------------------
def test_shared_parameter_defaults_match_awkward() -> None:
    deliberate = {"highlevel", "behavior", "attrs"}
    for name in (
        "sum",
        "prod",
        "any",
        "all",
        "count",
        "count_nonzero",
        "min",
        "max",
        "ptp",
        "mean",
        "var",
        "std",
        "moment",
        "argmin",
        "argmax",
        "sort",
        "argsort",
        "concatenate",
        "local_index",
        "flatten",
        "firsts",
        "num",
        "fill_none",
        "pad_none",
        "combinations",
        "argcombinations",
        "cartesian",
        "argcartesian",
        "zip",
        "drop_none",
        "nan_to_num",
        "isclose",
        "broadcast_arrays",
        "zeros_like",
        "ones_like",
        "full_like",
        "values_astype",
        "singletons",
        "unflatten",
    ):
        ak_params = inspect.signature(getattr(ak, name)).parameters
        gak_params = inspect.signature(getattr(gak, name)).parameters
        for p, spec in gak_params.items():
            if p in ak_params and p not in deliberate:
                ak_default = ak_params[p].default
                if ak_default is inspect.Parameter.empty or spec.default is inspect.Parameter.empty:
                    continue
                assert spec.default == ak_default, (
                    f"gak.{name}({p}={spec.default!r}) diverges from ak default {ak_default!r}"
                )


# ---- default-axis behavior (the divergences fixed by M24) ----------------------------------------
def test_concatenate_default_is_the_event_axis() -> None:
    s, (a, b) = _session(JAGGED, JAGGED * 10.0)
    got = _mat(s, gak.concatenate([a, b]))  # ak default axis=0
    assert ak.array_equal(got, ak.concatenate([JAGGED, JAGGED * 10.0]))


def test_sort_family_defaults_to_the_deepest_axis() -> None:
    s, (a,) = _session(JAGGED)
    assert ak.array_equal(_mat(s, gak.sort(a)), ak.sort(JAGGED))
    assert ak.array_equal(_mat(s, gak.argsort(a)), ak.argsort(JAGGED))
    assert ak.array_equal(_mat(s, gak.local_index(a)), ak.local_index(JAGGED))


def test_argmin_argmax_and_count_default_to_axis_none() -> None:
    s, (a,) = _session(JAGGED)
    assert int(s.materialize(gak.argmax(a))) == int(ak.argmax(JAGGED))  # a scalar index
    assert int(s.materialize(gak.count(a))) == int(ak.count(JAGGED))


# ---- zip parity -----------------------------------------------------------------------------------
def test_zip_accepts_sequences_as_tuple_records() -> None:
    s, (a, b) = _session(JAGGED, JAGGED * 2.0)
    rec = gak.zip([a, b])
    got = _mat(s, rec["1"])  # tuple records: fields "0", "1"
    assert ak.array_equal(got, JAGGED * 2.0)


def test_zip_depth_limit_builds_an_events_record_of_collections() -> None:
    jets = ak.Array([[{"pt": 50.0}], [], [{"pt": 70.0}, {"pt": 20.0}]])
    muons = ak.Array([[{"q": 1}, {"q": -1}], [{"q": 1}], []])  # DIFFERENT jaggedness
    s, (j, m) = _session(jets, muons)
    events = gak.zip({"Jet": j, "Muon": m}, depth_limit=1)  # no deep broadcast: collections coexist
    got = _mat(s, gak.num(events.Muon, axis=1))
    assert got.to_list() == [2, 1, 0]
    ref = ak.zip({"Jet": jets, "Muon": muons}, depth_limit=1)
    assert ak.array_equal(_mat(s, events.Jet.pt), ref.Jet.pt)


def test_zip_with_name_and_parameters_inline() -> None:
    s, (a, b) = _session(JAGGED, JAGGED)
    rec = gak.zip({"x": a, "y": b}, with_name="point", parameters={"origin": "m24"})
    out = _mat(s, rec)
    assert out.layout.content.parameter("__record__") == "point"
    assert out.layout.content.parameter("origin") == "m24"


# ---- combinatorics parity --------------------------------------------------------------------------
def test_combinations_replacement_and_axis() -> None:
    s, (a,) = _session(JAGGED)
    got = _mat(s, gak.combinations(a, 2, replacement=True, fields=["p", "q"]))
    ref = ak.combinations(JAGGED, 2, replacement=True, fields=["p", "q"])
    assert ak.array_equal(got.p, ref.p) and ak.array_equal(got.q, ref.q)
    got0 = _mat(s, gak.argcombinations(a, 2, axis=1))
    assert ak.array_equal(got0["0"], ak.argcombinations(JAGGED, 2, axis=1)["0"])


def test_cartesian_nested_list_and_with_name() -> None:
    s, (a, b) = _session(JAGGED, JAGGED * 3.0)
    got = _mat(s, gak.cartesian([a, b], nested=[0], with_name="pair"))
    ref = ak.cartesian([JAGGED, JAGGED * 3.0], nested=[0], with_name="pair")
    assert ak.array_equal(got["0"], ref["0"])
    assert got.layout.minmax_depth == ref.layout.minmax_depth


# ---- reducer parity --------------------------------------------------------------------------------
def test_reducers_keepdims_and_mask_identity() -> None:
    s, (a,) = _session(JAGGED)
    got = _mat(s, gak.sum(a, axis=1, keepdims=True))
    assert ak.array_equal(got, ak.sum(JAGGED, axis=1, keepdims=True))
    # mask_identity=True: the empty list reduces to None, not the identity
    got_m = _mat(s, gak.sum(a, axis=1, mask_identity=True))
    assert got_m.to_list() == ak.sum(JAGGED, axis=1, mask_identity=True).to_list()
    got_min = _mat(s, gak.min(a, axis=1, mask_identity=False))
    assert ak.array_equal(got_min, ak.min(JAGGED, axis=1, mask_identity=False))


def test_min_max_accept_initial() -> None:
    s, (a,) = _session(JAGGED)
    got = _mat(s, gak.min(a, axis=1, initial=2.5, mask_identity=False))
    assert ak.array_equal(got, ak.min(JAGGED, axis=1, initial=2.5, mask_identity=False))


def test_weighted_moments_match_eager() -> None:
    w = ak.Array([[1.0, 2.0, 1.0], [], [0.5, 3.0]])
    s, (a, gw) = _session(JAGGED, w)
    for g_fn, a_fn in ((gak.mean, ak.mean), (gak.var, ak.var), (gak.std, ak.std)):
        got = _mat(s, g_fn(a, axis=1, weight=gw))
        ref = a_fn(JAGGED, axis=1, weight=w)
        assert np.allclose(
            ak.to_numpy(ak.fill_none(got, np.nan)), ak.to_numpy(ak.fill_none(ref, np.nan)), equal_nan=True
        )
    got_m = _mat(s, gak.moment(a, 2, axis=1, weight=gw))
    ref_m = ak.moment(JAGGED, 2, axis=1, weight=w)
    assert np.allclose(
        ak.to_numpy(ak.fill_none(got_m, np.nan)), ak.to_numpy(ak.fill_none(ref_m, np.nan)), equal_nan=True
    )


def test_sort_stability_parameter_is_recorded_and_runs() -> None:
    s, (a,) = _session(FLAT)
    got = _mat(s, gak.sort(a, axis=0, stable=False))
    assert ak.array_equal(got, ak.sort(FLAT, axis=0, stable=False))


# ---- the long tail ---------------------------------------------------------------------------------
def test_drop_none_axis() -> None:
    arr = ak.Array([[1.0, None, 2.0], [None], [3.0]])
    s, (a,) = _session(arr)
    got = _mat(s, gak.drop_none(a, axis=1))
    assert got.to_list() == ak.drop_none(arr, axis=1).to_list()


def test_nan_to_num_replacements() -> None:
    arr = ak.Array([[np.nan, np.inf], [-np.inf]])
    s, (a,) = _session(arr)
    got = _mat(s, gak.nan_to_num(a, nan=-1.0, posinf=9.0, neginf=-9.0))
    assert got.to_list() == ak.nan_to_num(arr, nan=-1.0, posinf=9.0, neginf=-9.0).to_list()


def test_isclose_equal_nan() -> None:
    x = ak.Array([[np.nan, 1.0]])
    s, (a, b) = _session(x, x)
    got = _mat(s, gak.isclose(a, b, equal_nan=True))
    assert got.to_list() == [[True, True]]


def test_broadcast_arrays_depth_limit() -> None:
    s, (a, b) = _session(JAGGED, ak.Array([10.0, 20.0, 30.0]))
    _got_a, got_b = gak.broadcast_arrays(a, b, depth_limit=1)
    _ref_a, ref_b = ak.broadcast_arrays(JAGGED, ak.Array([10.0, 20.0, 30.0]), depth_limit=1)
    assert ak.array_equal(_mat(s, got_b), ref_b)


def test_like_constructors_preserve_dtype_by_default() -> None:
    arr = ak.values_astype(JAGGED, "float32")
    s, (a,) = _session(arr)
    got = _mat(s, gak.zeros_like(a))
    ref = ak.zeros_like(arr)
    assert str(got.type) == str(ref.type)  # float32 preserved, like awkward — not forced int64
    got1 = _mat(s, gak.ones_like(a))
    assert str(got1.type) == str(ak.ones_like(arr).type)


def test_new_parameters_are_deterministic_in_the_ir() -> None:
    def build() -> bytes:
        s, (a, b) = _session(JAGGED, JAGGED)
        rec = gak.zip({"x": a, "y": b}, with_name="p", parameters={"k": "v"})
        out = gak.sum(gak.combinations(rec, 2, replacement=True)["0"].x, axis=1, keepdims=True)
        return s.serialized_ir(out)

    assert build() == build()
