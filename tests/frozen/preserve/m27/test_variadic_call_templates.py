"""M27 — variadic call templates for the External family (the base tier: no heavy frameworks).

An External's callee — a correction, a model, a served endpoint, a histogram fill — has a real
SIGNATURE: multiple arguments, constants interleaved (a systematic name), named inputs, multiple
axes, multiple weights. M26 shipped every evaluator with a single hard-wired call shape; M27
makes the shape part of the PRESERVED NODE: ``params["args"]`` routes the node's graph inputs
to callee arguments (positional ``[["$0", "$1"], ["$2"]]`` — inner lists stack into feature
matrices — or named ``{"kin": [...], "mask": [...]}``; correctionlib additionally interleaves
constants), and replay MUST obey it exactly. No ``args`` means the legacy convention, pinned
byte-compatible — existing bundles do not change meaning.

This module covers everything testable with the BASE dependencies: correctionlib (a base dep —
native awkward/jagged passthrough, multi-input, positional systematics), the histogram family
(multi-axis, and the new multi-WEIGHT contract), the multi-input Triton pattern through the
injectable fake transport, the xgboost loud rejection, and the legacy-default pins.
"""

from __future__ import annotations

import json
from typing import Any

import awkward as ak
import fake_triton_multi
import numpy as np
import pytest
from graphed import Session
from graphed_awkward import AwkwardBackend, from_awkward
from graphed_corpus import make_events

from graphed_preserve import (
    CORRECTIONLIB_PLUGIN,
    HISTOGRAM_PLUGIN,
    TRITON_PLUGIN,
    XGBOOST_PLUGIN,
    PreserveError,
    build_bundle,
    record_external,
    reproduce,
)

HIST = {"name": "met", "bins": 20, "lo": 0.0, "hi": 200.0}


def _hist(value: Any, weight: Any) -> np.ndarray:
    v = np.asarray(ak.to_numpy(ak.Array(value)), dtype="float64")
    w = np.asarray(ak.to_numpy(ak.Array(weight)), dtype="float64")
    return np.histogram(v, bins=HIST["bins"], range=(HIST["lo"], HIST["hi"]), weights=w)[0].round(6)


# ---------------------------------- correctionlib --------------------------------------------------
JETSF = json.dumps(
    {
        "schema_version": 2,
        "corrections": [
            {
                "name": "jetsf",
                "version": 1,
                "inputs": [
                    {"name": "systematic", "type": "string"},
                    {"name": "pt", "type": "real"},
                    {"name": "eta", "type": "real"},
                ],
                "output": {"name": "sf", "type": "real"},
                "data": {
                    "nodetype": "category",
                    "input": "systematic",
                    "content": [
                        {
                            "key": "nominal",
                            "value": {
                                "nodetype": "formula",
                                "expression": "1.0 + 0.01*x + 0.001*y",
                                "parser": "TFormula",
                                "variables": ["pt", "eta"],
                            },
                        },
                        {
                            "key": "up",
                            "value": {
                                "nodetype": "formula",
                                "expression": "1.1 + 0.01*x + 0.001*y",
                                "parser": "TFormula",
                                "variables": ["pt", "eta"],
                            },
                        },
                    ],
                },
            }
        ],
    }
).encode()


def _eval(plugin: Any, payload: bytes, params: dict[str, Any], inputs: list[Any]) -> Any:
    resource = plugin.load(payload, params)
    try:
        return plugin.evaluate(resource, params, inputs)
    finally:
        plugin.close(resource)


def test_correctionlib_jagged_awkward_passes_through_natively() -> None:
    import correctionlib  # noqa: PLC0415

    pt = ak.Array([[30.0, 50.0], [], [80.0]])
    eta = ak.Array([[0.5, 1.0], [], [2.0]])
    params = {"name": "jetsf", "args": ["nominal", "$0", "$1"]}
    got = _eval(CORRECTIONLIB_PLUGIN, JETSF, params, [pt, eta])
    want = correctionlib.CorrectionSet.from_string(JETSF.decode())["jetsf"].evaluate("nominal", pt, eta)
    assert ak.num(got, axis=1).tolist() == [2, 0, 1]  # JAGGED STRUCTURE PRESERVED
    assert ak.all(got == want)


def test_correctionlib_constants_route_systematics_at_any_position() -> None:
    pt = np.array([30.0, 50.0])
    eta = np.array([0.5, 1.0])
    nominal = np.asarray(
        _eval(CORRECTIONLIB_PLUGIN, JETSF, {"name": "jetsf", "args": ["nominal", "$0", "$1"]}, [pt, eta])
    )
    up = np.asarray(
        _eval(CORRECTIONLIB_PLUGIN, JETSF, {"name": "jetsf", "args": ["up", "$0", "$1"]}, [pt, eta])
    )
    assert np.allclose(up - nominal, 0.1)  # the constant ARGUMENT selected a different variation


def test_correctionlib_legacy_default_is_unchanged() -> None:
    # no params["args"] -> the original (systematic, inputs[0]) shape, exactly
    flat_sf = json.dumps(
        {
            "schema_version": 2,
            "corrections": [
                {
                    "name": "sf",
                    "version": 1,
                    "inputs": [{"name": "systematic", "type": "string"}, {"name": "x", "type": "real"}],
                    "output": {"name": "sf", "type": "real"},
                    "data": {
                        "nodetype": "category",
                        "input": "systematic",
                        "content": [{"key": "nominal", "value": 1.5}],
                    },
                }
            ],
        }
    ).encode()
    out = _eval(
        CORRECTIONLIB_PLUGIN, flat_sf, {"name": "sf", "systematic": "nominal"}, [np.array([1.0, 2.0])]
    )
    assert np.asarray(out).tolist() == [1.5, 1.5]


def test_correctionlib_jagged_replay_through_a_bundle(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from graphed_awkward import gak  # noqa: PLC0415

    events = make_events(n_events=800, seed=13)
    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", events)
    params = {"name": "jetsf", "args": ["nominal", "$0", "$1"]}
    sf = record_external(s, CORRECTIONLIB_PLUGIN, JETSF, [ev.Jet.pt, ev.Jet.eta], params=params)
    weight = gak.prod(sf, axis=1)  # per-event product of per-JET (jagged!) scale factors
    value = ev.MET.pt
    reference = _hist(s.materialize(value), s.materialize(weight))

    bundle = build_bundle(
        tmp_path / "b",
        session=s,
        value=value,
        weight=weight,
        datasets={"events": events},
        payloads={CORRECTIONLIB_PLUGIN.content_hash(JETSF): JETSF},
        histogram=HIST,
    )
    assert np.array_equal(np.asarray(reproduce(bundle), dtype="float64"), reference)
    # the call template is PRESERVED NODE CONTENT: it rode the IR into the bundle and replay obeyed it


# ---------------------------------- histogram: multi-axis + multi-weight ---------------------------
def test_histogram_multi_axis_fill_replays_through_a_bundle(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import boost_histogram as bh  # noqa: PLC0415
    import graphed_histogram as gh  # noqa: PLC0415
    from graphed_awkward import gak  # noqa: PLC0415

    events = ak.Array({"x": [[1.0, 4.0], [], [7.0, 2.5]] * 40, "y": [[0.1, 0.4], [], [0.7, 0.2]] * 40})
    s = Session(AwkwardBackend())
    g = from_awkward(s, "events", events)
    h = gh.boost.Histogram(bh.axis.Regular(4, 0, 8), bh.axis.Regular(4, 0, 1), storage=bh.storage.Int64())
    h.fill(gak.flatten(g.x, axis=1), gak.flatten(g.y, axis=1))

    eager = bh.Histogram(bh.axis.Regular(4, 0, 8), bh.axis.Regular(4, 0, 1), storage=bh.storage.Int64())
    eager.fill(ak.flatten(events.x, axis=None), ak.flatten(events.y, axis=None))

    bundle = build_bundle(
        tmp_path / "b", session=s, value=h.fill_nodes()[0], datasets={"events": events}, payloads={}
    )
    got = reproduce(bundle)
    assert np.array_equal(got.values(flow=True), eager.values(flow=True))


def test_histogram_multiple_weights_multiply_on_replay(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import boost_histogram as bh  # noqa: PLC0415
    import graphed_histogram as gh  # noqa: PLC0415

    events = ak.Array(
        {"x": [1.0, 4.0, 7.0, 2.5] * 30, "w1": [0.5, 1.0, 2.0, 1.5] * 30, "w2": [1.0, 0.5, 1.0, 2.0] * 30}
    )
    s = Session(AwkwardBackend())
    g = from_awkward(s, "events", events)

    spec = gh.spec_of(bh.Histogram(bh.axis.Regular(4, 0, 8), storage=bh.storage.Weight()))
    params = {"spec": spec, "n_axes": 1, "weighted": True, "n_weights": 2, "sampled": False}
    fill = record_external(s, HISTOGRAM_PLUGIN, spec.encode(), [g.x, g.w1, g.w2], params=params)

    eager = bh.Histogram(bh.axis.Regular(4, 0, 8), storage=bh.storage.Weight())
    eager.fill(np.asarray(events.x), weight=np.asarray(events.w1) * np.asarray(events.w2))

    bundle = build_bundle(tmp_path / "b", session=s, value=fill, datasets={"events": events}, payloads={})
    got = reproduce(bundle)
    assert np.array_equal(got.view(flow=True)["value"], eager.view(flow=True)["value"])
    assert np.array_equal(got.view(flow=True)["variance"], eager.view(flow=True)["variance"])


# ---------------------------------- Triton: multiple NAMED inputs ----------------------------------
DESCRIPTOR = json.dumps(
    {"model": "scorer2", "version": "1", "weights": {"kin": [0.5, 0.25], "mask": [2.0]}}
).encode()
PARAMS = {
    "url": "triton://multi:8000",
    "model": "scorer2",
    "transport": "fake_triton_multi:transport",
    "args": {"kin": ["$0", "$1"], "mask": ["$2"]},
    "output_name": "y",
}


def test_triton_multiple_named_inputs_replay_through_a_bundle(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from graphed_awkward import gak  # noqa: PLC0415

    client = fake_triton_multi.serve(PARAMS["url"], DESCRIPTOR)
    events = make_events(n_events=600, seed=7)
    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", events)
    njet = gak.num(ev.Jet, axis=1)
    ht = gak.sum(ev.Jet.pt, axis=1)
    nmu = gak.num(ev.Muon, axis=1)
    weight = record_external(s, TRITON_PLUGIN, DESCRIPTOR, [njet, ht, nmu], params=PARAMS)
    value = ev.MET.pt
    reference = _hist(s.materialize(value), s.materialize(weight))

    bundle = build_bundle(
        tmp_path / "b",
        session=s,
        value=value,
        weight=weight,
        datasets={"events": events},
        payloads={TRITON_PLUGIN.content_hash(DESCRIPTOR): DESCRIPTOR},
        histogram=HIST,
    )
    assert np.array_equal(np.asarray(reproduce(bundle), dtype="float64"), reference)
    # every named input genuinely reached the server, on BOTH executions (materialize + reproduce)
    assert client.seen_input_names and all(seen == ["kin", "mask"] for seen in client.seen_input_names)


def test_triton_legacy_default_still_builds_one_input() -> None:
    # no args -> the m26 single-input convention, unchanged
    legacy_desc = b'{"model": "scorer", "version": "1", "weights": {"x": [0.45]}}'
    url = "triton://legacy:8000"
    fake_triton_multi.serve(url, legacy_desc)
    params = {
        "url": url,
        "model": "scorer",
        "transport": "fake_triton_multi:transport",
        "input_name": "x",
        "output_name": "y",
    }
    out = _eval(TRITON_PLUGIN, legacy_desc, params, [np.array([0.0, 1.0])])
    want = 1.0 / (1.0 + np.exp(-(0.45 * np.array([0.0, 1.0]))))
    assert np.allclose(np.asarray(out), want, atol=1e-6)


# ---------------------------------- xgboost: tabular-only, loudly ----------------------------------
def test_xgboost_rejects_multi_argument_templates_loudly() -> None:
    with pytest.raises(PreserveError, match="single"):
        XGBOOST_PLUGIN.evaluate(None, {"args": [["$0"], ["$1"]]}, [np.array([1.0]), np.array([2.0])])


def test_correctionlib_rejects_keyword_arguments_loudly() -> None:
    # correctionlib's evaluate is positional-only (a C++ binding): kwargs are meaningless and
    # must never be silently dropped
    with pytest.raises(PreserveError, match="keyword"):
        _eval(
            CORRECTIONLIB_PLUGIN,
            JETSF,
            {"name": "jetsf", "args": ["nominal", "$0"], "kwargs": {"eta": ["$1"]}},
            [np.array([1.0]), np.array([0.5])],
        )
