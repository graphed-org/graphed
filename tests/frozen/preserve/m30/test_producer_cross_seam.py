"""M30 — the producer seams meet preservation: gak- and graphed-histogram-recorded Externals
preserve, pass bundle integrity, and replay bit-for-bit.

M27 widened replay; M28 (graphed-awkward) and M29 (graphed-histogram) aligned the producers.
This suite is the cross-repo acceptance: nodes recorded through the REAL user-facing surfaces —
``gak.apply_correction``/``gak.onnx_inference`` with call templates, ``Histogram.fill`` with
multiple weights — flow through ``build_bundle``'s payload-integrity check (the M3-era
raw-bytes-vs-content-identity divergence made that impossible; pinned below) and ``reproduce``
returns exactly what record-time ``materialize`` produced.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import awkward as ak
import numpy as np
import pytest
from graphed import Session
from graphed_awkward import AwkwardBackend, from_awkward, gak
from graphed_awkward.payloads import correctionlib_contents_hash, onnx_weights_hash

from graphed_preserve import PreserveError, build_bundle, reproduce

HIST = {"name": "met", "bins": 16, "lo": 0.0, "hi": 120.0}

CSET = json.dumps(
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
                        }
                    ],
                },
            }
        ],
    }
).encode()

EVENTS = ak.Array(
    {
        "Jet": [[{"pt": 30.0, "eta": 0.5}, {"pt": 50.0, "eta": 1.0}], [], [{"pt": 80.0, "eta": 2.0}]] * 60,
        "MET": [12.0, 45.0, 8.0] * 60,
        "w1": [0.5, 1.0, 2.0] * 60,
        "w2": [1.0, 0.5, 1.5] * 60,
    }
)


def _hist(value: Any, weight: Any) -> np.ndarray:
    v = np.asarray(ak.to_numpy(ak.Array(value)), dtype="float64")
    w = np.asarray(ak.to_numpy(ak.Array(weight)), dtype="float64")
    return np.histogram(v, bins=HIST["bins"], range=(HIST["lo"], HIST["hi"]), weights=w)[0].round(6)


def test_gak_recorded_correction_passes_integrity_and_replays(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import correctionlib  # noqa: PLC0415

    cset = correctionlib.CorrectionSet.from_string(CSET.decode())
    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", EVENTS)

    # the REAL user surface: jagged multi-input correction with a positional systematic constant
    sf = gak.apply_correction(
        CSET, "jetsf", [ev.Jet.pt, ev.Jet.eta], cset["jetsf"].evaluate, args=["nominal", "$0", "$1"]
    )
    weight = gak.prod(sf, axis=1)
    value = ev.MET
    reference = _hist(s.materialize(value), s.materialize(weight))

    bundle = build_bundle(
        tmp_path / "b",
        session=s,
        value=value,
        weight=weight,
        datasets={"events": EVENTS},
        payloads={correctionlib_contents_hash(CSET): CSET},  # keyed by the ALIGNED identity
        histogram=HIST,
    )
    (entry,) = bundle.manifest["externals"]
    assert entry["kind"] == "correctionlib"
    assert entry["content_hash"] == correctionlib_contents_hash(CSET)  # one identity, both seams
    assert np.array_equal(np.asarray(reproduce(bundle), dtype="float64"), reference)


def test_gak_recorded_onnx_passes_integrity_and_replays(tmp_path) -> None:  # type: ignore[no-untyped-def]
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    w = numpy_helper.from_array(np.array([[0.5], [0.25]], dtype=np.float32), name="W")
    kin = helper.make_tensor_value_info("kin", TensorProto.FLOAT, [None, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 1])
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["kin", "W"], ["y"])], "m", [kin], [y], initializer=[w]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    onnx.checker.check_model(model)
    payload = model.SerializeToString()

    session_ort = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])

    def runner(x: Any) -> Any:
        out = session_ort.run(None, {"kin": np.asarray(x, dtype="float32")})[0].reshape(-1)
        return ak.Array(np.asarray(out, dtype="float64"))

    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", EVENTS)
    njet = gak.num(ev.Jet, axis=1)
    ht = gak.sum(ev.Jet.pt, axis=1)
    score = gak.onnx_inference(payload, [njet, ht], runner, args=[["$0", "$1"]])
    reference = _hist(s.materialize(ev.MET), s.materialize(score))

    bundle = build_bundle(
        tmp_path / "b",
        session=s,
        value=ev.MET,
        weight=score,
        datasets={"events": EVENTS},
        payloads={onnx_weights_hash(payload): payload},
        histogram=HIST,
    )
    (entry,) = bundle.manifest["externals"]
    assert entry["kind"] == "onnx_model"
    assert entry["content_hash"] == onnx_weights_hash(payload)
    assert np.array_equal(np.asarray(reproduce(bundle), dtype="float64"), reference)


def test_gh_multi_weight_fill_replays_through_a_bundle(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import boost_histogram as bh  # noqa: PLC0415
    import graphed_histogram as gh  # noqa: PLC0415

    s = Session(AwkwardBackend())
    g = from_awkward(s, "events", EVENTS)
    h = gh.boost.Histogram(bh.axis.Regular(8, 0.0, 60.0), storage=bh.storage.Weight())
    h.fill(g.MET, weight=[g.w1, g.w2])  # the M29 producer surface

    reference = s.materialize(h.fill_nodes()[0])
    bundle = build_bundle(
        tmp_path / "b", session=s, value=h.fill_nodes()[0], datasets={"events": EVENTS}, payloads={}
    )
    got = reproduce(bundle)
    assert np.array_equal(got.view(flow=True)["value"], reference.view(flow=True)["value"])
    assert np.array_equal(got.view(flow=True)["variance"], reference.view(flow=True)["variance"])


def test_the_legacy_m3_recording_still_cannot_bundle_and_says_why(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # the M3 path hashes raw file bytes — a DIFFERENT identity than the plugin's. The bundle
    # build must reject the mismatch loudly (cache-poisoning-safe), never store under two ids.
    p = tmp_path / "cset.json"
    p.write_bytes(CSET)
    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", EVENTS)
    sf = gak.apply_correction(str(p), "jetsf", [ev.Jet.pt], evaluator=lambda x: x * 0.0 + 1.5)
    raw_hash = "sha256:" + hashlib.sha256(CSET).hexdigest()
    with pytest.raises(PreserveError, match="hashes to"):
        build_bundle(
            tmp_path / "b",
            session=s,
            value=ev.MET,
            weight=gak.prod(sf, axis=1),
            datasets={"events": EVENTS},
            payloads={raw_hash: CSET},
            histogram=HIST,
        )
