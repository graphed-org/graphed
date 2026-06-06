"""The AGC-ttbar-style fixture for the M9 preservation tests.

A reduced AGC ttbar slice recorded through the graphed frontend, exercising the three things the plan
chose this analysis for: an **ONNX model**, **correctionlib scale factors**, and **systematics** (JES
kinematic + correctionlib weight up/down). The correction + model are real HEP-standard payloads held
as in-memory bytes and recorded via the M9 **External plugins** (``record_external`` with the
correctionlib / ONNX plugins), so each External node carries the plugin's deterministic *content*
hash (correctionlib contents, ONNX weights) — no file paths leak into the IR.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from graphed import Session
from graphed_awkward import AwkwardBackend, from_awkward
from graphed_corpus import make_events

from graphed_preserve import CORRECTIONLIB_PLUGIN, ONNX_PLUGIN, build_bundle, record_external

N_EVENTS = 2500
SEED = 2026
HIST = {"name": "ht", "bins": 30, "lo": 0.0, "hi": 800.0}
_DEFAULT_CONFIG = {"systematic": "nominal", "jes_factor": 1.0, "min_btag": 1}

_SF_EDGES = [0.0, 2.0, 4.0, 6.0, 100.0]
_SF_CONTENT = {
    "nominal": [0.90, 0.95, 1.00, 1.05],
    "up": [0.93, 0.98, 1.03, 1.08],
    "down": [0.87, 0.92, 0.97, 1.02],
}


# ---- real HEP-standard payloads (as bytes; no files) --------------------------------------------
def correctionlib_json(*, scale: float = 1.0) -> bytes:
    """A valid correctionlib v2 set: an event-level SF binned in njet, with a systematic category."""
    content = {
        syst: {
            "nodetype": "binning",
            "input": "x",
            "edges": _SF_EDGES,
            "content": [round(c * scale, 6) for c in vals],
            "flow": "clamp",
        }
        for syst, vals in _SF_CONTENT.items()
    }
    cset = {
        "schema_version": 2,
        "corrections": [
            {
                "name": "event_sf",
                "version": 1,
                "inputs": [{"name": "systematic", "type": "string"}, {"name": "x", "type": "real"}],
                "output": {"name": "sf", "type": "real"},
                "data": {
                    "nodetype": "category",
                    "input": "systematic",
                    "content": [{"key": k, "value": v} for k, v in content.items()],
                },
            }
        ],
    }
    return json.dumps(cset, sort_keys=True).encode("utf-8")


def onnx_model(*, weight: float = 0.04, bias: float = -0.5) -> bytes:
    """A tiny real ONNX model: score = Sigmoid(weight * x + bias), one float feature per event."""
    import onnx  # noqa: PLC0415
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    w = numpy_helper.from_array(np.array([[weight]], dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.array([bias], dtype=np.float32), name="B")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 1])
    score = helper.make_tensor_value_info("score", TensorProto.FLOAT, [None, 1])
    graph = helper.make_graph(
        [
            helper.make_node("Gemm", ["x", "W", "B"], ["t"]),
            helper.make_node("Sigmoid", ["t"], ["score"]),
        ],
        "score_model",
        [x],
        [score],
        initializer=[w, b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    onnx.checker.check_model(model)
    return model.SerializeToString()  # type: ignore[no-any-return]


# ---- the recorded analysis (externals via plugins) ----------------------------------------------
def record(
    events: Any, *, config: dict[str, Any], corr_bytes: bytes, model_bytes: bytes
) -> tuple[Session, Any, Any]:
    """Record the AGC slice; return (session, value=HT[sel], weight=(SF*score)[sel])."""
    from graphed_awkward import gak  # noqa: PLC0415

    syst = str(config.get("systematic", "nominal"))
    jes = float(config.get("jes_factor", 1.0))
    region_min_b = int(config.get("min_btag", 1))

    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", events)
    spt = ev.Jet.pt * jes  # JES kinematic variation -> changes the selection below
    jetmask = spt > 25.0
    ht = gak.sum(spt[jetmask], axis=1)
    njet = gak.num(ev.Jet[jetmask], axis=1)
    nb = gak.sum((ev.Jet.btag[jetmask] > 0.7) * 1, axis=1)
    sel = (njet >= 4) & (nb >= region_min_b)

    sf = record_external(
        s, CORRECTIONLIB_PLUGIN, corr_bytes, [njet], params={"name": "event_sf", "systematic": syst}
    )
    score = record_external(s, ONNX_PLUGIN, model_bytes, [ht], params={"input_name": "x"})
    weight = sf * score
    return s, ht[sel], weight[sel]


def histogram(values: Any, weights: Any) -> np.ndarray:
    import awkward as ak  # noqa: PLC0415

    v = np.asarray(ak.to_numpy(ak.Array(values)), dtype="float64")
    w = np.asarray(ak.to_numpy(ak.Array(weights)), dtype="float64")
    counts, _ = np.histogram(v, bins=HIST["bins"], range=(HIST["lo"], HIST["hi"]), weights=w)
    return np.round(counts, 6)


def materialize_reference(session: Session, value: Any, weight: Any) -> np.ndarray:
    """The build-time histogram, computed in-process via materialize (originals present)."""
    return histogram(session.materialize(value), session.materialize(weight))


def make_events_for(seed: int = SEED) -> Any:
    return make_events(n_events=N_EVENTS, seed=seed)


def build_agc(
    root: Path,
    *,
    config: dict[str, Any] | None = None,
    sf_scale: float = 1.0,
    onnx_weight: float = 0.04,
    seed: int = SEED,
) -> tuple[Any, np.ndarray]:
    """Build a full AGC preservation bundle under ``root``; return (bundle, build-time reference)."""
    cfg = dict(config or _DEFAULT_CONFIG)
    corr_bytes = correctionlib_json(scale=sf_scale)
    model_bytes = onnx_model(weight=onnx_weight)
    events = make_events_for(seed)
    session, value, weight = record(events, config=cfg, corr_bytes=corr_bytes, model_bytes=model_bytes)
    reference = materialize_reference(session, value, weight)
    payloads = {
        CORRECTIONLIB_PLUGIN.content_hash(corr_bytes): corr_bytes,
        ONNX_PLUGIN.content_hash(model_bytes): model_bytes,
    }
    bundle = build_bundle(
        Path(root) / "bundle",
        session=session,
        value=value,
        weight=weight,
        datasets={"events": events},
        payloads=payloads,
        histogram=HIST,
        config=cfg,
        seed=seed,
    )
    return bundle, reference


def build_opaque(root: Path) -> Any:
    """A bundle whose weight is an opaque (cloudpickled) ``map`` node — a preservation risk, for the
    inspect risk-flagging test."""
    from graphed_awkward import gak  # noqa: PLC0415

    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", make_events_for())
    ht = gak.sum(ev.Jet.pt, axis=1)
    weight = ht.map(lambda a: a)  # opaque External -> flagged, not preserved as durable IR
    return build_bundle(
        Path(root) / "bundle",
        session=s,
        value=ht,
        weight=weight,
        datasets={"events": make_events_for()},
        payloads={},
        histogram=HIST,
        config={},
        seed=SEED,
    )
