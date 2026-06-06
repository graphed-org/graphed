"""The AGC-ttbar-style fixture for the M9 preservation tests.

A reduced AGC ttbar slice recorded through the graphed frontend, deliberately exercising the three
things the plan chose this analysis for: an **ONNX model**, **correctionlib scale factors**, and
**systematics** (JES kinematic + correctionlib weight up/down). Real correctionlib JSON and a real
ONNX model are generated on disk so their External nodes content-hash genuine HEP-standard payloads.

The External eval callables are exactly ``graphed_preserve.externals`` — the same code ``reproduce``
uses — so an in-process ``materialize`` and a from-bundle ``reproduce`` agree bit-for-bit.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from graphed import Session
from graphed_awkward import AwkwardBackend, from_awkward
from graphed_corpus import make_events

from graphed_preserve import build_bundle
from graphed_preserve.externals import eval_correctionlib, eval_onnx

N_EVENTS = 2500
SEED = 2026
HIST = {"name": "ht", "bins": 30, "lo": 0.0, "hi": 800.0}

_SF_EDGES = [0.0, 2.0, 4.0, 6.0, 100.0]
_SF_CONTENT = {
    "nominal": [0.90, 0.95, 1.00, 1.05],
    "up": [0.93, 0.98, 1.03, 1.08],
    "down": [0.87, 0.92, 0.97, 1.02],
}


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ---- real HEP-standard payloads -----------------------------------------------------------------
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
                "inputs": [
                    {"name": "systematic", "type": "string"},
                    {"name": "x", "type": "real"},
                ],
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
    return model.SerializeToString()


def write_payloads(dirpath: Path, *, sf_scale: float = 1.0, onnx_weight: float = 0.04) -> dict[str, bytes]:
    """Write the correction + model files into ``dirpath`` (fixed basenames) and return the
    ``content_hash -> bytes`` map build_bundle stores."""
    dirpath.mkdir(parents=True, exist_ok=True)
    corr = correctionlib_json(scale=sf_scale)
    model = onnx_model(weight=onnx_weight)
    (dirpath / "event_sf.json").write_bytes(corr)
    (dirpath / "score.onnx").write_bytes(model)
    return {_sha256(corr): corr, _sha256(model): model}


# ---- the recorded analysis ----------------------------------------------------------------------
def record(events: Any, payload_dir: Path, *, config: dict[str, Any]) -> tuple[Session, Any, Any]:
    """Record the AGC slice; return (session, value=HT[sel], weight=(SF*score)[sel])."""
    corr_bytes = (payload_dir / "event_sf.json").read_bytes()
    model_bytes = (payload_dir / "score.onnx").read_bytes()
    syst = str(config.get("systematic", "nominal"))
    jes = float(config.get("jes_factor", 1.0))
    region_min_b = int(config.get("min_btag", 1))

    s = Session(AwkwardBackend())
    # External eval callables = the SAME code reproduce uses (so materialize == reproduce)
    corr_fn = lambda *vals: eval_correctionlib(corr_bytes, name="event_sf", systematic=syst, x=vals[0])  # noqa: E731
    onnx_fn = lambda *vals: eval_onnx(model_bytes, input_name="x", x=vals[0])  # noqa: E731

    with contextlib.chdir(payload_dir):  # record External paths as stable basenames (no abs-path leak)
        ev = from_awkward(s, "events", events)
        spt = ev.Jet.pt * jes  # JES kinematic variation -> changes the selection below
        jetmask = spt > 25.0
        ht = gak_sum(spt[jetmask])
        njet = gak_num(ev.Jet[jetmask])
        nb = gak_sum((ev.Jet.btag[jetmask] > 0.7) * 1)
        sel = (njet >= 4) & (nb >= region_min_b)

        sf = s.record_external(
            "correction", corr_fn, [njet], {"path": "event_sf.json", "name": "event_sf", "systematic": syst}
        )
        score = s.record_external("onnx", onnx_fn, [ht], {"path": "score.onnx", "input_name": "x"})
        weight = sf * score
        value = ht[sel]
        wgt = weight[sel]
    return s, value, wgt


def gak_sum(array: Any) -> Any:
    from graphed_awkward import gak  # noqa: PLC0415

    return gak.sum(array, axis=1)


def gak_num(array: Any) -> Any:
    from graphed_awkward import gak  # noqa: PLC0415

    return gak.num(array, axis=1)


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


_DEFAULT_CONFIG = {"systematic": "nominal", "jes_factor": 1.0, "min_btag": 1}


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
    pdir = Path(root) / "payloads"
    payloads = write_payloads(pdir, sf_scale=sf_scale, onnx_weight=onnx_weight)
    events = make_events_for(seed)
    session, value, weight = record(events, pdir, config=cfg)
    reference = materialize_reference(session, value, weight)
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
    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", make_events_for())
    ht = gak_sum(ev.Jet.pt)
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
