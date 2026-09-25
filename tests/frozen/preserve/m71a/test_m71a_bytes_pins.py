"""m71a A-C6: a plan that declares no output type keeps graphed 0.0.6's bytes (IR, compiled IR,
DurablePlan, and the bundle manifest's externals/opaque_nodes/analysis subtrees).

The pins were measured on 3c46e01 (graphed 0.0.6). Descriptors embed `platform.python_version()`,
so it is fixed here; payloads are byte literals so no sample generator can drift them.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.core import DurablePlan, OpSpec
from graphed.execute import compile_ir
from graphed.numpy import NumpyBackend, from_array
from graphed.numpy.gufunc import apply_gufunc
from graphed.preserve import build_bundle
from graphed.preserve.externals import CORRECTIONLIB_PLUGIN, ONNX_PLUGIN, record_external

CSET = (
    b'{"schema_version": 2, "corrections": [{"name": "sf", "version": 1, "inputs": [{"name": '
    b'"systematic", "type": "string"}, {"name": "x", "type": "real"}], "output": {"name": "sf", '
    b'"type": "real"}, "data": {"nodetype": "category", "input": "systematic", "content": [{"key": '
    b'"nominal", "value": 1.0}]}}]}'
)
MODEL = (
    b'\x08\t:]\n\x12\n\x01x\n\x01W\n\x01B\x12\x01y"\x04Gemm\x12\x01m*\x0f\x08\x01\x08\x01\x10\x01B'
    b"\x01WJ\x04\x00\x00\x00?*\r\x08\x01\x10\x01B\x01BJ\x04\x00\x00\x00\x00Z\x11\n\x01x\x12\x0c\n\n"
    b"\x08\x01\x12\x06\n\x00\n\x02\x08\x01b\x11\n\x01y\x12\x0c\n\n\x08\x01\x12\x06\n\x00\n\x02\x08"
    b"\x01B\x04\n\x00\x10\r"
)

PINS = {
    "awk.ir.opt0": "c56cee7a3664ab8c79a24534aaf12b26c8ccefe3fadc4d918dd0358996d68dd7",
    "awk.ir.opt1": "86fb5cfd13c819409f5657a0ba8ced3a41056fc8138249c0d4f38bb4f5a4a9c9",
    "awk.compile_ir": "86fb5cfd13c819409f5657a0ba8ced3a41056fc8138249c0d4f38bb4f5a4a9c9",
    "awk.durableplan": "b5d2bfff4008f53112c2bf587da5a023dc4204af3bdd9b30bc337037dbe619fc",
    "awk.manifest.externals": "2f754dd342a2142201dae90be5c65bdbaa76e343bd4cca2e094fe71bb871ed57",
    "awk.manifest.opaque_nodes": "fe94b827f0455bee6f4a6a265286f827f18748ede569565ee827058085bad146",
    "awk.manifest.analysis": "a5626dfc6038ed85150255504dc127f4627f276abe56f93e5467499b1af16fb7",
    "np.ir.opt0": "6b94f6ac5ce8f1f1eade6575535130b7eea0d72b30bdcb7effae12924b358a0b",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(tree: Any) -> bytes:
    return json.dumps(tree, sort_keys=True, separators=(",", ":")).encode()


def _events() -> ak.Array:
    return ak.zip(
        {
            "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "run": np.array([1, 2, 1], dtype=np.uint32),
        }
    )


def _pins(root: Path) -> dict[str, str]:
    s = Session(AwkwardBackend())
    ev = from_awkward(s, "ev", _events())
    sf = record_external(
        s, CORRECTIONLIB_PLUGIN, CSET, [ev.x], params={"name": "sf", "args": ["nominal", "$0"]}
    )
    nn = gak.onnx_inference(MODEL, [ev.x], lambda m: m[:, 0], args=[["$0"]])
    tc = gak.apply_correction(CSET, "sf", [ev.x], lambda *a: a, args=["nominal", "$0"])
    mp = ev.run.map(lambda r: r == 1, name="golden")
    out = gak.sum((sf * nn * tc)[mp == 1])

    compiled = compile_ir(s, out).ir
    spec = OpSpec.from_ref("json:dumps")
    pins = {
        "awk.ir.opt0": _sha(s.serialized_ir(out, optimize=False)),
        "awk.ir.opt1": _sha(s.serialized_ir(out)),
        "awk.compile_ir": _sha(compiled),
        "awk.durableplan": _sha(DurablePlan(ir=compiled, process=spec, combine=spec, empty=spec).to_bytes()),
    }
    bundle = build_bundle(
        root,
        session=s,
        value=out,
        payloads={
            CORRECTIONLIB_PLUGIN.content_hash(CSET): CSET,
            ONNX_PLUGIN.content_hash(MODEL): MODEL,
        },
        environment={"pinned": True},
        datasets={"ev": _events()},
    )
    manifest = json.loads((bundle.root / "manifest.json").read_bytes())
    for key in ("externals", "opaque_nodes", "analysis"):
        pins[f"awk.manifest.{key}"] = _sha(_canonical(manifest[key]))

    sn = Session(NumpyBackend())
    x = from_array(sn, "x", np.array([1.0, 2.0], dtype=np.float32))
    g = apply_gufunc(lambda a: a > 1, "()->()", x, output_dtype=bool, name="g")
    pins["np.ir.opt0"] = _sha(sn.serialized_ir(g, x.map(lambda a: a, name="id"), optimize=False))
    return pins


def test_an_undeclared_plan_keeps_its_0_0_6_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(platform, "python_version", lambda: "3.13.3")
    assert _pins(tmp_path / "a") == PINS
