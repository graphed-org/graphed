"""External payload **plugins** — the extensible, validated mechanism for preservable Externals (M9).

An :class:`ExternalPlugin` says, for one ``kind`` of external payload, three things:

1. ``content_hash(payload_bytes) -> str`` — a **deterministic, content-based** hash of the payload.
   For an ONNX model it is the hash of the *weights* (+ graph structure), for a correctionlib set the
   hash of its *contents* — not the raw file bytes (which carry incidental formatting/metadata).
2. ``evaluate(payload_bytes, params, inputs) -> value`` — how to actually run the payload on inputs.
3. ``samples() -> Sequence[bytes]`` — ≥2 distinct example payloads used to *validate the hash*.

``register_plugin`` validates a plugin's hash before trusting it (the user's explicit requirement):
it must be **deterministic across processes** (so it reproduces on machine B — this catches a hash
built from ``hash()``/``id()``/time/randomness) and **non-vacuous** (distinct payloads must not
collide — this catches a constant or trivially-weak hash). ``onnx`` and ``correctionlib`` ship as
plugins and double as templates for users writing their own.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import PreserveError

ContentHash = Callable[[bytes], str]
Evaluate = Callable[[bytes, Mapping[str, Any], list[Any]], Any]


@dataclass(frozen=True)
class ExternalPlugin:
    """How to hash, evaluate, and (self-)validate one ``kind`` of External payload."""

    kind: str
    content_hash: ContentHash
    evaluate: Evaluate
    samples: Callable[[], Sequence[bytes]]
    framework: str = ""


_REGISTRY: dict[str, ExternalPlugin] = {}


def register_plugin(plugin: ExternalPlugin, *, validate: bool = True) -> ExternalPlugin:
    """Register ``plugin`` under its ``kind``. By default its ``content_hash`` is validated first
    (deterministic across processes + non-vacuous); pass ``validate=False`` only for a plugin whose
    hash is already proven (the built-ins are validated by the test suite)."""
    if validate:
        validate_plugin(plugin)
    _REGISTRY[plugin.kind] = plugin
    return plugin


def get_plugin(kind: str) -> ExternalPlugin | None:
    return _REGISTRY.get(kind)


def registered_kinds() -> list[str]:
    return sorted(_REGISTRY)


# ---- hash validation ----------------------------------------------------------------------------
def validate_plugin(plugin: ExternalPlugin) -> None:
    """Reject a plugin whose ``content_hash`` is non-deterministic or vacuous (plan M9 requirement)."""
    samples = list(plugin.samples())
    if len(samples) < 2:
        raise PreserveError(
            f"plugin {plugin.kind!r}: samples() must return >=2 distinct payloads to validate the hash"
        )
    if len(set(samples)) != len(samples):
        raise PreserveError(f"plugin {plugin.kind!r}: samples() must be distinct")

    first = plugin.content_hash(samples[0])
    if not isinstance(first, str) or not first:
        raise PreserveError(f"plugin {plugin.kind!r}: content_hash must return a non-empty string")
    if plugin.content_hash(samples[0]) != first:
        raise PreserveError(f"plugin {plugin.kind!r}: content_hash is not deterministic within a process")

    # non-vacuity: distinct payloads must not collide onto one hash (catches a constant/trivial hash)
    hashes = [plugin.content_hash(s) for s in samples]
    if len(set(hashes)) != len(samples):
        raise PreserveError(
            f"plugin {plugin.kind!r}: content_hash is vacuous — distinct payloads share a hash ({hashes})"
        )

    # cross-process determinism: the hash must reproduce on another machine/process. Running under two
    # different PYTHONHASHSEEDs catches a hash derived from builtin hash()/id()/time/randomness.
    a = _hash_in_subprocess(plugin.content_hash, samples[0], seed="0")
    b = _hash_in_subprocess(plugin.content_hash, samples[0], seed="1")
    if not (a == b == first):
        raise PreserveError(
            f"plugin {plugin.kind!r}: content_hash is not deterministic across processes "
            f"(in-process {first!r} vs subprocess {a!r}/{b!r}) — it must depend only on payload content, "
            f"never on hash()/id()/time/randomness"
        )


def _hash_in_subprocess(content_hash: ContentHash, sample: bytes, *, seed: str) -> str:
    import cloudpickle  # noqa: PLC0415

    # serialize the hash fn BY VALUE (embed its code) so the subprocess needs no access to the user's
    # module — exactly the source-free condition the bundle must reproduce under on machine B.
    module = sys.modules.get(getattr(content_hash, "__module__", "") or "")
    registered = False
    if module is not None:
        try:
            cloudpickle.register_pickle_by_value(module)
            registered = True
        except Exception:
            registered = False
    try:
        blob = cloudpickle.dumps((content_hash, sample))
    finally:
        if registered and module is not None:
            cloudpickle.unregister_pickle_by_value(module)

    code = (
        "import sys, cloudpickle\n"
        "fn, s = cloudpickle.loads(sys.stdin.buffer.read())\n"
        "sys.stdout.write(fn(s))\n"
    )
    env = {**os.environ, "PYTHONHASHSEED": seed}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        input=blob,
        capture_output=True,
        env=env,
    )
    if proc.returncode != 0:
        raise PreserveError(f"content_hash failed in a subprocess: {proc.stderr.decode(errors='replace')}")
    out: str = proc.stdout.decode()
    return out


def evaluate_external(node: Mapping[str, Any], inputs: list[Any], payload: bytes) -> Any:
    """Evaluate an External node via its registered plugin (dispatch by descriptor kind)."""
    kind = node["descriptor"]["kind"]
    plugin = get_plugin(kind)
    if plugin is None:
        raise PreserveError(f"no plugin registered for external kind {kind!r} (a preservation risk)")
    return plugin.evaluate(payload, node["params"], inputs)


def record_external(
    session: Any,
    plugin: ExternalPlugin,
    payload: bytes,
    inputs: Sequence[Any],
    *,
    params: Mapping[str, Any] | None = None,
) -> Any:
    """Record a preservable External in a graphed ``session`` using ``plugin``.

    The node's descriptor carries the plugin's **deterministic content hash** of ``payload`` (so it is
    preserved, not opaque). The build-time eval and the M9 reproduce-time eval both go through
    ``plugin.evaluate`` on the same payload bytes, so a bundle reproduces bit-for-bit. This is the
    entry point users follow to add their own External kinds."""
    content_hash = plugin.content_hash(payload)
    node_params: dict[str, Any] = {
        "kind": plugin.kind,
        "content_hash": content_hash,
        "framework": plugin.framework,
        **(params or {}),
    }

    def _fn(*values: Any) -> Any:
        return plugin.evaluate(payload, node_params, list(values))

    return session.record_external("external", _fn, list(inputs), node_params)


# ---- a trivial template: hash == sha256 of the raw payload bytes --------------------------------
def sha256_bytes(payload: bytes) -> str:
    """The simplest content hash: SHA-256 of the raw bytes. A fine template when the payload bytes
    ARE the canonical content (no incidental formatting/metadata to normalize away)."""
    return "sha256:" + hashlib.sha256(payload).hexdigest()


# ============================ built-in plugins (and templates) ===================================
# ---- correctionlib: hash of CONTENTS (canonical JSON), not file bytes ---------------------------
def correctionlib_content_hash(payload: bytes) -> str:
    import json  # noqa: PLC0415

    data = json.loads(payload)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(b"correctionlib-contents-v1" + canonical).hexdigest()


def eval_correctionlib(payload: bytes, params: Mapping[str, Any], inputs: list[Any]) -> Any:
    import awkward as ak  # noqa: PLC0415
    import correctionlib  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    cset = correctionlib.CorrectionSet.from_string(payload.decode("utf-8"))
    name = str(params.get("name", ""))
    systematic = str(params.get("systematic", "nominal"))
    x = np.asarray(ak.to_numpy(ak.Array(inputs[0])), dtype="float64")
    return ak.Array(np.asarray(cset[name].evaluate(systematic, x), dtype="float64"))


def _correctionlib_samples() -> list[bytes]:
    def _cset(sf: float) -> bytes:
        import json  # noqa: PLC0415

        doc = {
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
                        "content": [{"key": "nominal", "value": sf}],
                    },
                }
            ],
        }
        return json.dumps(doc).encode("utf-8")

    return [_cset(1.0), _cset(1.5)]


# ---- ONNX: hash of WEIGHTS (+ graph op structure), not file bytes -------------------------------
def onnx_content_hash(payload: bytes) -> str:
    import onnx  # noqa: PLC0415
    from onnx import numpy_helper  # noqa: PLC0415

    model = onnx.load_from_string(payload)
    h = hashlib.sha256()
    h.update(b"onnx-weights-v1")
    for init in sorted(model.graph.initializer, key=lambda t: t.name):
        h.update(init.name.encode("utf-8"))
        h.update(numpy_helper.to_array(init).tobytes())
    for node in model.graph.node:  # structure too: same weights, different graph -> different hash
        h.update(node.op_type.encode("utf-8"))
    return "sha256:" + h.hexdigest()


def eval_onnx(payload: bytes, params: Mapping[str, Any], inputs: list[Any]) -> Any:
    import awkward as ak  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415

    sess = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
    name = str(params.get("input_name", "")) or sess.get_inputs()[0].name
    x = np.asarray(ak.to_numpy(ak.Array(inputs[0])), dtype="float32").reshape(-1, 1)
    out = sess.run(None, {name: x})[0].reshape(-1)
    return ak.Array(np.asarray(out, dtype="float64"))


def _onnx_samples() -> list[bytes]:
    import numpy as np  # noqa: PLC0415
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    def _model(weight: float) -> bytes:
        w = numpy_helper.from_array(np.array([[weight]], dtype=np.float32), name="W")
        b = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="B")
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 1])
        graph = helper.make_graph(
            [helper.make_node("Gemm", ["x", "W", "B"], ["y"])], "m", [x], [y], initializer=[w, b]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
        return model.SerializeToString()  # type: ignore[no-any-return]

    return [_model(0.5), _model(0.9)]


CORRECTIONLIB_PLUGIN = ExternalPlugin(
    kind="correctionlib",
    content_hash=correctionlib_content_hash,
    evaluate=eval_correctionlib,
    samples=_correctionlib_samples,
    framework="correctionlib",
)
ONNX_PLUGIN = ExternalPlugin(
    kind="onnx_model",
    content_hash=onnx_content_hash,
    evaluate=eval_onnx,
    samples=_onnx_samples,
    framework="onnxruntime",
)

# register the built-ins (validated by the M9 test suite, so skip the per-import subprocess check)
register_plugin(CORRECTIONLIB_PLUGIN, validate=False)
register_plugin(ONNX_PLUGIN, validate=False)
