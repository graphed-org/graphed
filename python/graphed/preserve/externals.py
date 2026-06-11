"""External payload **plugins** — the extensible, validated mechanism for preservable Externals (M9).

An :class:`ExternalPlugin` says, for one ``kind`` of external payload:

1. ``content_hash(payload_bytes) -> str`` — a **deterministic, content-based** hash of the payload.
   For an ONNX model it is the hash of the *weights* (+ graph structure), for a correctionlib set the
   hash of its *contents* — not the raw file bytes (which carry incidental formatting/metadata).
2. ``load(payload_bytes, params) -> resource`` — materialize the payload **once per worker** (a loaded
   model, or a live connection); defaults to returning the bytes, so simple plugins can ignore it.
3. ``evaluate(resource, params, inputs) -> value`` — run the loaded resource on inputs (per call).
4. ``close(resource)`` — release the resource at end of run (e.g. close a connection); default no-op.
5. ``samples() -> Sequence[bytes]`` — ≥2 distinct example payloads used to *validate the hash*.

A :class:`ResourceCache` loads each payload once and reuses it across calls/nodes — ``open_once`` (M7)
for Externals, so a model/connection is not re-created per partition.

``register_plugin`` validates a plugin's hash before trusting it (the user's explicit requirement):
it must be **deterministic across processes** (so it reproduces on machine B — this catches a hash
built from ``hash()``/``id()``/time/randomness) and **non-vacuous** (distinct payloads must not
collide — this catches a constant or trivially-weak hash). ``onnx`` and ``correctionlib`` ship as
plugins and double as templates for users writing their own.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import PreserveError

ContentHash = Callable[[bytes], str]
Load = Callable[[bytes, Mapping[str, Any]], Any]
Evaluate = Callable[[Any, Mapping[str, Any], list[Any]], Any]
Close = Callable[[Any], None]


def _identity_load(payload: bytes, params: Mapping[str, Any]) -> Any:
    """Default ``load``: the resource IS the payload bytes (simple plugins evaluate from bytes)."""
    return payload


def _noop_close(resource: Any) -> None:
    return None


SynthesizePayload = Callable[[Mapping[str, Any]], "bytes | None"]


@dataclass(frozen=True)
class ExternalPlugin:
    """How to hash, load, evaluate, and (self-)validate one ``kind`` of External payload.

    ``load(payload_bytes, params) -> resource`` materializes the payload **once per worker** (a loaded
    model, or a live connection); ``evaluate(resource, params, inputs)`` then runs it per call. The
    default ``load`` returns the payload bytes, so a simple plugin can ignore it and evaluate straight
    from bytes. ``close(resource)`` releases the resource at the end of a run (e.g. a connection).
    """

    kind: str
    content_hash: ContentHash
    evaluate: Evaluate
    samples: Callable[[], Sequence[bytes]]
    load: Load = _identity_load
    close: Close = _noop_close
    framework: str = ""
    # M25: a plugin whose payload is DERIVABLE from the node's own params (e.g. a histogram
    # fill's canonical spec) may synthesize it at build time, so callers supply no bytes.
    synthesize: SynthesizePayload | None = None


class ResourceCache:
    """Per-run cache of loaded External resources: a payload is ``load``-ed once and reused across
    every call (and across nodes sharing the same payload+params), then ``close``-d on ``close()``.
    This is ``open_once`` (M7) for Externals — no re-loading a model per partition."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str, str], tuple[ExternalPlugin, Any]] = {}

    def resource(
        self, plugin: ExternalPlugin, payload: bytes, params: Mapping[str, Any], *, content_hash: str
    ) -> Any:
        key = (plugin.kind, content_hash, json.dumps(dict(params), sort_keys=True, default=str))
        if key not in self._items:
            self._items[key] = (plugin, plugin.load(payload, params))
        return self._items[key][1]

    def close(self) -> None:
        while self._items:
            _key, (plugin, resource) = self._items.popitem()
            plugin.close(resource)


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


def evaluate_external(
    node: Mapping[str, Any], inputs: list[Any], payload: bytes, cache: ResourceCache | None = None
) -> Any:
    """Evaluate an External node via its registered plugin (dispatch by descriptor kind).

    With a ``cache`` the payload is ``load``-ed once per worker and reused; without one it is loaded,
    evaluated, and closed in a single shot."""
    kind = node["descriptor"]["kind"]
    plugin = get_plugin(kind)
    if plugin is None:
        raise PreserveError(f"no plugin registered for external kind {kind!r} (a preservation risk)")
    params = node["params"]
    if cache is not None:
        resource = cache.resource(plugin, payload, params, content_hash=node["descriptor"]["content_hash"])
        return plugin.evaluate(resource, params, inputs)
    resource = plugin.load(payload, params)
    try:
        return plugin.evaluate(resource, params, inputs)
    finally:
        plugin.close(resource)


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
    holder: dict[str, Any] = {}  # load the resource once for this node (build-time materialize)

    def _fn(*values: Any) -> Any:
        if "resource" not in holder:
            holder["resource"] = plugin.load(payload, node_params)
        return plugin.evaluate(holder["resource"], node_params, list(values))

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


def load_correctionlib(payload: bytes, params: Mapping[str, Any]) -> Any:
    """Parse the correction set once (per worker)."""
    import correctionlib  # noqa: PLC0415

    return correctionlib.CorrectionSet.from_string(payload.decode("utf-8"))


def eval_correctionlib(cset: Any, params: Mapping[str, Any], inputs: list[Any]) -> Any:
    import awkward as ak  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

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


def load_onnx(payload: bytes, params: Mapping[str, Any]) -> Any:
    """Create the ONNX Runtime session once (per worker) — not per call."""
    import onnxruntime as ort  # noqa: PLC0415

    return ort.InferenceSession(payload, providers=["CPUExecutionProvider"])


def eval_onnx(session: Any, params: Mapping[str, Any], inputs: list[Any]) -> Any:
    import awkward as ak  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    name = str(params.get("input_name", "")) or session.get_inputs()[0].name
    x = np.asarray(ak.to_numpy(ak.Array(inputs[0])), dtype="float32").reshape(-1, 1)
    out = session.run(None, {name: x})[0].reshape(-1)
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
    load=load_correctionlib,  # parse the correction set once per worker
    framework="correctionlib",
)
ONNX_PLUGIN = ExternalPlugin(
    kind="onnx_model",
    content_hash=onnx_content_hash,
    evaluate=eval_onnx,
    samples=_onnx_samples,
    load=load_onnx,  # build the inference session once per worker
    framework="onnxruntime",
)


# register the built-ins (validated by the M9 test suite, so skip the per-import subprocess check)
def histogram_content_hash(payload: bytes) -> str:
    """The spec string's SHA-256 — IDENTICAL to the fill node's descriptor hash by construction
    (graphed-histogram derives the node identity from the same canonical spec encoding)."""
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def eval_histogram(resource: Any, params: Mapping[str, Any], inputs: list[Any]) -> Any:
    """Reconstruct the fill from the node's own params and run it (M23's evaluator, verbatim)."""
    from graphed_histogram.boost import FillEvaluator  # noqa: PLC0415  (optional integration)

    payload = resource if isinstance(resource, bytes) else bytes(resource)
    evaluator = FillEvaluator(
        spec=payload.decode(),
        n_axes=int(params.get("n_axes", 1)),
        has_weight=bool(params.get("weighted", False)),
        has_sample=bool(params.get("sampled", False)),
    )
    return evaluator(*inputs)


def _histogram_samples() -> list[bytes]:
    # two distinct canonical specs (plain JSON: validating the hash needs no graphed-histogram)
    return [
        b'{"axes":[{"bins":10,"metadata":{},"overflow":true,"start":0.0,"stop":1.0,"type":"Regular","underflow":true}],"storage":"Double","version":1}',
        b'{"axes":[{"bins":20,"metadata":{},"overflow":true,"start":0.0,"stop":2.0,"type":"Regular","underflow":true}],"storage":"Int64","version":1}',
    ]


def _histogram_synthesize(params: Mapping[str, Any]) -> bytes | None:
    spec = params.get("spec")
    return str(spec).encode() if spec is not None else None


HISTOGRAM_PLUGIN = ExternalPlugin(
    kind="histogram",
    content_hash=histogram_content_hash,
    evaluate=eval_histogram,
    samples=_histogram_samples,
    framework="boost_histogram",
    synthesize=_histogram_synthesize,
)

register_plugin(CORRECTIONLIB_PLUGIN, validate=False)
register_plugin(ONNX_PLUGIN, validate=False)
register_plugin(HISTOGRAM_PLUGIN, validate=False)
