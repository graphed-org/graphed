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

from ..errors import PreserveError

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
