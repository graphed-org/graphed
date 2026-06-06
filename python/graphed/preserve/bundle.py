"""The preservation bundle: ``build_bundle`` / ``reproduce`` / ``inspect`` (plan M9).

A bundle is a directory holding a canonical ``manifest.json`` (the content-addressed bill-of-materials
whose hash is the bundle fingerprint) and a content-addressed ``store/`` (the M8
``graphed_checkpoint.Store``) holding every referenced blob: the canonical serialized IR, each input
dataset, each correction/model payload, and the provenance sourcemap. The bundle is **runnable from
references alone** — no original user code, environment, author, or input files are needed on the
reproducing machine (the environment is captured for audit; inputs are resolved from the archive).
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import os
import pickle
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphed_checkpoint import Store
from graphed_core import GraphStore

from .errors import PreserveError, UnresolvedPayload
from .externals import evaluate_external, get_plugin
from .interpreter import run_ir
from .manifest import FORMAT_VERSION, canonical_bytes, fingerprint

# rounded for cross-platform-stable histogram contents (mirrors graphed-corpus STABLE_DECIMALS)
_STABLE_DECIMALS = 6
_ENV_PACKAGES = (
    "graphed-core",
    "graphed",
    "graphed-awkward",
    "graphed-checkpoint",
    "graphed-corpus",
    "correctionlib",
    "onnx",
    "onnxruntime",
    "awkward",
    "numpy",
)


# ---- on-disk dataset codec (deterministic; awkward via to_buffers) -------------------------------
def _pack_array(arr: Any) -> bytes:
    import awkward as ak  # noqa: PLC0415

    form, length, container = ak.to_buffers(ak.Array(arr))
    return pickle.dumps((form.to_json(), length, dict(container)), protocol=5)


def _unpack_array(blob: bytes) -> Any:
    import awkward as ak  # noqa: PLC0415

    form_json, length, container = pickle.loads(blob)
    return ak.from_buffers(ak.forms.from_json(form_json), length, container)


def _capture_environment(container_digest: str | None) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in _ENV_PACKAGES:
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    env: dict[str, Any] = {"python": platform.python_version(), "packages": packages}
    if container_digest is not None:
        env["container_digest"] = container_digest
    return env


@dataclass(frozen=True)
class Bundle:
    """An on-disk preservation bundle (a directory: ``manifest.json`` + ``store/``)."""

    root: Path
    manifest: dict[str, Any]

    @property
    def store(self) -> Store:
        return Store(self.root / "store")

    def fingerprint(self) -> str:
        return fingerprint(self.manifest)

    @classmethod
    def open(cls, root: str | os.PathLike[str]) -> Bundle:
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return cls(root=root, manifest=manifest)


def _resolve(store: Store, content_hash: str, *, what: str) -> bytes:
    blob = store.get(content_hash)
    if blob is None:
        raise UnresolvedPayload(content_hash, what=what)
    return blob


# ---- build --------------------------------------------------------------------------------------
def build_bundle(
    root: str | os.PathLike[str],
    *,
    session: Any,
    value: Any,
    weight: Any,
    datasets: dict[str, Any],
    payloads: dict[str, bytes],
    histogram: dict[str, Any],
    config: dict[str, Any] | None = None,
    seed: int = 0,
    environment: dict[str, Any] | None = None,
    container_digest: str | None = None,
) -> Bundle:
    """Capture a recorded analysis (``session`` + the ``value`` and ``weight`` output Arrays) as a
    self-contained bundle. ``datasets`` maps each source name to its input array; ``payloads`` maps
    each External descriptor ``content_hash`` to the correction/model file bytes; ``histogram`` is the
    ``{name, bins, lo, hi}`` spec applied to ``value`` weighted by ``weight``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    store = Store(root / "store")

    # 1. the canonical IR (opt_level=0: auditable, 1:1 with user ops, no stage fusion)
    ir = session.serialized_ir(value, weight, optimize=False)
    ir_hash = store.put(ir)
    nodes = GraphStore.deserialize(ir).nodes()

    # 2. input datasets -> content-addressed store
    sources_manifest = {name: store.put(_pack_array(arr)) for name, arr in datasets.items()}

    # 3. external payloads (via plugins) + flag any external with no plugin as a preservation risk
    externals_manifest: list[dict[str, Any]] = []
    opaque_nodes: list[int] = []
    for node in nodes:
        if node["kind"] != "external":
            continue
        desc = node["descriptor"]
        ch = desc["content_hash"]
        plugin = get_plugin(desc["kind"])
        if plugin is None:
            # no plugin (e.g. an opaque cloudpickled `map`, or an unregistered kind) -> not preservable
            opaque_nodes.append(node["id"])
            continue
        if ch not in payloads:
            raise PreserveError(f"no payload bytes supplied for external {ch} (node {node['id']})")
        blob = payloads[ch]
        actual = plugin.content_hash(blob)
        if actual != ch:  # integrity: the bytes must hash to the recorded id (cache-poisoning-safe)
            raise PreserveError(
                f"payload for node {node['id']} hashes to {actual}, not the recorded {ch} (mismatched/poisoned)"
            )
        externals_manifest.append(
            {
                "node_id": node["id"],
                "kind": desc["kind"],
                "content_hash": ch,
                "store": store.put(blob),
                "io_schema": desc["io_schema"],
            }
        )

    # 4. provenance sourcemap (basenames only -> build-location independent), stored
    sourcemap = {
        str(nid): {**prov, "filename": os.path.basename(str(prov.get("filename", "")))}
        for nid, prov in session.sourcemap().items()
    }
    prov_hash = store.put(json.dumps(sourcemap, sort_keys=True).encode("utf-8"))

    # 5. the content-addressed bill-of-materials
    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "analysis": {
            "ir": ir_hash,
            "outputs": {"value": int(value.node_id), "weight": int(weight.node_id)},
            "histogram": dict(histogram),
        },
        "sources": sources_manifest,
        "externals": externals_manifest,
        "opaque_nodes": opaque_nodes,
        "provenance": prov_hash,
        "environment": environment or _capture_environment(container_digest),
        "config": dict(config or {}),
        "seed": int(seed),
    }
    (root / "manifest.json").write_bytes(canonical_bytes(manifest))
    return Bundle(root=root, manifest=manifest)


# ---- reproduce ----------------------------------------------------------------------------------
def reproduce(bundle: Bundle) -> Any:
    """Re-instantiate and run the preserved analysis from references alone; return its histogram.

    Resolves the IR, datasets, and correction/model payloads from the bundle's content-addressed
    store (raising :class:`UnresolvedPayload` for anything missing), interprets the IR through the
    awkward backend + payload-backed external evaluators, and applies the histogram spec."""
    from graphed_awkward import AwkwardBackend  # noqa: PLC0415

    store = bundle.store
    m = bundle.manifest
    nodes = GraphStore.deserialize(_resolve(store, m["analysis"]["ir"], what="IR")).nodes()
    backend = AwkwardBackend()

    cache: dict[str, Any] = {}

    def source(node: dict[str, Any]) -> Any:
        name = node["name"]
        h = m["sources"].get(name)
        if h is None:
            raise UnresolvedPayload(name, what="dataset for source")
        if name not in cache:
            cache[name] = _unpack_array(_resolve(store, h, what="dataset"))
        return cache[name]

    ext_by_node = {e["node_id"]: e for e in m["externals"]}

    def external(node: dict[str, Any], inputs: list[Any]) -> Any:
        entry = ext_by_node.get(node["id"])
        if entry is None:
            raise PreserveError(f"external node {node['id']} is not in the manifest (opaque/unpreserved?)")
        payload = _resolve(store, entry["store"], what=f"{entry['kind']} payload")
        return evaluate_external(node, inputs, payload)

    values = run_ir(
        nodes,
        source=source,
        external=external,
        eval_op=lambda name, ins, params: backend.eval_stage(name, ins, params),
    )
    out = m["analysis"]["outputs"]
    return _histogram(values[out["value"]], values[out["weight"]], m["analysis"]["histogram"])


def _histogram(value: Any, weight: Any, spec: dict[str, Any]) -> Any:
    import awkward as ak  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    v = np.asarray(ak.to_numpy(ak.Array(value)), dtype="float64")
    w = np.asarray(ak.to_numpy(ak.Array(weight)), dtype="float64")
    counts, _edges = np.histogram(v, bins=int(spec["bins"]), range=(spec["lo"], spec["hi"]), weights=w)
    return np.round(counts, _STABLE_DECIMALS)


# ---- inspect (no execution) ---------------------------------------------------------------------
def inspect(bundle: Bundle) -> str:
    """Render the preserved analysis — the IR + M6 provenance + payload inventory + risk flags —
    WITHOUT executing anything or resolving the (possibly absent) data/payloads."""
    store = bundle.store
    m = bundle.manifest
    nodes = GraphStore.deserialize(_resolve(store, m["analysis"]["ir"], what="IR")).nodes()
    sourcemap = json.loads(_resolve(store, m["provenance"], what="sourcemap"))

    lines: list[str] = [
        f"Preservation Bundle  fingerprint={bundle.fingerprint()}",
        f"  environment: python {m['environment'].get('python', '?')}; "
        f"{len(m['environment'].get('packages', {}))} pinned packages"
        + (
            f"; container {m['environment']['container_digest']}"
            if "container_digest" in m["environment"]
            else ""
        ),
        f"  config: {m['config']}   seed: {m['seed']}",
        f"  histogram: {m['analysis']['histogram']}",
        "  graph (IR, opt_level=0):",
    ]
    for node in nodes:
        prov = sourcemap.get(str(node["id"]), {})
        loc = f"{prov.get('filename', '?')}:{prov.get('lineno', '?')}"
        label = node["name"] or node["kind"]
        lines.append(
            f"    n{node['id']:<3} {node['kind']:<9} {label:<14} params={node['params']} "
            f"<- {node['inputs']}   [{loc}]"
        )
    lines.append("  external payloads (HEP standards, content-addressed):")
    for e in m["externals"]:
        lines.append(f"    n{e['node_id']} {e['kind']} ({e['io_schema']}) {e['content_hash']}")
    lines.append("  input datasets:")
    for name, h in m["sources"].items():
        lines.append(f"    {name}: sha256:{h}")
    if m["opaque_nodes"]:
        lines.append(f"  ⚠ PRESERVATION RISK — opaque (cloudpickled) nodes: {m['opaque_nodes']}")
    else:
        lines.append("  no opaque nodes (every node is durable IR or a content-addressed payload)")
    return "\n".join(lines)
