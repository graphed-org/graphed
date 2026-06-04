"""PayloadDescriptors for HEP-standard external inputs (plan M3 / A.3.1).

Corrections via correctionlib (JSON), models via ONNX, datasets via id + file content hashes. We
content-hash the actual file so any change to the correction set / model / dataset changes the
descriptor (and therefore the External node's identity, and later the M9 bundle fingerprint).
Reuse the standards — invent no formats.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

from graphed_core import PayloadDescriptor


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def correctionlib_descriptor(json_path: str, correction_name: str) -> PayloadDescriptor:
    """Descriptor for a correctionlib correction set: content hash + name + schema version."""
    content_hash = _sha256(json_path)
    schema_version = "unknown"
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        schema_version = str(data.get("schema_version", "unknown"))
    except Exception:  # pragma: no cover - malformed JSON still gets a hash
        pass
    return PayloadDescriptor(
        kind="correctionlib",
        content_hash=content_hash,
        framework="correctionlib",
        version=schema_version,
        io_schema=correction_name,
        preprocessing_ref=None,
    )


def onnx_descriptor(model_path: str) -> PayloadDescriptor:
    """Descriptor for an ONNX model: content hash + opset + I/O schema."""
    content_hash = _sha256(model_path)
    opset = "unknown"
    io_schema = "opaque"
    try:
        import onnx

        model = onnx.load(model_path)
        opset = ",".join(str(o.version) for o in model.opset_import)
        ins = ",".join(i.name for i in model.graph.input)
        outs = ",".join(o.name for o in model.graph.output)
        io_schema = f"{ins}->{outs}"
    except Exception:  # pragma: no cover - missing onnx still gets a hash
        pass
    return PayloadDescriptor(
        kind="onnx_model",
        content_hash=content_hash,
        framework="onnxruntime",
        version=opset,
        io_schema=io_schema,
        preprocessing_ref=None,
    )


def dataset_descriptor(dataset_id: str, file_paths: list[str]) -> PayloadDescriptor:
    """Descriptor for a ROOT/parquet dataset source: dataset id + per-file content hashes."""
    hashes = ";".join(_sha256(p) for p in file_paths)
    return PayloadDescriptor(
        kind="dataset",
        content_hash=hashes,
        framework="uproot",
        version="",
        io_schema=dataset_id,
        preprocessing_ref=None,
    )


def opaque_callable_descriptor(fn_name: str) -> PayloadDescriptor:
    """Descriptor for an opaque Python callable (preservation risk; M9 does real hashing)."""
    return PayloadDescriptor(
        kind="opaque_callable",
        content_hash=f"unhashed-opaque:{fn_name}",
        framework="python",
        version=platform.python_version(),
        io_schema="opaque->opaque",
        preprocessing_ref=None,
    )
