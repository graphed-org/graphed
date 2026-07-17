"""The bundle manifest — a content-addressed bill-of-materials (plan M9).

The manifest binds every component of a preservation bundle **by content hash** (the IR, each input
dataset, each correction/model payload, the provenance sourcemap) plus the environment, config, and
seed. It is serialized canonically (sorted keys, no insignificant whitespace, no timestamps or
absolute paths), and **its own hash is the bundle's top-level fingerprint**: change any result- or
content-determining input and the fingerprint changes; change nothing and it is identical.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

FORMAT_VERSION = 1


def canonical_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Deterministic serialization of the manifest (the bytes that get hashed)."""
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def fingerprint(manifest: Mapping[str, Any]) -> str:
    """The bundle's top-level content hash (SHA-256 of the canonical manifest)."""
    return "sha256:" + hashlib.sha256(canonical_bytes(manifest)).hexdigest()
