"""M9 acceptance — `inspect` renders logic + payload inventory + risk flags WITHOUT executing.

Plan M9: "``inspect(bundle)`` renders the analysis logic (the graph IR + the M6 user-source
provenance/sourcemap) as human-readable output WITHOUT executing anything, and lists every External
payload with its descriptor ... and every ``opaque=True`` node flagged as a preservation risk."
Review focus: ``inspect`` must be faithful to what ``reproduce`` runs (no drift).
"""

from __future__ import annotations

from typing import Any

import agc
import numpy as np
import pytest
from graphed_core import GraphStore

from graphed_preserve import UnresolvedPayload, inspect, reproduce


def test_inspect_renders_ir_provenance_and_payload_inventory(agc_bundle: tuple[Any, np.ndarray]) -> None:
    bundle, _ = agc_bundle
    text = inspect(bundle)
    assert bundle.fingerprint() in text
    assert "correctionlib" in text and "onnx_model" in text  # payload inventory with descriptors
    assert "events" in text  # input dataset listed
    assert "agc.py:" in text  # M6 user-source provenance (basename:line)
    assert "no opaque nodes" in text  # this analysis is fully durable IR + content-addressed payloads


def test_inspect_neither_executes_nor_resolves_data(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bundle, _ = agc.build_agc(tmp_path)
    objects = bundle.root / "store" / "objects"
    # remove the data + payloads (everything execution needs) — keep only IR + sourcemap
    (objects / bundle.manifest["sources"]["events"]).unlink()
    for e in bundle.manifest["externals"]:
        (objects / e["store"]).unlink()
    # inspect still renders (it only reads the IR + sourcemap) ...
    assert "graph (IR" in inspect(bundle)
    # ... while reproduce now fails loudly, proving inspect did not execute
    with pytest.raises(UnresolvedPayload):
        reproduce(bundle)


def test_inspect_is_faithful_to_the_reproduced_graph(agc_bundle: tuple[Any, np.ndarray]) -> None:
    bundle, _ = agc_bundle
    ir = bundle.store.get(bundle.manifest["analysis"]["ir"])
    n_nodes = len(GraphStore.deserialize(ir).nodes())
    # every IR node reproduce walks is rendered in inspect's graph section (one "n<id> " line each)
    lines = inspect(bundle).splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("graph (IR"))
    end = next(i for i in range(start + 1, len(lines)) if not lines[i].startswith("    n"))
    assert end - (start + 1) == n_nodes


def test_opaque_cloudpickled_node_is_flagged_as_a_risk(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bundle = agc.build_opaque(tmp_path)
    assert bundle.manifest["opaque_nodes"], "an opaque map node must be recorded as a risk"
    assert "PRESERVATION RISK" in inspect(bundle)
