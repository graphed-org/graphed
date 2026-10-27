"""Reproduce and the External evaluator refuse what a bundle or binding lacks."""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.preserve import (
    CORRECTIONLIB_PLUGIN,
    TRITON_PLUGIN,
    build_bundle,
    record_external,
    reproduce,
)
from graphed.preserve.bundle import Bundle
from graphed.preserve.errors import PreserveError, UnresolvedPayloadError
from graphed.preserve.externals._base import _PluginEvaluator
from graphed.services import UnboundServiceError

CSET = (
    b'{"schema_version": 2, "corrections": [{"name": "sf", "version": 1, "inputs": [{"name": '
    b'"systematic", "type": "string"}, {"name": "x", "type": "real"}], "output": {"name": "sf", '
    b'"type": "real"}, "data": {"nodetype": "category", "input": "systematic", "content": '
    b'[{"key": "nominal", "value": 1.0}]}}]}'
)
EVENTS = ak.Array({"x": np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)})


def _bundle(root: Path) -> Bundle:
    session = Session(AwkwardBackend())
    events = from_awkward(session, "events", EVENTS)
    weight = record_external(
        session,
        CORRECTIONLIB_PLUGIN,
        CSET,
        [events.x],
        params={"name": "sf", "args": ["nominal", "$0"]},
    )
    return build_bundle(
        root,
        session=session,
        value=events.x,
        weight=weight,
        datasets={"events": EVENTS},
        payloads={CORRECTIONLIB_PLUGIN.content_hash(CSET): CSET},
        histogram={"name": "x", "bins": 2, "lo": 0.0, "hi": 4.0},
    )


def test_reproduce_refuses_a_source_without_a_dataset(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "b")
    assert reproduce(bundle).tolist() == [2.0, 2.0]
    stripped = Bundle(root=bundle.root, manifest={**bundle.manifest, "sources": {}})
    with pytest.raises(UnresolvedPayloadError, match="dataset for source events"):
        reproduce(stripped)


def test_reproduce_refuses_an_external_missing_from_the_manifest(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "b")
    stripped = Bundle(root=bundle.root, manifest={**bundle.manifest, "externals": []})
    with pytest.raises(PreserveError, match="is not in the manifest"):
        reproduce(stripped)


def test_an_unbound_service_evaluator_refuses_to_run() -> None:
    evaluator = _PluginEvaluator(TRITON_PLUGIN, b"m", {"service": "tagger"})
    with pytest.raises(UnboundServiceError, match="service 'tagger' has no endpoint"):
        evaluator(np.zeros(1))
