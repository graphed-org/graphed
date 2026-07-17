"""M9 acceptance — a missing referenced payload fails loudly, never silently.

Plan M9: "A bundle whose referenced payloads are missing from the archive fails ``reproduce()`` with
a precise 'unresolved payload <hash>' error, never a silent wrong result."
"""

from __future__ import annotations

import shutil

import agc
import pytest

from graphed_preserve import Bundle, UnresolvedPayload, reproduce


def _copy(bundle, dest):  # type: ignore[no-untyped-def]
    shutil.copytree(bundle.root, dest)
    return Bundle.open(dest)


def test_missing_correction_payload_raises_with_its_hash(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bundle, _ = agc.build_agc(tmp_path / "build")
    corr = next(e for e in bundle.manifest["externals"] if e["kind"] == "correctionlib")
    copy = _copy(bundle, tmp_path / "c")
    (copy.root / "store" / "objects" / corr["store"]).unlink()
    with pytest.raises(UnresolvedPayload) as exc:
        reproduce(copy)
    assert corr["store"] in str(exc.value)


def test_missing_dataset_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bundle, _ = agc.build_agc(tmp_path / "build")
    copy = _copy(bundle, tmp_path / "d")
    (copy.root / "store" / "objects" / bundle.manifest["sources"]["events"]).unlink()
    with pytest.raises(UnresolvedPayload):
        reproduce(copy)


def test_missing_ir_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bundle, _ = agc.build_agc(tmp_path / "build")
    copy = _copy(bundle, tmp_path / "i")
    (copy.root / "store" / "objects" / bundle.manifest["analysis"]["ir"]).unlink()
    with pytest.raises(UnresolvedPayload):
        reproduce(copy)
