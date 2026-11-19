"""The XGBoost plugin takes UBJSON payloads and model file paths (#255)."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed.awkward import from_awkward
from graphed.preserve import PreserveError
from graphed.preserve.externals import XGBOOST_PLUGIN, record_external
from graphed.preserve.externals._helpers import _canonical_json_hash_impl

xgb = pytest.importorskip("xgboost")

X = np.random.default_rng(0).normal(size=(64, 2))


def _booster(objective: str) -> Any:
    params: dict[str, Any] = {"objective": objective}
    label: Any = X[:, 0] > 0
    if objective.startswith("multi"):
        params["num_class"] = 3
        label = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0)
    return xgb.train(params, xgb.DMatrix(X, label=label), num_boost_round=3)


def _saved(tmp_path: Path, booster: Any, ext: str) -> Path:
    path = tmp_path / f"model.{ext}"
    booster.save_model(str(path))
    return path


def _record(payload: Any) -> tuple[Any, Any]:
    src = from_awkward("e", ak.zip({"a": X[:, 0], "b": X[:, 1]}), session=graphed.new_session())
    return src, record_external(src.session, XGBOOST_PLUGIN, payload, [src.a, src.b])


@pytest.mark.parametrize("ext", ["ubj", "json"])
def test_the_repro_predicts_as_the_booster_does(tmp_path: Path, ext: str) -> None:
    booster = _booster("binary:logistic")
    payload = _saved(tmp_path, booster, ext).read_bytes()
    _src, out = _record(payload)
    got = ak.to_numpy(graphed.materialize(out))
    assert np.array_equal(got, booster.predict(xgb.DMatrix(X)))


@pytest.mark.parametrize("objective", ["binary:logistic", "multi:softprob"])
def test_one_booster_hashes_the_same_as_ubjson_json_and_gzipped_json(
    tmp_path: Path, objective: str
) -> None:
    booster = _booster(objective)
    ubj = _saved(tmp_path, booster, "ubj").read_bytes()
    js = _saved(tmp_path, booster, "json").read_bytes()
    assert ubj[:2] == b"{L" and js[:2] == b'{"'
    hashes = {XGBOOST_PLUGIN.content_hash(p) for p in (ubj, js, gzip.compress(js))}
    assert hashes == {_canonical_json_hash_impl(b"xgboost-json-v1", js)}


def test_two_boosters_hash_apart_as_ubjson(tmp_path: Path) -> None:
    one = _saved(tmp_path, _booster("binary:logistic"), "ubj").read_bytes()
    other = _saved(tmp_path / "..", _booster("multi:softprob"), "ubj").read_bytes()
    assert XGBOOST_PLUGIN.content_hash(one) != XGBOOST_PLUGIN.content_hash(other)


@pytest.mark.parametrize("spell", [str, Path], ids=["str", "Path"])
def test_a_path_payload_records_the_node_its_bytes_record(tmp_path: Path, spell: Any) -> None:
    path = _saved(tmp_path, _booster("binary:logistic"), "ubj")
    src = from_awkward("e", ak.zip({"a": X[:, 0], "b": X[:, 1]}), session=graphed.new_session())
    by_path = record_external(src.session, XGBOOST_PLUGIN, spell(path), [src.a, src.b])
    by_bytes = record_external(src.session, XGBOOST_PLUGIN, path.read_bytes(), [src.a, src.b])
    assert by_path.node_id == by_bytes.node_id


def test_a_payload_of_no_known_format_is_refused_by_name() -> None:
    with pytest.raises(PreserveError, match="JSON, gzipped JSON or UBJSON; this is none"):
        XGBOOST_PLUGIN.content_hash(b"\x89PNG definitely not a model")


@pytest.mark.parametrize(
    ("payload", "why"),
    [(b"{L", "buffer"), (b"{i\x01aZ}junk", "trailing bytes"), (b"{i\x01aQ}", "'Q'")],
    ids=["truncated", "trailing", "marker"],
)
def test_malformed_ubjson_is_refused_by_name(payload: bytes, why: str) -> None:
    with pytest.raises(PreserveError, match=f"not valid UBJSON.*{why}"):
        XGBOOST_PLUGIN.content_hash(payload)
