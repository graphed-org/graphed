"""What graphed hashes or ships for a concatenation is independent of PYTHONHASHSEED (#238)."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys

import awkward as ak
import pytest

from graphed.awkward._form_order import ordered_array
from graphed.awkward.shuffle import from_wire, to_wire
from graphed.checkpoint.codec import PickleCodec
from graphed.preserve.bundle import _pack_array, _unpack_array

# ak.concatenate intersects the parameters of its inputs as a set
PROBE = """
import awkward as ak, hashlib, json, sys
from graphed.awkward.shuffle import to_wire
from graphed.checkpoint.codec import PickleCodec
from graphed.preserve.bundle import _pack_array
a = ak.Array([[1.0], [2.0, 3.0]])
for key in ("__array__", "collection_name", "p3", "p4"):
    a = ak.with_parameter(a, key, key + "-value")
c = ak.concatenate([a, a])
class PtRecord(ak.Record): pass
class PtArray(ak.Array): pass
r = ak.with_name(ak.Array([{"x": 1.0}, {"x": 2.0}]), "Pt")
for key in ("collection_name", "p3", "p4"):
    r = ak.with_parameter(r, key, key + "-value")
pts = ak.Array(ak.concatenate([r, r]), behavior={"Pt": PtRecord, ("*", "Pt"): PtArray})
assert type(pts) is PtArray and type(pts[0]) is PtRecord
import pickle
import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.awkward.backend import _AsRead
from graphed.awkward.io import _record_types
from graphed.checkpoint.resume import _dumps
def captured(x, _c=c):
    return x
session = Session(AwkwardBackend())
ev = from_awkward("ev", ak.Array({"pts": ak.concatenate([r, r])}), session=session)
varied = graphed.vary(ev.pts, "s", points={"up": ev.pts})
labels = ["nominal", "up"]
digest = lambda b: hashlib.sha256(b).hexdigest()
print(json.dumps({
    "task key": digest(_dumps(captured, {}, set())),
    "replay capture": digest(pickle.dumps(_AsRead(c), protocol=5)),
    "write metadata": digest(json.dumps(_record_types(session, varied, labels)).encode()),
    "order": list(c.layout.parameters),
    "checkpoint": digest(PickleCodec().encode({"partial": c, "n": 2})),
    "subclass": digest(PickleCodec().encode(pts)),
    "record": digest(PickleCodec().encode(pts[0])),
    "bundle": digest(_pack_array(c)),
    "shuffle": digest(to_wire(c)),
}))
"""


@pytest.fixture(scope="module")
def runs() -> list[dict[str, object]]:
    out = []
    for seed in range(5):
        env = {**os.environ, "PYTHONHASHSEED": str(seed)}
        done = subprocess.run(
            [sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, check=True
        )
        out.append(json.loads(done.stdout))
    return out


def test_the_seeds_reorder_awkwards_parameters(runs: list[dict[str, object]]) -> None:
    assert len({tuple(r["order"]) for r in runs}) > 1  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "site",
    [
        "checkpoint",
        "subclass",
        "record",
        "bundle",
        "shuffle",
        "task key",
        "replay capture",
        "write metadata",
    ],
)
def test_each_encoder_writes_one_byte_stream_across_seeds(
    runs: list[dict[str, object]], site: str
) -> None:
    assert len({r[site] for r in runs}) == 1


def _unordered() -> ak.Array:
    # out of key order in every process, as ak.concatenate returns them in some
    a = ak.with_parameter(ak.Array([[1.0], [2.0]]), "collection_name", "X")
    return ak.with_parameter(a, "__array__", "NanoCollection")


def test_the_decoded_values_keep_their_data_and_parameters() -> None:
    c = _unordered()
    for back in (
        PickleCodec().decode(PickleCodec().encode(c)),
        _unpack_array(_pack_array(c)),
        from_wire(to_wire(c)),
    ):
        assert back.to_list() == c.to_list()
        assert back.layout.parameters == c.layout.parameters


def test_an_ordered_array_keeps_behavior_and_attrs() -> None:
    c = ak.Array(_unordered(), attrs={"k": 1}, behavior={"b": 1})
    out = ordered_array(c)
    assert list(out.layout.parameters) == sorted(c.layout.parameters)
    assert out.attrs == {"k": 1} and out.behavior == {"b": 1}
    assert ordered_array(out) is out


def test_a_value_without_awkward_arrays_pickles_as_before() -> None:
    value = {"x": 1.5, "y": [1, 2]}
    assert PickleCodec().encode(value) == pickle.dumps(value, protocol=PickleCodec.PROTOCOL)


def test_a_decoded_record_keeps_its_data_and_parameters() -> None:
    r = ak.with_parameter(ak.Array([{"x": 1.0}, {"x": 2.0}]), "collection_name", "X")
    rec = ak.with_parameter(r, "__record__", "Pt")[1]
    back = PickleCodec().decode(PickleCodec().encode(rec))
    assert back.to_list() == rec.to_list()
    assert back.layout.array.parameters == rec.layout.array.parameters


def test_encoding_leaves_an_unset_attrs_unset() -> None:
    c = _unordered()
    assert PickleCodec().decode(PickleCodec().encode(c))._attrs is None
    assert c._attrs is None
