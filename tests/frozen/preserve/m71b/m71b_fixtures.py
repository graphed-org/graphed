"""Fixtures for the m71b suite: preservable plugins whose values are NOT their first input's type.

`LUMI` (a bool per event over the uint32 `run`), `JSEL` (a jagged bool over `Jet.pt`) and `PHO` (a
`var * Photon` record over `Jet.pt`) are the graphed#57 shapes. `PHOTON_ARRAY` lives only in
`BEHAVIOR`. The m71b-new keyword is reached inside test bodies only.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import awkward as ak
import numpy as np

from graphed import GraphedTypeError, Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.numpy import NumpyBackend, from_array
from graphed.preserve.externals import ExternalPlugin, sha256_bytes


class PhotonRecord(ak.Record):
    pass


class PhotonArray(ak.Array):
    @property
    def pt2(self) -> Any:
        return self.pt * 2


BEHAVIOR: dict[Any, Any] = {"Photon": PhotonRecord, ("*", "Photon"): PhotonArray}

EVENTS = ak.zip(
    {
        "run": np.array([1, 2, 1, 3], dtype=np.uint32),
        "x": np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32),
        "Jet": ak.zip({"pt": ak.Array([[30.0, 20.0], [], [40.0], [10.0, 50.0, 60.0]])}),
    },
    depth_limit=1,
)

PHOTON = "var * Photon[pt: float32, eta: float32]"


def _samples() -> list[bytes]:
    return [b"a", b"b"]


def lumi(resource: Any, params: Mapping[str, Any], inputs: Sequence[Any]) -> Any:
    return ak.Array(np.asarray(inputs[0]) == 1)


def jet_select(resource: Any, params: Mapping[str, Any], inputs: Sequence[Any]) -> Any:
    return inputs[0] > 25.0


def photons(resource: Any, params: Mapping[str, Any], inputs: Sequence[Any]) -> Any:
    pt = ak.values_astype(inputs[0], "float32")
    return ak.zip({"pt": pt, "eta": pt * 0.01}, with_name="Photon")


LUMI = ExternalPlugin(kind="m71b_lumimask", content_hash=sha256_bytes, evaluate=lumi, samples=_samples)
JSEL = ExternalPlugin(kind="m71b_jetsel", content_hash=sha256_bytes, evaluate=jet_select, samples=_samples)
PHO = ExternalPlugin(kind="m71b_photons", content_hash=sha256_bytes, evaluate=photons, samples=_samples)

#: every `params` mapping `MASK` was evaluated with, in call order
SEEN: list[dict[str, Any]] = []


def mask(resource: Any, params: Mapping[str, Any], inputs: Sequence[Any]) -> Any:
    SEEN.append(dict(params))
    return inputs[0] > 1.0


MASK = ExternalPlugin(kind="m71b_roundtrip_mask", content_hash=sha256_bytes, evaluate=mask, samples=_samples)


def recorded() -> tuple[Session, Any]:
    session = Session(AwkwardBackend(behavior=BEHAVIOR))
    return session, from_awkward(session, "events", EVENTS)


def eager() -> Any:
    return ak.Array(EVENTS.layout, behavior=BEHAVIOR)


def numpy_recorded() -> tuple[Session, Any]:
    session = Session(NumpyBackend())
    return session, from_array(session, "x", np.array([0.5, 1.5], dtype=np.float32))


def describe(session: Session, array: Any) -> str:
    return str(session.form(array).describe())


def params_of(session: Session, array: Any) -> dict[str, Any]:
    node = next(n for n in session._store.nodes() if n["id"] == array.node_id)
    return dict(node["params"])


def same(value: Any, expected: Any) -> bool:
    """Equal values AND equal types."""
    return bool(ak.to_list(value) == ak.to_list(expected) and str(ak.type(value)) == str(ak.type(expected)))


def refusal(call: Callable[[], object]) -> GraphedTypeError:
    try:
        call()
    except GraphedTypeError as exc:
        return exc
    raise AssertionError("expected a GraphedTypeError")


def here() -> int:
    """The caller's NEXT line: the line a one-line refused call sits on."""
    return sys._getframe(1).f_lineno + 1
