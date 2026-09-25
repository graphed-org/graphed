"""Fixtures for the m71a awkward suite: a session whose backend carries a `Photon` behavior.

`PhotonArray` is registered ONLY in `BEHAVIOR` (never global `ak.behavior`), so a declared
`Photon` record resolves `pt2`/`scaled` through the backend alone. Every m71a-new surface (the
`output_type=` keyword) is reached inside test bodies only, so the tree collects on a base without it.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import awkward as ak
import numpy as np

from graphed import GraphedTypeError, Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.numpy import NumpyBackend, from_array


class PhotonRecord(ak.Record):
    pass


class PhotonArray(ak.Array):
    @property
    def pt2(self) -> Any:
        return self.pt * 2

    def scaled(self, k: float) -> Any:
        return self.pt * k


BEHAVIOR: dict[Any, Any] = {"Photon": PhotonRecord, ("*", "Photon"): PhotonArray}

EVENTS = ak.zip(
    {
        "run": np.array([1, 2, 1, 3], dtype=np.uint32),
        "x": np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32),
        "Jet": ak.zip({"pt": ak.Array([[30.0, 20.0], [], [40.0], [10.0, 50.0, 60.0]])}),
    },
    depth_limit=1,
)

PHOTON = "Photon[pt: float32, eta: float32]"

#: every numpy numerical dtype (``np.typecodes`` "AllInteger" + "AllFloat" + bool, fixed-width only)
NUMERIC = (
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
)


def f(a: Any) -> Any:
    return a > 1


def recorded() -> tuple[Session, Any]:
    session = Session(AwkwardBackend(behavior=BEHAVIOR))
    return session, from_awkward(session, "events", EVENTS)


def eager() -> Any:
    """The eager, behavior-carrying counterpart of the recorded `events`."""
    return ak.Array(EVENTS.layout, behavior=BEHAVIOR)


def numpy_recorded() -> tuple[Session, Any]:
    session = Session(NumpyBackend())
    return session, from_array(session, "x", np.array([0.5, 1.5], dtype=np.float32))


def describe(session: Session, array: Any) -> str:
    return str(session.form(array).describe())


def params_of(session: Session, array: Any) -> dict[str, Any]:
    """The record-time store params behind `array` (the house route)."""
    node = next(n for n in session._store.nodes() if n["id"] == array.node_id)
    return dict(node["params"])


def same(value: Any, expected: Any) -> bool:
    """Equal values AND equal types, so a right-valued wrong-typed result fails."""
    return bool(ak.to_list(value) == ak.to_list(expected) and str(ak.type(value)) == str(ak.type(expected)))


# ---- callables whose values have the declared type --------------------------------------------


def pair(a: Any) -> Any:
    """`{pt: float32, eta: float32}` per element of a float32 column."""
    return ak.zip({"pt": a, "eta": a * 2})


def photon(a: Any) -> Any:
    return ak.zip({"pt": a, "eta": a * 2}, with_name="Photon")


def jet_photons(p: Any) -> Any:
    """`var * Photon[pt: float32, eta: float32]` over `Jet.pt`."""
    pt = ak.values_astype(p, "float32")
    return ak.zip({"pt": pt, "eta": pt * 0.01}, with_name="Photon")


def masked_jets(p: Any) -> Any:
    """`var * ?{pt: float32, idx: int64}` over `Jet.pt`."""
    rec = ak.zip({"pt": ak.values_astype(p, "float32"), "idx": ak.local_index(p)})
    return ak.mask(rec, p > 25)


def strings(a: Any) -> Any:
    return ak.from_numpy(ak.to_numpy(a).astype("U5"))


def byte_strings(a: Any) -> Any:
    return ak.from_numpy(ak.to_numpy(a).astype("S5"))


def triples(a: Any) -> Any:
    return ak.from_numpy(np.repeat(ak.to_numpy(a)[:, None], 3, axis=1))


def refusal(call: Callable[[], object]) -> GraphedTypeError:
    try:
        call()
    except GraphedTypeError as exc:
        return exc
    raise AssertionError("expected a GraphedTypeError")


def here() -> int:
    """The caller's NEXT line: the line a one-line refused call sits on."""
    return sys._getframe(1).f_lineno + 1
