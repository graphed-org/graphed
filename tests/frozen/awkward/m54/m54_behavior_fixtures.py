"""Fixtures for the m54 behavior-methods suite (awkward backend).

`JetArray` is registered ONLY in this module's behavior dict, never in global `ak.behavior`, so the
backend is the sole route to it; vector's own `deltaR`/`rotateZ` cover the globally registered half.
Every m54-new surface (`graphed.BoundMethod`, a method CALL) is reached inside test bodies only, so
the tree COLLECTS against a tree with no m54 implementation and fails at RUN time (TEST_SANITY).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import Session

vector = pytest.importorskip("vector")
vector.register_awkward()

from graphed.awkward import AwkwardBackend, from_awkward, gak  # noqa: E402


class JetArray(vector.backends.awkward.MomentumArray4D):  # type: ignore[misc,name-defined]
    """A four-vector whose extra methods take arguments; known to `BEHAVIOR` only."""

    def scaled(self, k: float, *, offset: float = 0.0, gain: float = 1.0) -> Any:
        """A positional constant and TWO keyword constants, so keyword ORDER is testable."""
        return (self.pt * k + offset) * gain

    def near(self, other: Any, *, threshold: float | None = None) -> Any:
        """`other` is an array, passable positionally or by keyword."""
        delta = self.deltaR(other)
        return delta if threshold is None else delta < threshold

    def combo(self, coeffs: Any) -> Any:
        """A sequence constant, read by index — list and tuple are the same call."""
        return self.pt * coeffs[0] + self.eta * coeffs[1]

    def blend(self, *, x: Any, y: Any) -> Any:
        """TWO array arguments, keyword-only and ASYMMETRIC, so swapping them is a different call."""
        return self.deltaR(x) + 2.0 * self.deltaR(y)

    def weighted(self, weights: Any) -> Any:
        """A str-keyed dict constant: the accepted collection the refusals are contrasted with."""
        return self.pt * weights["pt"] + self.eta * weights["eta"]

    def mixed(self, parts: Any) -> Any:
        """A dict CONSTANT holding arrays, asymmetric in its keys."""
        return self.deltaR(parts["x"]) + 2.0 * self.deltaR(parts["y"])

    @staticmethod
    def combine(x: Any, y: Any) -> Any:
        """A staticmethod: callable, so a method."""
        return x.deltaR(y)

    @classmethod
    def spread(cls, x: Any, y: Any) -> Any:
        """A classmethod: not callable, but a descriptor, so a method."""
        return x.deltaR(y)

    #: a bare class constant: not a descriptor, so it is read like a field
    MUON_MASS = 0.105

    def _boost(self, k: float) -> Any:
        return self.pt * k

    #: a `partialmethod`: not callable, a descriptor, so a method
    doubled = functools.partialmethod(_boost, 2.0)

    @functools.singledispatchmethod
    def widen(self, k: Any) -> Any:
        """A `singledispatchmethod`: not callable, a descriptor, so a method."""
        raise NotImplementedError

    @widen.register(float)
    def _widen_float(self, k: float) -> Any:
        return self.pt * k

    @widen.register(str)
    def _widen_str(self, k: str) -> Any:
        return self.eta

    def split(self) -> tuple[Any, Any]:
        return self.pt * 2.0, self.eta * 3.0

    def count_scalar(self) -> int:
        return 42

    @property
    def heavy(self) -> Any:
        """A backend-only PROPERTY: must record an array, never a bound method."""
        return self.mass > 5.0

    @property
    def calibrated(self) -> Any:
        """Reads `attrs`; a re-wrap built from the layout alone answers the default, half this."""
        return self.pt * self.attrs.get("calib", 1.0)

    def calibrate(self, k: float) -> Any:
        return self.pt * k * self.attrs.get("calib", 1.0)

    @functools.cached_property
    def bulk(self) -> Any:
        """A `cached_property` is NOT callable, so it is read like a field, not called."""
        return self.mass * 2.0


BEHAVIOR: dict[Any, Any] = dict(vector.backends.awkward.behavior)

# vector's Momentum4D UFUNC overloads, re-keyed onto the `Jet` name so `a + a` resolves through the
# BACKEND's dict alone — these keys never reach global `ak.behavior`. The `isinstance(str)` guard
# skips the class-registration keys (`("*", "Momentum4D")`), which must not shadow `JetArray`.
for _key, _overload in list(vector.backends.awkward.behavior.items()):
    if (
        isinstance(_key, tuple)
        and len(_key) > 1
        and not isinstance(_key[0], str)
        and set(_key[1:]) == {"Momentum4D"}
    ):
        BEHAVIOR[(_key[0], *("Jet",) * (len(_key) - 1))] = _overload

BEHAVIOR[("*", "Jet")] = JetArray

#: the jes shift the `Varied` fixtures apply to `Jet.pt`
JES = {"up": 1.1, "down": 0.9}


def _collection(seed: int, *, extra_field: bool = False) -> ak.Array:
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 4, 8)
    n = int(counts.sum())
    fields = {
        "pt": rng.uniform(20.0, 100.0, n),
        "eta": rng.uniform(-2.4, 2.4, n),
        "phi": rng.uniform(-np.pi, np.pi, n),
        "mass": rng.uniform(0.0, 10.0, n),
    }
    if extra_field:
        fields["scaled"] = rng.uniform(0.0, 1.0, n)
    return ak.with_name(ak.unflatten(ak.Array(fields), counts), "Jet")


#: `Shadow` carries a real leaf named `scaled`, shadowing `JetArray.scaled`
EVENTS = ak.Array(
    {
        "Jet": _collection(54),
        "Probe": _collection(154),
        "Shadow": _collection(254, extra_field=True),
    }
)


def recorded() -> tuple[Session, Any, Any]:
    """A session carrying `BEHAVIOR`, plus the two named jet collections."""
    session = Session(AwkwardBackend(behavior=BEHAVIOR))
    events = from_awkward(session, "events", EVENTS)
    return (
        session,
        gak.with_name(events.Jet, "Jet"),
        gak.with_name(events.Probe, "Jet"),
    )


def collection(session: Session, name: str) -> Any:
    """A third named collection of the same session, by name."""
    events = from_awkward(session, "events", EVENTS)
    return gak.with_name(events[name], "Jet")


def eager(name: str) -> ak.Array:
    """The eager, behavior-carrying counterpart of collection `name`."""
    return ak.Array(EVENTS[name].layout, behavior=BEHAVIOR)


def eager_jets(label: str) -> ak.Array:
    """`Jet` at one label of the jes family: nominal, or pt scaled by that universe's factor."""
    jets = eager("Jet")
    if label == "nominal":
        return jets
    shifted = ak.with_field(jets, jets.pt * JES[label.removeprefix("jes_")], "pt")
    return ak.Array(shifted.layout, behavior=BEHAVIOR)


def varied_jets(jets: Any) -> Any:
    """`graphed.vary` over the RECORD, so every universe keeps the Jet behavior name."""
    return graphed.vary(
        jets,
        "jes",
        up=gak.with_field(jets, jets.pt * JES["up"], "pt"),
        down=gak.with_field(jets, jets.pt * JES["down"], "pt"),
    )


def touched_columns(
    expression: Callable[[ak.Array], Any], source: ak.Array | None = None
) -> set[str]:
    """The leaves `expression` reads, measured by an EAGER reporting typetracer over `source`
    (default `EVENTS`) — the independent oracle graphed's own projection is compared against."""
    form = (EVENTS if source is None else source).layout.form_with_key()
    paths: dict[str, str] = {}

    def walk(node: Any, path: list[str]) -> None:
        if node.is_numpy:
            paths[node.form_key] = ".".join(path)
        elif node.is_record:
            for field, content in zip(node.fields, node.contents, strict=True):
                walk(content, [*path, field])
        elif node.is_list or node.is_option or node.is_indexed:
            walk(node.content, path)

    walk(form, [])
    tracer, report = ak.typetracer.typetracer_with_report(form)
    expression(ak.Array(tracer, behavior=BEHAVIOR))
    return {paths[key] for key in report.data_touched if key in paths}


def node_of(session: Session, array: Any) -> dict[str, Any]:
    """The record-time store entry behind `array` (the house route, as m28/m48 use)."""
    return next(n for n in session._store.nodes() if n["id"] == array.node_id)


def op_count(session: Session) -> int:
    return len(session._ops)


def jets_source() -> tuple[Session, Any, int]:
    """A session whose SOURCE record is itself Jet-named, so a method applies directly to it."""
    session = Session(AwkwardBackend(behavior=BEHAVIOR))
    source = from_awkward(session, "jets", EVENTS.Jet)
    return session, source, source_id(session)


def source_id(session: Session) -> int:
    return next(n["id"] for n in session._store.nodes() if n["kind"] == "source")


def top_level(leaves: set[str]) -> set[str]:
    """`read_columns` answers top-level source fields; the oracle answers dotted leaves."""
    return {leaf.split(".")[0] for leaf in leaves}


#: the attr the calibration members read
CALIB = 2.0


def attrs_sources() -> tuple[ak.Array, ak.Array]:
    """A Jet-named array and an events record, both carrying `attrs={"calib": CALIB}`."""
    return (
        ak.Array(EVENTS.Jet.layout, behavior=BEHAVIOR, attrs={"calib": CALIB}),
        ak.Array(EVENTS.layout, behavior=BEHAVIOR, attrs={"calib": CALIB}),
    )
