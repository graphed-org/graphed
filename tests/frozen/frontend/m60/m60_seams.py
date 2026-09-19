"""Fixtures for the frontend m60 frozen suite — library-integration seams.

The `m60_` prefix is load-bearing: the frozen frontend tree runs one process per milestone dir, and
under prepend import mode a bare helper name binds to whichever sibling dir imported it first.

Numpy-idiom and awkward-free: the free-threaded frontend job installs only `pytest hypothesis
numpy`, so nothing here may import `graphed.awkward`, `awkward`, `pyarrow` or `pandas`.

Every source value is a small integer and every multiplier a power of ten, so each materialized
result below is exact and comparable literally.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

import graphed
import graphed.core
import graphed.numpy
from graphed import Array, Session
from graphed.context import EventContext
from graphed.core import PayloadDescriptor
from graphed.numpy import NumpyBackend, from_record

#: the toy source payload; `DATUM + 1` and `DATUM * 100` are the two distinct-callable answers
DATUM = 2


@dataclass(frozen=True)
class ToyForm:
    tag: str

    def describe(self) -> str:
        return self.tag


class ToyBackend:
    """A backend that types everything and carries data through, so a node's IDENTITY and its
    evaluated VALUE are both observable. `external_payload` derives the content hash from the `fn`
    param exactly as the shipping backends do, so an opaque callable's identity travels in the
    param — which is what makes two colliding callables one node."""

    def op_form(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> ToyForm:
        return ToyForm(op)

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        return (op, list(inputs), dict(params))

    def boundary_ops(self) -> frozenset[str]:
        return frozenset({"source", "map"})

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        return used

    def external_payload(self, op: str, params: Mapping[str, object]) -> PayloadDescriptor | None:
        return PayloadDescriptor(
            kind="opaque_callable",
            content_hash=f"unhashed-opaque:{params.get('fn', 'lambda')}",
            framework="python",
            version="toy",
            io_schema="opaque->opaque",
            preprocessing_ref=None,
        )


def source(session: Session, name: str, value: object = DATUM) -> Array:
    return session.source(name, form=ToyForm("source"), data=value)


def toy_session(name: str = "x", value: object = DATUM) -> tuple[Session, Array]:
    session = Session(ToyBackend())
    return session, source(session, name, value)


def nodes_of(session: Session, *outputs: Array) -> list[dict[str, Any]]:
    """Every node of the un-optimized serialized IR, in record order (ids are 1:1 there)."""
    graph = graphed.core.GraphStore.deserialize(session.serialized_ir(*outputs, optimize=False))
    return list(graph.nodes())


def fn_params(session: Session, *outputs: Array) -> list[str]:
    """The `fn` param of every External node reaching `outputs`, in record order."""
    return [str(node["params"]["fn"]) for node in nodes_of(session, *outputs) if node["kind"] == "external"]


# ---- X: two sessions whose node ids COLLIDE ---------------------------------------------------
def colliding_sessions() -> tuple[Session, Array, Session, Array]:
    """Two sessions built to the same SHAPE and different content: `beta.node_id == alpha.node_id`,
    so handing `beta` to `alpha`'s session addresses a real node of it — a silent answer, not a
    KeyError. The op names differ, so which store answered is observable."""
    own, x = toy_session("x")
    alpha = own.record_op("alpha", [x])
    other, y = toy_session("y", DATUM * 10)
    beta = other.record_op("beta", [y])
    return own, alpha, other, beta


def walk_handlers() -> dict[str, Callable[..., object]]:
    """`Session.walk`'s three required handlers, each loud: a guarded walk must refuse before any
    of them is reached."""

    def never(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a walk of a foreign array must refuse before evaluating a node")

    return {"source": never, "op": never, "external": never}


# ---- O: callables that derive the same name ----------------------------------------------------
def scaler(factor: Any) -> Callable[[Any], Any]:
    """Two closures of ONE factory: distinct objects, one code object, one `__name__`."""

    def bump(value: Any) -> Any:
        return value * factor

    return bump


def plus_one(value: Any) -> Any:
    return value + 1


def times_hundred(value: Any) -> Any:
    return value * 100


def numpy_session() -> tuple[Session, Any]:
    """The `apply_gufunc` surface's operand: one 1-D column, so `"()->()"` binds no core dim."""
    session = Session(NumpyBackend())
    return session, graphed.numpy.from_array(session, "x", VECTOR)


# ---- V / E: a third-party verb recording its own External family -------------------------------
THIRD_PARTY = PayloadDescriptor(
    kind="m60",
    content_hash="sha256:m60-third-party",
    framework="m60",
    version="1",
    io_schema="array",
    preprocessing_ref=None,
)


def _tenfold(value: Any) -> Any:
    return value * 10.0


def third_party_verb(array: Array) -> Array:
    """The M23 seam (`descriptor=` + `form=`), so this verb records on ANY backend."""
    session = array.session
    return session.record_external(
        "m60tag", _tenfold, [array], {"fn": "m60tag"}, descriptor=THIRD_PARTY, form=session.form(array)
    )


#: the vector the numpy-backed universes are built over
VECTOR = np.array([1.0, 2.0, 4.0])
#: the labels `varied_vector` fans out over, in §2.4 union order
VARIED_LABELS = ("nominal", "jes_up", "jes_down")


def varied_vector() -> tuple[Session, Any, Array]:
    """A three-universe `Varied` over one numpy-backed column, plus the unvaried column itself."""
    session = Session(NumpyBackend())
    record = from_record(session, "ev", pt=VECTOR)
    pt = record["pt"]
    context = EventContext(session, pt, collections={"pt": pt})
    shifted = graphed.vary(context, "jes", collections={"pt": {"up": pt * 2.0, "down": pt * 0.5}})
    return session, shifted["pt"], pt
