"""Toy backend/forms for the M17 record-subset getitem (no numpy semantics needed).

Per the M11 factorization, the base ``graphed.Array`` exposes no reduction METHODS — backends
build their idiomatic surface from the protected ``_reduction``/``_scan``/``_norm_axis``
infrastructure. ``ReduceArray`` is the minimal such surface, used to pin the infrastructure's
recording semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import graphed_core
from graphed_core import PayloadDescriptor

from graphed import Array, Session


@dataclass(frozen=True)
class ToyForm:
    """A bare form: no ndim — negative axes cannot be normalized against it."""

    tag: str

    def describe(self) -> str:
        return self.tag


@dataclass(frozen=True)
class MetaForm:
    tag: str
    shape: tuple[int | None, ...]
    dtype: str
    ndim: int

    def describe(self) -> str:
        return self.tag


class ReduceArray(Array):
    """A backend's idiomatic reduction surface, built ONLY from the protected infrastructure."""

    __slots__ = ()

    def red(self, kind: str, axis: int | None = None, *, keepdims: bool = False, ddof: int = 0) -> Array:
        return self._reduction(kind, axis, keepdims=keepdims, ddof=ddof)

    def scan(self, kind: str, axis: int | None = None) -> Array:
        return self._scan(kind, axis)


class ToyBackend:
    def array_type(self) -> type[Array]:
        return ReduceArray

    def op_form(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> ToyForm:
        return ToyForm(op)

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        return (op, list(inputs), dict(params))

    def boundary_ops(self) -> frozenset[str]:
        return frozenset({"source"})

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        return used

    def external_payload(self, op: str, params: Mapping[str, object]) -> PayloadDescriptor | None:
        return PayloadDescriptor(
            kind="toy", content_hash="h", framework="f", version="v", io_schema="s", preprocessing_ref=None
        )


def source(s: Session, name: str) -> ReduceArray:
    arr = s.source(name, form=ToyForm("source"), data=None)
    assert isinstance(arr, ReduceArray)
    return arr


def meta_source(s: Session, name: str, *, ndim: int = 2) -> ReduceArray:
    shape: tuple[int | None, ...] = (None,) + (3,) * (ndim - 1)
    arr = s.source(name, form=MetaForm("source", shape, "float64", ndim), data=None)
    assert isinstance(arr, ReduceArray)
    return arr


def recorded(s: Session, arr: Array) -> dict[str, object]:
    """The (kind, name, params) of the node ``arr`` denotes, read back from the serialized IR."""
    g = graphed_core.GraphStore.deserialize(s.serialized_ir(arr, optimize=False))
    (node,) = [n for n in g.nodes() if n["id"] == arr.node_id]
    return node
