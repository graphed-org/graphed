"""Toy backend for the M14 multi-input external surface (data-carrying sources)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import graphed_core
from graphed_core import PayloadDescriptor

from graphed import Array, Session


@dataclass(frozen=True)
class ToyForm:
    tag: str

    def describe(self) -> str:
        return self.tag


class ToyBackend:
    def op_form(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> ToyForm:
        return ToyForm(op)

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        return (op, list(inputs), dict(params))

    def boundary_ops(self) -> frozenset[str]:
        return frozenset({"source", "map"})

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        return used

    def external_payload(self, op: str, params: Mapping[str, object]) -> PayloadDescriptor | None:
        if op != "map":
            return None
        return PayloadDescriptor(
            kind="opaque_callable",
            content_hash=f"unhashed-opaque:{params.get('fn', 'lambda')}",
            framework="python",
            version="toy",
            io_schema="opaque->opaque",
            preprocessing_ref=None,
        )


def source(s: Session, name: str, value: object) -> Array:
    return s.source(name, form=ToyForm("source"), data=value)


def recorded(s: Session, arr: Array) -> dict[str, object]:
    g = graphed_core.GraphStore.deserialize(s.serialized_ir(arr, optimize=False))
    (node,) = [n for n in g.nodes() if n["id"] == arr.node_id]
    return node
