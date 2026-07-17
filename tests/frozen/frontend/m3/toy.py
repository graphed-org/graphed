"""A permissive toy backend for exercising the M3 Array surface without numpy/awkward."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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
        return frozenset({"source"})

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        return used

    def external_payload(self, op: str, params: Mapping[str, object]) -> PayloadDescriptor | None:
        return PayloadDescriptor(
            kind="toy", content_hash="h", framework="f", version="v", io_schema="s", preprocessing_ref=None
        )


def source(s: Session, name: str) -> Array:
    return s.source(name, form=ToyForm("source"), data=None)
