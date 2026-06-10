"""Toy backends/forms for the M11 elementwise-parity surface (no numpy semantics needed)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from graphed_core import PayloadDescriptor

from graphed import Array, Session


@dataclass(frozen=True)
class ToyForm:
    """A bare form: no array metadata — attribute access must keep recording field ops (M3)."""

    tag: str

    def describe(self) -> str:
        return self.tag


@dataclass(frozen=True)
class MetaForm:
    """A form that carries array metadata, the way M11 backend forms (NumpyForm) do."""

    tag: str
    shape: tuple[int | None, ...]
    dtype: str
    ndim: int

    def describe(self) -> str:
        return self.tag


class ToyBackend:
    def op_form(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> ToyForm:
        return ToyForm(op)

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        return (op, list(inputs), dict(params))

    def boundary_ops(self) -> frozenset[str]:
        return frozenset({"source", "sum"})

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        return used

    def external_payload(self, op: str, params: Mapping[str, object]) -> PayloadDescriptor | None:
        return PayloadDescriptor(
            kind="toy", content_hash="h", framework="f", version="v", io_schema="s", preprocessing_ref=None
        )


def source(s: Session, name: str) -> Array:
    return s.source(name, form=ToyForm("source"), data=None)


def meta_source(
    s: Session, name: str, *, shape: tuple[int | None, ...] = (None, 3), dtype: str = "float64"
) -> Array:
    return s.source(name, form=MetaForm("source", shape, dtype, len(shape)), data=None)
