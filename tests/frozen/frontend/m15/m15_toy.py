"""Toy backend + counting loader for the M15 parquet-base suite (no numpy/awkward semantics)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

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


@dataclass
class CountingLoader:
    """A whole-dataset loader that counts its calls — the laziness witness."""

    value: object
    calls: list[int] = field(default_factory=list)

    def __call__(self) -> object:
        self.calls.append(1)
        return self.value


def session() -> Session:
    return Session(ToyBackend())


__all__ = ["Array", "CountingLoader", "ToyBackend", "ToyForm", "session"]
