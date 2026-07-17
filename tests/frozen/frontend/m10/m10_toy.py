"""A self-contained toy backend for the M10 frozen suite (plain Python lists, no numpy).

Counts every `eval_stage` dispatch so the suite can pin the M10 claim: IR-driven evaluation costs
one dispatch per REDUCED op, not one per recorded op.
"""

from __future__ import annotations

import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from graphed import Array, Session
from graphed.core import PayloadDescriptor

_NUMERIC = {"int", "float"}


@dataclass(frozen=True)
class ListForm:
    kind: str

    def describe(self) -> str:
        return self.kind


class CountingListBackend:
    """Evaluates over plain Python lists; `calls` records every eval_stage dispatch."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def op_form(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> ListForm:
        forms = [f for f in inputs if isinstance(f, ListForm)]
        if op in {"add", "mul"} and "scalar" in params:
            (a,) = forms
            return ListForm(a.kind)
        if op in {"add", "sub", "mul"}:
            a, b = forms
            if a.kind not in _NUMERIC or b.kind not in _NUMERIC:
                raise TypeError(f"{op} needs numeric operands")
            return ListForm("float" if "float" in (a.kind, b.kind) else "int")
        if op == "sum":
            return ListForm("scalar")
        if op == "map":
            return ListForm("object")
        raise TypeError(f"unsupported op {op!r}")

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        self.calls.append(op)
        if op in {"add", "mul"} and "scalar" in params:
            s = params["scalar"]
            assert isinstance(s, float)
            fn: Callable[[float], float] = (lambda x: x + s) if op == "add" else (lambda x: x * s)
            seq = inputs[0]
            assert isinstance(seq, list)
            return [fn(x) for x in seq]
        if op in {"add", "sub", "mul"}:
            left, right = list(inputs[0]), list(inputs[1])  # type: ignore[call-overload]
            table: dict[str, Callable[[float, float], float]] = {
                "add": lambda a, b: a + b,
                "sub": lambda a, b: a - b,
                "mul": lambda a, b: a * b,
            }
            fn2 = table[op]
            return [fn2(a, b) for a, b in zip(left, right, strict=False)]
        if op == "sum":
            seq = inputs[0]
            assert isinstance(seq, list)
            return sum(seq)
        raise TypeError(f"unsupported op {op!r}")

    def boundary_ops(self) -> frozenset[str]:
        return frozenset({"source", "sum", "map"})

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        return used

    def external_payload(self, op: str, params: Mapping[str, object]) -> PayloadDescriptor | None:
        if op != "map":
            return None
        return PayloadDescriptor(
            kind="opaque_callable",
            content_hash=f"toy-opaque:{params.get('fn', 'lambda')}",
            framework="python",
            version=platform.python_version(),
            io_schema="opaque->opaque",
            preprocessing_ref=None,
        )


def from_list(session: Session, name: str, data: list[float]) -> Array:
    kind = "float" if any(isinstance(x, float) for x in data) else "int"
    return session.source(name, form=ListForm(kind), data=list(data))
