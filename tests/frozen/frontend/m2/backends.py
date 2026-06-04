"""Two toy backends used to prove backend-independence (no numpy dependency in graphed itself).

Both produce identical recorded graph structure; only their forms/evaluation differ.
"""

from __future__ import annotations

import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from graphed_core import PayloadDescriptor

from graphed import Array, Session

_NUMERIC = {"int", "float"}


@dataclass(frozen=True)
class ListForm:
    kind: str

    def describe(self) -> str:
        return self.kind


def _infer_kind(data: list[object]) -> str:
    if data and all(isinstance(x, bool) for x in data):
        return "bool"
    if any(isinstance(x, float) for x in data):
        return "float"
    return "int"


class ListBackend:
    """Evaluates over plain Python lists."""

    negate = False

    def op_form(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> ListForm:
        forms = [f for f in inputs if isinstance(f, ListForm)]
        if op in {"add", "sub", "mul", "div"}:
            a, b = forms
            if a.kind not in _NUMERIC or b.kind not in _NUMERIC:
                raise TypeError(f"{op} needs numeric operands, got {a.kind} and {b.kind}")
            return ListForm("float" if "float" in (a.kind, b.kind) else "int")
        if op == "filter":
            data, mask = forms
            if mask.kind != "bool":
                raise TypeError(f"filter mask must be bool, got {mask.kind}")
            return ListForm(data.kind)
        if op == "sum":
            (a,) = forms
            if a.kind not in _NUMERIC:
                raise TypeError(f"sum needs numeric, got {a.kind}")
            return ListForm("scalar")
        if op == "map":
            return ListForm("object")
        raise TypeError(f"unsupported op {op!r}")

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        sign = -1 if self.negate else 1
        if op in {"add", "sub", "mul"}:
            left, right = list(inputs[0]), list(inputs[1])
            fn = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b}[op]
            return [sign * fn(a, b) for a, b in zip(left, right, strict=False)]
        if op == "sum":
            return sign * sum(inputs[0])
        if op == "filter":
            data, mask = list(inputs[0]), list(inputs[1])
            return [x for x, m in zip(data, mask, strict=False) if m]
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
            content_hash=f"unhashed-opaque:{params.get('fn', 'lambda')}",
            framework="python",
            version=platform.python_version(),
            io_schema="opaque->opaque",
            preprocessing_ref=None,
        )


class NegListBackend(ListBackend):
    """Same structure and forms as ListBackend, but evaluation is negated (different results)."""

    negate = True


def from_list(session: Session, name: str, data: list[object]) -> Array:
    return session.source(name, form=ListForm(_infer_kind(data)), data=list(data))
