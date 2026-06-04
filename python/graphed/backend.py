"""Backend and Form protocols (plan M2).

The IR (in graphed-core) is backend-agnostic: it records op names, params, and input edges, never a
backend's data or form types. A `Backend` supplies type/shape inference (`op_form`), evaluation
(`eval_stage`), the boundary-op set, column projection (M5; a stub here), and the reproducibility
metadata for any external/opaque op it emits (`external_payload`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from graphed_core import PayloadDescriptor

# Param values are scalars the graphed-core store can intern (must match graphed_core's ParamValue).
ParamValue = int | float | bool | str


@runtime_checkable
class Form(Protocol):
    """An opaque per-array type/shape token. graphed core never inspects its internals."""

    def describe(self) -> str:
        """Human-readable summary (used in diagnostics)."""
        ...


@runtime_checkable
class Backend(Protocol):
    """The seam between the backend-agnostic IR and a concrete array library (plan M2/A.4)."""

    def op_form(self, op: str, inputs: Sequence[Form], params: Mapping[str, object]) -> Form:
        """Infer the result form of `op`. Raise a TypeError-like error if the op is ill-typed."""
        ...

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        """Evaluate one stage. M2 has no fusion, so a stage is a single op."""
        ...

    def boundary_ops(self) -> frozenset[str]:
        """Op names that are boundaries (source | reduction | repartition | materialize | external)."""
        ...

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        """Column projection hook (M5). A no-op stub at M2."""
        ...

    def external_payload(self, op: str, params: Mapping[str, object]) -> PayloadDescriptor | None:
        """Reproducibility metadata for an external/opaque op, or None. Flag opaque callables as a
        preservation risk (plan A.3.1)."""
        ...
