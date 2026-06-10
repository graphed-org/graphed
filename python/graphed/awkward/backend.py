"""AwkwardBackend: typetracer form inference + real-array evaluation (plan M3).

`op_form` runs ops on **typetracer** arrays (metadata only — no event data is read), `eval_stage`
runs the same ops on real arrays. Both go through the single `apply` dispatch in `_ops`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import awkward as ak
from graphed import Session
from graphed_core import PayloadDescriptor

from . import payloads
from ._ops import apply


@dataclass(eq=False)
class AwkwardForm:
    """Opaque form backed by a metadata-only typetracer array (implements graphed.Form)."""

    tt: ak.Array

    def describe(self) -> str:
        return str(self.tt.type)

    @property
    def is_typetracer(self) -> bool:
        return ak.backend(self.tt) == "typetracer"


_BOUNDARY = frozenset(
    {"source", "external", "correction", "onnx", "map", "ak.sum", "ak.any", "ak.all", "ak.count"}
)


_EXTERNAL = frozenset({"map", "correction", "onnx", "external"})


class AwkwardBackend:
    def __init__(self, behavior: Mapping[str, object] | None = None) -> None:
        # M18: a registered behavior dict (e.g. vector's) makes behavior PROPERTIES work through
        # plain attribute access — on the typetracer at record time and on real arrays in eval.
        self._behavior = dict(behavior) if behavior else None

    def op_form(self, op: str, inputs: Sequence[AwkwardForm], params: Mapping[str, object]) -> AwkwardForm:
        if op in _EXTERNAL:
            # Opaque/external op: output form is not derivable from inputs. Approximate it by the
            # first input's form (corrections/inference are ~shape-preserving for these fixtures).
            return inputs[0]
        operands = [f.tt for f in inputs]
        return AwkwardForm(apply(op, operands, params, behavior=self._behavior))

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        return apply(op, inputs, params, behavior=self._behavior)

    def boundary_ops(self) -> frozenset[str]:
        return _BOUNDARY

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        return used  # M5

    def external_payload(self, op: str, params: Mapping[str, object]) -> PayloadDescriptor | None:
        if op == "correction":
            return payloads.correctionlib_descriptor(str(params["path"]), str(params["name"]))
        if op == "onnx":
            return payloads.onnx_descriptor(str(params["path"]))
        if op == "map":
            return payloads.opaque_callable_descriptor(str(params.get("fn", "lambda")))
        if op == "external":
            # a generic, user-defined External: the caller supplies a pre-computed *deterministic*
            # content hash (e.g. via a graphed-preserve plugin) — the backend is just the conduit.
            return PayloadDescriptor(
                kind=str(params.get("kind", "external")),
                content_hash=str(params["content_hash"]),
                framework=str(params.get("framework", "")),
                version=str(params.get("version", "")),
                io_schema=str(params.get("io_schema", "")),
                preprocessing_ref=None,
            )
        return None


def _typetracer(array: ak.Array) -> ak.Array:
    return ak.Array(ak.Array(array).layout.to_typetracer(forget_length=True))


def from_awkward(session: Session, name: str, array: object, **descriptor: object) -> Any:
    """Create a metadata-only source from an in-memory awkward array (form via typetracer; the real
    array is retained only for evaluation)."""
    real = ak.Array(array)
    return session.source(name, form=AwkwardForm(_typetracer(real)), data=real)


# from_parquet moved to io.py (M15): the multi-file, blind-partition specialization of the
# graphed.parquet base; the M3 single-file shape is a special case of it.
