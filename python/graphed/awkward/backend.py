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
from graphed.core import PayloadDescriptor

from . import join, payloads, shuffle
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
    #: the backend's versioned shuffle-format token (folded into the V2 task ids, §7.2)
    identity = "graphed-awkward/0"

    def __init__(self, behavior: Mapping[str, object] | None = None) -> None:
        # M18: a registered behavior dict (e.g. vector's) makes behavior PROPERTIES work through
        # plain attribute access — on the typetracer at record time and on real arrays in eval.
        self._behavior = dict(behavior) if behavior else None

    def op_form(self, op: str, inputs: Sequence[AwkwardForm], params: Mapping[str, object]) -> AwkwardForm:
        if op == "exchange":
            return inputs[0]  # a pure data-movement boundary is identity on the payload form (§3.3a)
        if op == "join":
            # M40 (§3.3): flat relational record-merge form; how=left/outer ⇒ missing side option-typed
            return AwkwardForm(join.join_form([f.tt for f in inputs], params))
        if op in _EXTERNAL:
            # Opaque/external op: output form is not derivable from inputs. Approximate it by the
            # first input's form (corrections/inference are ~shape-preserving for these fixtures).
            return inputs[0]
        operands = [f.tt for f in inputs]
        return AwkwardForm(apply(op, operands, params, behavior=self._behavior))

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        if op == "join":  # needs `self` (the shared kernel routes through JoinBackend primitives)
            return self._eval_join(inputs, params)
        return apply(op, inputs, params, behavior=self._behavior)

    def _eval_join(self, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        left, right = inputs[0], inputs[1]
        on = join.on_from_params(params)
        how = str(params.get("how", "inner"))
        if ak.backend(left) == "typetracer":  # projection replay: structural merge, no matching/data read
            return join.merge_records(left, right, on=on)
        if bool(params.get("grouped", False)):  # gak.join(grouped=True): awkward-only regroup post-op
            return join.join_grouped(left, right, on=on, how=how)
        # the shared radix-hash kernel (JoinBackend prims only); local import avoids an import cycle.
        from graphed.shuffle import join_blocks  # noqa: PLC0415

        return join_blocks(self, left, right, on=on, how=how)

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

    # ---- ShuffleBackend exchange half (M39 §3.0) — thin delegates to the pure `shuffle` module ----
    def partition(
        self,
        block: Any,
        key_field: str,
        parts: int,
        *,
        salt: int = 0,
        boundaries: object = None,
    ) -> tuple[Any, ...]:
        return shuffle.partition(block, key_field, parts, salt=salt, boundaries=boundaries)

    def concat(self, blocks: Sequence[Any]) -> Any:
        return shuffle.concat(blocks)

    def slice_rows(self, block: Any, start: int, stop: int) -> Any:
        return shuffle.slice_rows(block, start, stop)

    def estimated_bytes(self, block_or_form: object) -> int:
        return shuffle.estimated_bytes(block_or_form)

    def to_wire(self, block: Any) -> bytes:
        return shuffle.to_wire(block)

    def from_wire(self, data: bytes) -> Any:
        return shuffle.from_wire(data)

    # ---- JoinBackend join half (M40 §3.3) — thin delegates to the pure `join` module ----
    def match_indices(
        self, build: Any, probe: Any, *, on: Sequence[str], how: str = "inner"
    ) -> tuple[Any, Any]:
        return join.match_indices(build, probe, on=on, how=how)

    def take(self, block: Any, index: Any) -> Any:
        return join.take(block, index)

    def merge_records(self, left: Any, right: Any, *, on: Sequence[str]) -> Any:
        return join.merge_records(left, right, on=on)


def _typetracer(array: ak.Array) -> ak.Array:
    return ak.Array(ak.Array(array).layout.to_typetracer(forget_length=True))


def from_awkward(session: Session, name: str, array: object, **descriptor: object) -> Any:
    """Create a metadata-only source from an in-memory awkward array (form via typetracer; the real
    array is retained only for evaluation)."""
    real = ak.Array(array)
    return session.source(name, form=AwkwardForm(_typetracer(real)), data=real)


# from_parquet moved to io.py (M15): the multi-file, blind-partition specialization of the
# graphed.parquet base; the M3 single-file shape is a special case of it.
