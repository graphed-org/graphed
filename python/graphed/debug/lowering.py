"""Opt-level lowering (plan M6).

``lower(session, array, opt_level=...)`` walks the recorded graph and produces a `LoweredGraph`:

- ``opt_level == 0``: **1:1 op↔node** — every recorded op is its own single-member stage, no fusion;
  the debug runner additionally runs inter-op consistency assertions. This is the canonical mode for
  pinpointing where a value first goes wrong.
- ``opt_level >= 1``: maximal runs of ops between boundaries are **fused** into multi-member stages
  (the same boundary rule as the M4 optimizer), so the structure matches what actually executes.

Either way each op keeps its exact user provenance, so an error maps to the same analysis line
regardless of opt level. Lowering uses only the frontend's public surface (``walk`` + ``provenance``
+ ``form`` + ``backend.boundary_ops``); it does not reach into Session internals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from graphed import Array, Session

from .errors import SourceFrame


@dataclass(frozen=True)
class LoweredOp:
    node_id: int
    op: str
    kind: str  # "source" | "op" | "external"
    provenance: SourceFrame
    input_ids: tuple[int, ...]
    form: str
    boundary: bool


@dataclass(frozen=True)
class LoweredStage:
    index: int
    members: tuple[LoweredOp, ...]  # ordered; exactly one for opt_level=0

    @property
    def head(self) -> LoweredOp:
        return self.members[-1]

    @property
    def boundary(self) -> bool:
        return self.head.boundary

    @property
    def node_id(self) -> int:
        return self.head.node_id


@dataclass(frozen=True)
class LoweredGraph:
    opt_level: int
    stages: tuple[LoweredStage, ...]
    output_id: int

    @property
    def ops(self) -> tuple[LoweredOp, ...]:
        return tuple(m for s in self.stages for m in s.members)

    @property
    def one_to_one(self) -> bool:
        return all(len(s.members) == 1 for s in self.stages)

    def op_for(self, node_id: int) -> LoweredOp:
        for op in self.ops:
            if op.node_id == node_id:
                return op
        raise KeyError(node_id)


def _form_str(form: object) -> str:
    describe = getattr(form, "describe", None)
    return describe() if callable(describe) else str(form)


def _collect(session: Session, array: Array) -> list[LoweredOp]:
    """Topologically-ordered ops (inputs before uses) via the public graph walk."""
    boundary_ops = session.backend.boundary_ops()
    ops: list[LoweredOp] = []

    def frame(nid: int) -> SourceFrame:
        prov = session.provenance(Array(session, nid))
        return SourceFrame(prov.filename, prov.lineno, prov.function, prov.source)

    def form(nid: int) -> str:
        return _form_str(session.form(Array(session, nid)))

    def on_source(nid: int) -> int:
        # a source's "op" is its name, so visualization/projection can key off it
        ops.append(LoweredOp(nid, session.source_name(nid), "source", frame(nid), (), form(nid), True))
        return nid

    def on_op(nid: int, name: str, ins: list[object], _params: Mapping[str, object]) -> int:
        ids = tuple(cast(int, i) for i in ins)
        ops.append(LoweredOp(nid, name, "op", frame(nid), ids, form(nid), name in boundary_ops))
        return nid

    def on_external(nid: int, _fn: object, ins: list[object]) -> int:
        ids = tuple(cast(int, i) for i in ins)
        ops.append(LoweredOp(nid, "external", "external", frame(nid), ids, form(nid), True))
        return nid

    session.walk(array, source=on_source, op=on_op, external=on_external)
    return ops


def _fuse(ops: list[LoweredOp]) -> list[LoweredStage]:
    """Group maximal runs of non-boundary ops that chain (single use) into one stage. Mirrors the M4
    boundary rule so the lowered structure matches what executes."""
    by_id = {op.node_id: op for op in ops}
    uses: dict[int, int] = dict.fromkeys(by_id, 0)
    for op in ops:
        for i in op.input_ids:
            uses[i] = uses.get(i, 0) + 1

    # union-find: an op fuses into its single consumer iff both are non-boundary ops
    parent = {op.node_id: op.node_id for op in ops}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    consumers: dict[int, list[int]] = {op.node_id: [] for op in ops}
    for op in ops:
        for i in op.input_ids:
            consumers[i].append(op.node_id)
    for op in ops:
        if op.boundary:
            continue
        cs = consumers[op.node_id]
        if len(cs) == 1 and not by_id[cs[0]].boundary:
            parent[find(op.node_id)] = find(cs[0])  # fuse into the consumer

    groups: dict[int, list[LoweredOp]] = {}
    for op in ops:  # ops already topo-ordered
        groups.setdefault(find(op.node_id), []).append(op)
    ordered = sorted(groups.values(), key=lambda g: max(m.node_id for m in g))
    return [LoweredStage(i, tuple(g)) for i, g in enumerate(ordered)]


def lower(session: Session, array: Array, *, opt_level: int = 1) -> LoweredGraph:
    """Lower the graph reaching ``array`` to a `LoweredGraph` at the given opt level."""
    if opt_level < 0:
        raise ValueError("opt_level must be >= 0")
    ops = _collect(session, array)
    one_to_one = [LoweredStage(i, (op,)) for i, op in enumerate(ops)]
    stages = one_to_one if opt_level == 0 else _fuse(ops)
    return LoweredGraph(opt_level=opt_level, stages=tuple(stages), output_id=array.node_id)
