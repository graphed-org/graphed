"""IR-driven execution (M10, plan A.1-A.3): the REDUCED serialized IR is what executes.

Before this milestone the only evaluators were `Session.materialize` (a node-by-node walk of the
un-reduced Python op log) and per-partition *re-recording* of the analysis — one interpreter
dispatch per recorded op per partition, the dask failure mode the project exists to avoid
(§A.3 #2/#6/#7). This module closes that gap:

- `compile_ir(session, *outputs)` is the compile step: mark the outputs, reduce (DCE + CSE +
  equality-saturation stage fusion), serialize. The result is a small, picklable
  :class:`CompiledGraph` — pure bytes plus the source names it needs, no Session, no user code.
- `evaluate_ir(compiled, backend, sources)` runs that artifact: deserialize once, then ONE backend
  dispatch per *reduced* node (fused stage members evaluate inline), with sources bound by name.
  A worker holds no Session and never re-records; dispatch count scales with the reduced graph,
  not the recorded history.

Opaque `External` nodes are not embedded in the IR (they are a preservation risk, plan A.3.1);
`evaluate_ir` resolves them through an explicit ``externals`` mapping keyed by payload content
hash, and fails loudly when one is missing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import graphed.core

from .backend import Backend
from .errors import GraphedError
from .session import Session

__all__ = ["CompiledGraph", "compile_ir", "evaluate_ir"]


@dataclass(frozen=True)
class CompiledGraph:
    """A compiled analysis: the reduced canonical IR plus the source names it reads. Picklable and
    self-contained — exactly what ships to an executor worker (no Session, no analysis function)."""

    ir: bytes
    source_names: tuple[str, ...]

    def evaluate(
        self,
        backend: Backend,
        sources: Mapping[str, object],
        *,
        externals: Mapping[str, Callable[..., object]] | None = None,
    ) -> list[object]:
        return evaluate_ir(self, backend, sources, externals=externals)


def compile_ir(
    session: Session,
    *outputs: Any,
    optimize: bool = True,
    maximal_fusion: bool = False,
) -> CompiledGraph:
    """Compile the session's recorded graph for the given output arrays.

    Reduction runs once, here — workers receive the already-reduced bytes. An incremental session
    finishes from its maintained canonical view (per-step work already paid at record time).
    The artifact carries EXACTLY the requested outputs (M22), so compiling different
    expressions sequentially from one session never cross-talks."""
    if maximal_fusion and not optimize:
        raise ValueError("maximal_fusion requires optimize=True")
    if optimize and not outputs:
        raise ValueError("compile_ir(optimize=True) needs at least one output Array")
    ids = [arr.node_id for arr in outputs]
    if not optimize:
        blob = bytes(session._store.serialize(outputs=ids))
    elif session._reducer is not None:
        blob = bytes(
            session._reducer.finalize(session._store, maximal_fusion=maximal_fusion, outputs=ids)[
                0
            ].serialize()
        )
    else:
        blob = bytes(session._store.reduce(maximal_fusion=maximal_fusion, outputs=ids)[0].serialize())
    names = tuple(session.source_name(nid) for nid in session.source_ids())
    return CompiledGraph(ir=blob, source_names=names)


def evaluate_ir(
    compiled: CompiledGraph | bytes,
    backend: Backend,
    sources: Mapping[str, object],
    *,
    externals: Mapping[str, Callable[..., object]] | None = None,
) -> list[object]:
    """Evaluate a compiled (reduced) IR: one backend dispatch per reduced node, fused stage members
    inline. ``sources`` binds each source name to its data (or a zero-arg loader); ``externals``
    binds each External payload's ``content_hash`` to its evaluator. Returns the outputs in mark
    order."""
    blob = compiled.ir if isinstance(compiled, CompiledGraph) else compiled
    store = graphed.core.GraphStore.deserialize(bytes(blob))
    vals: list[object] = []
    for nd in store.nodes():
        kind = nd["kind"]
        ins = [vals[i] for i in nd["inputs"]]
        if kind == "source":
            name = nd["name"]
            if name not in sources:
                raise GraphedError(f"evaluate_ir: no data bound for source {name!r}")
            value = sources[name]
            vals.append(value() if callable(value) else value)
        elif kind in ("op", "reduction"):
            vals.append(backend.eval_stage(nd["name"], ins, nd["params"]))
        elif kind == "stage":
            mvals: list[object] = []
            for m in nd["members"]:
                mins = [ins[i] if tag == "input" else mvals[i] for tag, i in m["inputs"]]
                mvals.append(backend.eval_stage(m["name"], mins, m["params"]))
            vals.append(mvals[-1])
        elif kind == "external":
            chash = nd["descriptor"]["content_hash"]
            if externals is None or chash not in externals:
                raise GraphedError(
                    f"evaluate_ir: External payload {chash!r} needs an evaluator "
                    "(pass externals={content_hash: callable})"
                )
            vals.append(externals[chash](*ins))
        else:  # pragma: no cover - the codec only emits the kinds above
            raise GraphedError(f"evaluate_ir: unknown node kind {kind!r}")
    return [vals[o] for o in store.outputs()]
