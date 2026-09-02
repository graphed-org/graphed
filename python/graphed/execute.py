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
from dataclasses import dataclass, field
from typing import Any

import graphed.core

from .backend import Backend
from .errors import GraphedError
from .session import Session
from .varied import refuse_container

__all__ = ["CompiledGraph", "Correspondence", "compile_ir", "evaluate_ir"]

#: A user source location as plain string/int data — `SourceFrame`'s fields, in its field order.
Frame = tuple[str, int, str, str]
#: A reduced-store address: the node id, plus the position inside a fused stage (`None` when the
#: node reduced to itself, i.e. a boundary).
Key = tuple[int, int | None]


@dataclass(frozen=True)
class Correspondence:
    """§8.2(i): where the RECORD-time graph landed in the compiled reduced store.

    The reduction re-indexes four times (DCE, canonicalization, CSE, stage fusion), so record-time
    ids address nothing in the shipped IR. Both halves are keyed for a worker that only has the
    reduced bytes.

    Deliberately NOT ``slots=True``: consumers locate ``frames`` by LAYOUT (walking ``__dict__``)
    rather than by name, since the field spelling is only pinned at the m49 freeze.
    """

    #: record node id -> where it landed. Absent for a record id DCE dropped.
    node_map: dict[int, Key]
    #: one frame per key of ``node_map``'s image, in key order. The re-keying is many-to-one, so
    #: §8.2(ii)'s tie-break applies: the LOWEST record id mapping to a key supplies the frame.
    frames: tuple[tuple[Key, Frame], ...]


@dataclass(frozen=True)
class CompiledGraph:
    """A compiled analysis: the reduced canonical IR plus the source names it reads. Picklable and
    self-contained — exactly what ships to an executor worker (no Session, no analysis function)."""

    ir: bytes
    source_names: tuple[str, ...]
    #: §2.5: registered variation labels that reach no marked output — a DIAGNOSTIC, sorted,
    #: empty when every one does. DCE already prunes the work; this is what stops a systematic
    #: being paid for at build time and quietly never filled.
    unreached_labels: tuple[str, ...] = ()
    #: §8.2(i): the record→reduced correspondence and the per-key user frames. Unconditional.
    correspondence: Correspondence = field(default_factory=lambda: Correspondence({}, ()))
    #: §2.5's shift-after-weight ordering diagnostic: sorted `(factor family, collection)` pairs,
    #: one per ambient weight factor whose cone reaches a collection varied AFTER it was
    #: registered — that factor fills every shift universe with its PRE-shift value. Empty when the
    #: order is sound. Detected at RECORD time (`context._vary_shift`); neither operand survives to
    #: here, so the compile-time walk the unreached-label diagnostic uses cannot see it.
    shift_after_weight: tuple[tuple[str, str], ...] = ()

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
    refuse_container("graphed.compile_ir", *outputs)
    if maximal_fusion and not optimize:
        raise ValueError("maximal_fusion requires optimize=True")
    if optimize and not outputs:
        raise ValueError("compile_ir(optimize=True) needs at least one output Array")
    ids = [arr.node_id for arr in outputs]
    if not optimize:
        blob = bytes(session._store.serialize(outputs=ids))
        # opt_level=0 is 1:1 (M6): the serialized arena keeps the record ids it was built with
        landings: list[Key | None] = [(nid, None) for nid in range(session._store.node_count())]
    else:
        reduced = (
            session._reducer.finalize(session._store, maximal_fusion=maximal_fusion, outputs=ids)[0]
            if session._reducer is not None
            else session._store.reduce(maximal_fusion=maximal_fusion, outputs=ids)[0]
        )
        blob, landings = bytes(reduced.serialize()), reduced.node_map()
    node_map = {nid: landed for nid, landed in enumerate(landings) if landed is not None}
    names = tuple(session.source_name(nid) for nid in session.source_ids())
    reached: set[str] = set()
    for arr in outputs:
        reached |= getattr(arr, "_labels", None) or frozenset()
    registered = {label for labels, _ref in session._varied for label in labels}
    return CompiledGraph(
        ir=blob,
        source_names=names,
        unreached_labels=tuple(sorted(registered - reached)),
        correspondence=Correspondence(node_map=node_map, frames=_frames_by_key(session, node_map)),
        shift_after_weight=tuple(sorted(session._shift_after_weight)),
    )


def _frames_by_key(session: Session, node_map: dict[int, Key]) -> tuple[tuple[Key, Frame], ...]:
    """Re-key ``Session._provenance`` onto §8.2(i)'s keys, in key order.

    Many-to-one: the reducer merges record ids recorded at different user lines onto one key, even
    inside a stage where ``member_index`` cannot separate them. §8.2(ii) binds the tie-break to the
    LOWEST record id, matching the driver-side ``setdefault`` house rule and making the shipped
    frame a function of the graph rather than of dict order.
    """
    chosen: dict[Key, Frame] = {}
    for nid in sorted(node_map):
        prov = session._provenance.get(nid)
        if prov is None or node_map[nid] in chosen:
            continue
        chosen[node_map[nid]] = (prov.filename, prov.lineno, prov.function, prov.source)
    return tuple(sorted(chosen.items(), key=lambda e: (e[0][0], -1 if e[0][1] is None else e[0][1])))


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
