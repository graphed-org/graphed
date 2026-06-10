"""The build session: records a deferred program into the graphed-core store via a backend.

A `Session` owns one `graphed_core.GraphStore`, the chosen `Backend`, and the side tables mapping
each interned `NodeId` to its `Form` and its concrete inputs (sources / opaque callables) for
evaluation. The recorded graph is backend-independent; only forms and evaluation depend on the
backend.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import graphed_core

from .array import Array
from .backend import Backend, Form, ParamValue
from .errors import GraphedTypeError
from .provenance import Provenance, capture


class Session:
    def __init__(self, backend: Backend, *, incremental: bool = False) -> None:
        self._store = graphed_core.GraphStore()
        self._backend = backend
        self._forms: dict[int, Form] = {}
        self._sources: dict[int, object] = {}
        self._source_names: dict[int, str] = {}
        self._ops: dict[int, tuple[str, dict[str, ParamValue], list[int]]] = {}
        self._externals: dict[int, tuple[Callable[[object], object], list[int]]] = {}
        self._provenance: dict[int, Provenance] = {}
        # M10 (plan A.1): with incremental=True the session maintains the reduced view AS THE
        # GRAPH IS BUILT — every record steps an IncrementalReducer whose per-step work is the
        # delta, so compile time never pays a whole-history optimization.
        self._reducer = graphed_core.IncrementalReducer() if incremental else None
        # M11 factorization: the backend may supply its own Array proxy subclass (its idiomatic
        # user surface — e.g. graphed_numpy.NumpyArray's method/property style). The base Array
        # stays backend-idiom-neutral; backends without array_type get it unchanged.
        factory = getattr(backend, "array_type", None)
        self._array_cls: type[Array] = factory() if callable(factory) else Array

    def _step_reducer(self) -> None:
        if self._reducer is not None:
            self._reducer.step(self._store)

    # ---- introspection ---------------------------------------------------------
    @property
    def backend(self) -> Backend:
        return self._backend

    def node_count(self) -> int:
        return self._store.node_count()

    def to_dot(self) -> str:
        return self._store.to_dot()

    def serialized_ir(self, *outputs: Array, optimize: bool = True) -> bytes:
        """The recorded analysis as a canonical, versioned, byte-identical durable IR (plan M8).

        Pass the result ``Array``(s) of the analysis as ``outputs``: the artifact carries EXACTLY
        that output set — outputs are a property of THIS request, not session state (M22), so
        sequential calls with different outputs are fully independent
        (byte-identical to fresh-session serializations). With ``optimize=True`` (the default) the
        graph is reduced by the M4 optimizer (DCE + CSE + equality-saturation stage fusion), so
        the bytes carry the **optimized interned graph** — the same content-addressed artifact an
        executor runs. This is the "compile" step for a ``graphed_core.DurablePlan`` deployment:
        record an analysis once, serialize it once, then re-target it at many datasets with
        ``DurablePlan.with_partitions`` / ``for_datasets``.
        """
        if optimize and not outputs:
            raise ValueError("serialized_ir(optimize=True) needs at least one output Array")
        ids = [arr.node_id for arr in outputs]
        for arr in outputs:
            # legacy side effect, kept for back-compat (the frozen m8 suite pins that a later
            # marks-path reduce sees these outputs); the BYTES above ignore marks entirely
            self._store.mark_output(arr.node_id)
        if not optimize:
            return bytes(self._store.serialize(outputs=ids))
        if self._reducer is not None:
            # incremental session (M10): finish from the maintained canonical view — one linear
            # pass over the concise form instead of a whole-history optimization.
            return bytes(self._reducer.finalize(self._store, outputs=ids)[0].serialize())
        return bytes(self._store.reduce(outputs=ids)[0].serialize())

    def form(self, array: Array) -> Form:
        return self._forms[array.node_id]

    def provenance(self, array: Array) -> Provenance:
        return self._provenance[array.node_id]

    def source_ids(self) -> list[int]:
        """The node ids of all source nodes (used by projection)."""
        return list(self._sources)

    def sources(self) -> dict[int, object]:
        """The source nodes' concrete data objects, keyed by node id (a public, read-only view —
        host-reader integrations inspect this instead of reaching into session internals)."""
        return dict(self._sources)

    def reduction_state(self) -> dict[str, int] | None:
        """Incremental-reduction introspection (M10): how many nodes the maintained reduced view
        has consumed (`watermark`), cumulative reducer work (`total_work` — equals `watermark`, the
        incrementality witness), and the canonical size. ``None`` for a non-incremental session."""
        if self._reducer is None:
            return None
        return {
            "watermark": self._reducer.watermark(),
            "total_work": self._reducer.total_work(),
            "canonical_count": self._reducer.canonical_count(),
        }

    def source_name(self, node_id: int) -> str:
        return self._source_names[node_id]

    def form_of(self, node_id: int) -> Form:
        return self._forms[node_id]

    def sourcemap(self) -> dict[int, dict[str, object]]:
        """Per-node user-source provenance (file / line / function / sub-expression text) — the M6
        sourcemap. M9's ``inspect`` renders it so every preserved node maps back to the analysis line
        that created it, without executing anything."""
        return {
            nid: {
                "filename": p.filename,
                "lineno": p.lineno,
                "function": p.function,
                "source": p.source,
            }
            for nid, p in self._provenance.items()
        }

    def source_value(self, node_id: int) -> object:
        """The concrete data for a source node (resolving a lazy loader). Used by debug execution."""
        value = self._sources[node_id]
        return value() if callable(value) else value

    # ---- builders --------------------------------------------------------------
    def source(self, name: str, *, form: Form, data: object, **params: ParamValue) -> Array:
        node_id = self._store.add_source(name, dict(params))
        self._forms.setdefault(node_id, form)
        self._sources.setdefault(node_id, data)
        self._source_names.setdefault(node_id, name)
        self._provenance.setdefault(node_id, capture())
        self._step_reducer()
        return self._array_cls(self, node_id)

    def record_op(
        self,
        op: str,
        inputs: Sequence[Array],
        params: Mapping[str, ParamValue] | None = None,
        *,
        reduction: bool = False,
    ) -> Array:
        params_d: dict[str, ParamValue] = dict(params or {})
        prov = capture()
        in_forms = [self._forms[a.node_id] for a in inputs]
        try:
            form = self._backend.op_form(op, in_forms, params_d)
        except GraphedTypeError:
            raise
        except Exception as exc:  # backend type/shape error -> user-located error
            raise GraphedTypeError(op, prov, str(exc)) from exc
        ids = [a.node_id for a in inputs]
        if reduction:
            node_id = self._store.add_reduction(op, ids, params_d)
        else:
            node_id = self._store.add_op(op, ids, params_d)
        self._forms.setdefault(node_id, form)
        self._ops.setdefault(node_id, (op, params_d, ids))
        self._provenance.setdefault(node_id, prov)
        self._step_reducer()
        return self._array_cls(self, node_id)

    def record_external(
        self,
        op: str,
        fn: Callable[[object], object],
        inputs: Sequence[Array],
        params: Mapping[str, ParamValue] | None = None,
    ) -> Array:
        params_d: dict[str, ParamValue] = dict(params or {})
        prov = capture()
        descriptor = self._backend.external_payload(op, params_d)
        if descriptor is None:
            raise GraphedTypeError(op, prov, "backend returned no payload descriptor for external op")
        in_forms = [self._forms[a.node_id] for a in inputs]
        try:
            form = self._backend.op_form(op, in_forms, params_d)
        except GraphedTypeError:
            raise
        except Exception as exc:  # backend type/shape error -> user-located error (as record_op)
            raise GraphedTypeError(op, prov, str(exc)) from exc
        ids = [a.node_id for a in inputs]
        node_id = self._store.add_external(descriptor, ids, params_d)
        self._forms.setdefault(node_id, form)
        self._externals.setdefault(node_id, (fn, ids))
        self._provenance.setdefault(node_id, prov)
        self._step_reducer()
        return self._array_cls(self, node_id)

    # ---- generic graph walk (shared by materialize + projection) ----------------
    def walk(
        self,
        array: Array,
        *,
        source: Callable[[int], object],
        op: Callable[[int, str, list[object], Mapping[str, ParamValue]], object],
        external: Callable[[int, Callable[..., object], list[object]], object],
    ) -> object:
        """Evaluate the graph from ``array`` with caller-supplied handlers for sources, ops, and
        externals. `materialize` evaluates real data; projection evaluates reporting tracers."""
        cache: dict[int, object] = {}

        def ev(node_id: int) -> object:
            if node_id in cache:
                return cache[node_id]
            if node_id in self._sources:
                value = source(node_id)
            elif node_id in self._externals:
                fn, ids = self._externals[node_id]
                value = external(node_id, fn, [ev(i) for i in ids])
            else:
                op_name, params, ids = self._ops[node_id]
                value = op(node_id, op_name, [ev(i) for i in ids], params)
            cache[node_id] = value
            return value

        return ev(array.node_id)

    # ---- evaluation (reference, node-by-node; the real executor is M7) ----------
    def materialize(self, array: Array) -> object:
        def source(node_id: int) -> object:
            value = self._sources[node_id]
            return value() if callable(value) else value  # lazy loader (e.g. parquet)

        return self.walk(
            array,
            source=source,
            op=lambda _nid, name, ins, params: self._backend.eval_stage(name, ins, params),
            external=lambda _nid, fn, ins: fn(*ins),
        )
