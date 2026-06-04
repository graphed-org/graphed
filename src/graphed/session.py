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
    def __init__(self, backend: Backend) -> None:
        self._store = graphed_core.GraphStore()
        self._backend = backend
        self._forms: dict[int, Form] = {}
        self._sources: dict[int, object] = {}
        self._ops: dict[int, tuple[str, dict[str, ParamValue], list[int]]] = {}
        self._externals: dict[int, tuple[Callable[[object], object], list[int]]] = {}
        self._provenance: dict[int, Provenance] = {}

    # ---- introspection ---------------------------------------------------------
    @property
    def backend(self) -> Backend:
        return self._backend

    def node_count(self) -> int:
        return self._store.node_count()

    def to_dot(self) -> str:
        return self._store.to_dot()

    def form(self, array: Array) -> Form:
        return self._forms[array.node_id]

    def provenance(self, array: Array) -> Provenance:
        return self._provenance[array.node_id]

    # ---- builders --------------------------------------------------------------
    def source(self, name: str, *, form: Form, data: object, **params: ParamValue) -> Array:
        node_id = self._store.add_source(name, dict(params))
        self._forms.setdefault(node_id, form)
        self._sources.setdefault(node_id, data)
        self._provenance.setdefault(node_id, capture())
        return Array(self, node_id)

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
        return Array(self, node_id)

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
        form = self._backend.op_form(op, in_forms, params_d)
        ids = [a.node_id for a in inputs]
        node_id = self._store.add_external(descriptor, ids, params_d)
        self._forms.setdefault(node_id, form)
        self._externals.setdefault(node_id, (fn, ids))
        self._provenance.setdefault(node_id, prov)
        return Array(self, node_id)

    # ---- evaluation (reference, node-by-node; the real executor is M7) ----------
    def materialize(self, array: Array) -> object:
        cache: dict[int, object] = {}

        def ev(node_id: int) -> object:
            if node_id in cache:
                return cache[node_id]
            if node_id in self._sources:
                value = self._sources[node_id]
            elif node_id in self._externals:
                fn, ids = self._externals[node_id]
                value = fn(*[ev(i) for i in ids])
            else:
                op, params, ids = self._ops[node_id]
                value = self._backend.eval_stage(op, [ev(i) for i in ids], params)
            cache[node_id] = value
            return value

        return ev(array.node_id)
