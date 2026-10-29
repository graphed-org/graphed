"""The build session: records a deferred program into the graphed-core store via a backend.

A `Session` owns one `graphed.core.GraphStore`, the chosen `Backend`, and the side tables mapping
each interned `NodeId` to its `Form` and its concrete inputs (sources / opaque callables) for
evaluation. The recorded graph is backend-independent; only forms and evaluation depend on the
backend.
"""

from __future__ import annotations

import contextvars
import functools
import threading
import types
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, TypeVar

import graphed.core

from .array import Array, attach, handle_of, labels_carried
from .backend import Backend, Form, ParamValue
from .errors import GraphedTypeError, OutputTypeError
from .provenance import Provenance, capture
from .services import Bindable
from .systematics.lineage import unify_contexts
from .systematics.points import Point
from .systematics.state import SystematicsState

if TYPE_CHECKING:
    from .services import ServiceSpec


@dataclass(frozen=True)
class _CompileView:
    """What one compile reads of a Session, taken together under its lock."""

    ir: bytes
    #: per record id, the ``(stage, member)`` key it landed on, else ``None``
    landings: Sequence[tuple[int, int | None] | None]
    source_names: tuple[str, ...]
    provenance: dict[int, Provenance]
    registered_labels: frozenset[str]
    shift_after_weight: Mapping[tuple[str, str], frozenset[int]]


@dataclass(frozen=True)
class CheckedExternal:
    """The evaluator of an External that declared its type.

    Its value must have that type, checked by the backend on every call and never cast. Picklable
    when ``fn`` is.
    """

    fn: Callable[..., object]
    #: ``"output_type"`` (the element type) or ``"output_dtype"`` (a static leaf dtype)
    key: str
    declared: str
    #: the recording backend's ``check_output_type``
    check: Callable[[object, Sequence[object], str, str], str | None]
    op: str
    provenance: Provenance
    node: str

    def __call__(self, *inputs: object) -> object:
        """Evaluate, then refuse a value that disagrees with the declared type."""
        value = self.fn(*inputs)
        actual = self.check(value, inputs, self.key, self.declared)
        if actual is not None:
            detail = f"{self.node} declares {self.key} {self.declared!r}; its value is {actual!r}"
            raise OutputTypeError(self.op, self.provenance, detail)
        return value

    def bind_services(self, endpoints: Mapping[str, str]) -> CheckedExternal:
        """Return this check around the evaluator with ``endpoints`` bound."""
        return (
            replace(self, fn=self.fn.bind_services(endpoints))
            if isinstance(self.fn, Bindable)
            else self
        )

    def __getattr__(self, name: str) -> Any:
        # the wrapped evaluator's own attributes (a plugin evaluator's `endpoint`) read through;
        # dunders and a half-unpickled instance must not recurse into `fn`
        """Read an attribute through to the wrapped evaluator."""
        if name.startswith("__") or "fn" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["fn"], name)


class Session:
    """A recording session: the interned graph of every array recorded on ``backend``.

    ``with session:`` makes it the Session array-less reads in this context record into, until the
    block exits; see :func:`current_session`.
    """

    def __init__(self, backend: Backend, *, incremental: bool = False) -> None:
        """Record on ``backend``; ``incremental`` reduces the graph as it is recorded."""
        self._store = graphed.core.GraphStore()
        self._backend = backend
        self._forms: dict[int, Form] = {}
        self._sources: dict[int, object] = {}
        self._source_names: dict[int, str] = {}
        # source name -> (form description, data, identity)
        self._slots: dict[str, tuple[object, object, object]] = {}
        self._ops: dict[int, tuple[str, dict[str, ParamValue], list[int]]] = {}
        self._externals: dict[int, tuple[Callable[[object], object], list[int]]] = {}
        self._services: dict[str, ServiceSpec] = {}
        self._provenance: dict[int, Provenance] = {}
        # M10 (plan A.1): with incremental=True the session maintains the reduced view AS THE
        # GRAPH IS BUILT — every record steps an IncrementalReducer whose per-step work is the
        # delta, so compile time never pays a whole-history optimization.
        self._reducer = graphed.core.IncrementalReducer() if incremental else None
        # M11 factorization: the backend may supply its own Array proxy subclass (its idiomatic
        # user surface — e.g. graphed.numpy.NumpyArray's method/property style). The base Array
        # stays backend-idiom-neutral; backends without array_type get it unchanged.
        factory = getattr(backend, "array_type", None)
        self._array_cls: type[Array] = factory() if callable(factory) else Array
        # `reachable_sources`' memo (node_id -> reachable source node_ids)
        self._source_ids_cache: dict[int, frozenset[int]] = {}
        # Session methods hold it around their state access and never across a callback (a backend
        # hook, a loader, an evaluator); code outside this module reads the tables only through
        # methods returning copies. Reentrant: a `vary` transaction holds it around `record_*`.
        self._lock = threading.RLock()
        #: the variation registries `graphed.vary` writes, which the Session never interprets
        self.systematics = SystematicsState(self._lock)
        # m60 integ-m60-O: per-Session identity for an opaque callable. `callable -> (fn, name)`
        # (the object is held, so an id used as a stand-in key cannot be recycled onto a later
        # callable) plus the ONE name space every name handed out lives in, declared or derived.
        self._fn_names: dict[object, tuple[object, str]] = {}
        self._fn_taken: set[str] = set()

    def __enter__(self) -> Session:
        """Open a scope: array-less reads in this context record here until it exits."""
        _scopes.set((*_scopes.get(), self))
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close this scope; the Session and what it recorded stay usable."""
        stack = _scopes.get()
        if not stack or stack[-1] is not self:
            raise RuntimeError("graphed: Session scopes exited out of order")
        _scopes.set(stack[:-1])

    def _mine(self, arrays: Sequence[Any], located: tuple[str, Provenance] | None = None) -> None:
        """m60 integ-m60-X: refuse an `Array` this Session did not record.

        A node id only means something in its own store, so a foreign `Array` whose id exists here
        would be spliced in as whatever node wears that id — a wrong answer, not an error. The
        `record_*` family passes its `(op, provenance)`, and gets a user-located
        `GraphedTypeError`; the readers pass nothing and get a plain `TypeError`
        (`GraphedTypeError` is not one, which is what separates the two halves).
        """
        if all(a.session is self for a in arrays):
            return
        message = (
            "an input was recorded in a different Session; record every input in one (read the "
            "later ones inside `with array.session:`, or pass the reader session=)"
        )
        if located is None:
            raise TypeError(f"graphed: {message}")
        raise GraphedTypeError(located[0], located[1], message)

    def _fn_name(self, fn: object, name: str | None) -> str:
        """m60 integ-m60-O: the `fn` param an opaque callable is recorded under.

        Every backend derives an External's payload hash from this param, so two DISTINCT
        callables wearing one name would intern to one node and the second one's result would be
        the first one's. Every name handed out — declared through `name=` or derived from
        `__name__` — therefore lives in ONE per-Session name space, and a derived candidate
        advances (`q`, `q#1`, `q#2`, …) until it is one nothing else answers to. An explicit
        `name=` is the caller's own identity declaration: returned untouched (equal names intern,
        which is the point) and entered in that space, so a later derivation cannot land on it.

        A declared `name=` IS the identity: declaring a name another callable already wears —
        declared or derived — means the same node, the caller having said so.

        THE IDENTITY RULE: two callable objects share a memo entry only where PYTHON ITSELF
        defines them as the same call — an object whose TYPE is `types.MethodType`, whose call IS
        `__func__(__self__, ...)`, keyed `(id(fn.__self__), id(fn.__func__))`; every other callable
        is its own identity, `id(fn)`. Never `==`/`hash`, which two behaviourally distinct
        callables may declare of themselves; the memo entry holds `fn`, and through it the owner
        and the function, so no id in a key is ever recycled onto a later object.

        Ceiling: any other re-accessed callable — a builtin method, a method-wrapper, a
        `partialmethod` access (a partial with a copied `__self__`) — mints a node per access.
        Losing CSE for those is never a wrong answer; merging two distinct calls is, and every
        widening past Python's own definition of sameness admitted a neighbour.

        Ceiling: under a concurrent build the assignment of ordinals to colliding callables
        follows thread interleaving.
        """
        with self._lock:
            if name is not None:
                self._fn_taken.add(name)
                return name
            # not isinstance: `__class__` is the callable's to claim
            if type(fn) is types.MethodType:
                key: object = (id(fn.__self__), id(fn.__func__))
            else:
                key = id(fn)
            held = self._fn_names.get(key)
            if held is not None:
                return held[1]
            derived = str(getattr(fn, "__name__", "lambda"))
            unique, ordinal = derived, 0
            while unique in self._fn_taken:
                ordinal += 1
                unique = f"{derived}#{ordinal}"
            self._fn_taken.add(unique)
            self._fn_names[key] = (fn, unique)  # holds `fn` (and so its owner): no id is recycled
            return unique

    def _step_reducer(self) -> None:
        if self._reducer is not None:
            self._reducer.step(self._store)

    def _wrap(self, node_id: int, inputs: Sequence[Array] = ()) -> Array:
        """Build the frontend proxy for a recorded node, carrying the M48 §2.3e merge.

        This is the propagation CHOKEPOINT: every `Array` in the repo is constructed here, so gak
        functions and module verbs inherit context-handle and variation-label propagation by
        construction. Handles on ONE ancestry chain merge to the most-derived one; divergent
        handles are an error AT THIS OP naming both, which is §2.3e's early detection.
        """
        array = self._array_cls(self, node_id)
        if inputs:
            labels: frozenset[str] | None = None
            for item in inputs:
                carried = labels_carried(item)
                if carried:
                    labels = carried if labels is None else labels | carried
            attach(array, unify_contexts(*(handle_of(item) for item in inputs)), labels)
        return array

    # ---- introspection ---------------------------------------------------------
    @property
    def backend(self) -> Backend:
        """The backend this session records on."""
        return self._backend

    # read-only views of `systematics`, under the names the frozen suites read
    @property
    def _points(self) -> Mapping[str, Point]:
        return self.systematics.points

    @property
    def _points_by_point(self) -> Mapping[Point, str]:
        return self.systematics.point_owners

    def node_count(self) -> int:
        """Return the number of interned nodes."""
        return self._store.node_count()

    def to_dot(self) -> str:
        """Return the graph in Graphviz dot format."""
        return self._store.to_dot()

    def serialized_ir(self, *outputs: Array, optimize: bool = True) -> bytes:
        """Return the recorded analysis as a canonical, versioned durable IR (plan M8).

        The IR is byte-identical.

        Pass the result ``Array``(s) of the analysis as ``outputs``: the artifact carries EXACTLY
        that output set — outputs are a property of THIS request, not session state (M22). With
        ``optimize=True`` (the default) the graph is reduced by the M4 optimizer (DCE + CSE +
        equality-saturation stage fusion), so the bytes carry the **optimized interned graph** —
        the same content-addressed artifact an executor runs — and are byte-identical to a
        fresh-session serialization of the same outputs, whatever else this Session recorded.
        ``optimize=False`` writes the whole record arena with its record ids, flagging only
        ``outputs``, so its bytes change with every node the Session holds.
        This is the "compile" step for a ``graphed.core.DurablePlan`` deployment:
        record an analysis once, serialize it once, then re-target it at many datasets with
        ``DurablePlan.with_partitions`` / ``for_datasets``.
        """
        self._mine(outputs)
        if optimize and not outputs:
            raise ValueError("serialized_ir(optimize=True) needs at least one output Array")
        return self._compile_view([arr.node_id for arr in outputs], optimize=optimize).ir

    def _compile_view(
        self,
        ids: Sequence[int],
        *,
        optimize: bool,
        maximal_fusion: bool = False,
        cone: bool = False,
    ) -> _CompileView:
        """Compile ``ids`` (the 1:1 cone with ``cone``) beside the tables the artifact reads."""
        with self._lock:
            landings: Sequence[tuple[int, int | None] | None]
            if cone:
                graph = self._store.cone(outputs=list(ids))
                ir, landings = bytes(graph.serialize()), graph.node_map()
            elif not optimize:
                ir = bytes(self._store.serialize(outputs=list(ids)))
                # opt_level=0 is 1:1 (M6): the serialized arena keeps the record ids it was built
                # with
                landings = [(nid, None) for nid in range(self._store.node_count())]
            else:
                # incremental session (M10): finish from the maintained canonical view — one
                # linear pass over the concise form instead of a whole-history optimization.
                reduced = (
                    self._reducer.finalize(
                        self._store, maximal_fusion=maximal_fusion, outputs=list(ids)
                    )[0]
                    if self._reducer is not None
                    else self._store.reduce(maximal_fusion=maximal_fusion, outputs=list(ids))[0]
                )
                ir, landings = bytes(reduced.serialize()), reduced.node_map()
            return _CompileView(
                ir=ir,
                landings=landings,
                source_names=tuple(self._source_names[nid] for nid in self._sources),
                provenance=dict(self._provenance),
                registered_labels=self.systematics.registered_labels(),
                shift_after_weight=self.systematics.shift_after_weight,
            )

    def _nodes(self) -> list[dict[str, Any]]:
        """Return the store's node list (a copy)."""
        return self._store.nodes()

    def _external_view(self) -> tuple[list[dict[str, Any]], dict[int, tuple[Any, list[int]]]]:
        """Return the store's node list and the External evaluators by node id, taken together."""
        with self._lock:
            return self._store.nodes(), dict(self._externals)

    def form(self, array: Array) -> Form:
        """Return the form of ``array``."""
        self._mine([array])
        with self._lock:
            return self._forms[array.node_id]

    def provenance(self, array: Array) -> Provenance:
        """Return the user source location that recorded ``array``."""
        self._mine([array])
        with self._lock:
            return self._provenance[array.node_id]

    def source_ids(self) -> list[int]:
        """Return the node ids of all source nodes (used by projection)."""
        with self._lock:
            return list(self._sources)

    def sources(self) -> dict[int, object]:
        """Return the source nodes' concrete data objects, keyed by node id.

        A public, read-only view — host-reader integrations inspect this instead of reaching into
        session internals.
        """
        with self._lock:
            return dict(self._sources)

    def declare_service(self, spec: ServiceSpec) -> None:
        """Declare a service nodes may name by ``params["service"]``.

        Declaring a name again must repeat the same spec.
        """
        with self._lock:
            held = self._services.setdefault(spec.name, spec)
        if held != spec:
            raise ValueError(f"service {spec.name!r} is already declared as {held}, not {spec}")

    def services(self) -> dict[str, ServiceSpec]:
        """Return the declared services by name (a copy)."""
        with self._lock:
            return dict(self._services)

    def service_for(self, name: str) -> ServiceSpec:
        """Return the declared spec called ``name``.

        An undeclared name is refused naming the declared ones.
        """
        with self._lock:
            spec = self._services.get(name)
            declared = ", ".join(sorted(self._services)) or "none"
        if spec is None:
            raise ValueError(
                f"service {name!r} is not declared on this Session (declared: {declared})"
            )
        return spec

    def reduction_state(self) -> dict[str, int] | None:
        """Incremental-reduction introspection (M10).

        How many nodes the maintained reduced view has consumed (`watermark`), cumulative reducer
        work (`total_work` — equals `watermark`, the incrementality witness), and the canonical
        size. ``None`` for a non-incremental session.
        """
        if self._reducer is None:
            return None
        return {
            "watermark": self._reducer.watermark(),
            "total_work": self._reducer.total_work(),
            "canonical_count": self._reducer.canonical_count(),
        }

    def source_name(self, node_id: int) -> str:
        """Return the name of source node ``node_id``."""
        with self._lock:
            return self._source_names[node_id]

    def form_of(self, node_id: int) -> Form:
        """Return the form of node ``node_id``."""
        with self._lock:
            return self._forms[node_id]

    def recorded_op(
        self, node_id: int
    ) -> tuple[str, Mapping[str, ParamValue], tuple[int, ...]] | None:
        """Return ``(op, params, input ids)`` node ``node_id`` was recorded with, else ``None``.

        ``None`` for a source or an External.
        """
        with self._lock:
            recorded = self._ops.get(node_id)
        return None if recorded is None else (recorded[0], recorded[1], tuple(recorded[2]))

    def reachable_sources(self, node_id: int) -> frozenset[int]:
        """Return the source node ids reachable from node ``node_id``.

        Memoized per node: the IR is append-only and hash-consed, so a node's set never changes,
        and a shared (deep) prefix is walked once however many nodes read it.
        """
        with self._lock:
            return self._reachable_sources(node_id)

    def _reachable_sources(self, node_id: int) -> frozenset[int]:
        cache = self._source_ids_cache
        hit = cache.get(node_id)
        if hit is not None:
            return hit
        # iterative post-order (a deep chain blows the recursion limit): a node is appended only
        # after every input it reaches, so the forward pass reads each input's set resolved
        order: list[int] = []
        seen: set[int] = set()
        stack: list[tuple[int, bool]] = [(node_id, False)]
        while stack:
            current, done = stack.pop()
            if done:
                order.append(current)
                continue
            if current in seen:
                continue
            seen.add(current)
            stack.append((current, True))
            for input_id in self._inputs_of(current):
                if input_id not in cache and input_id not in seen:
                    stack.append((input_id, False))
        for current in order:
            if current in self._sources:
                cache[current] = frozenset((current,))
            else:
                reached: frozenset[int] = frozenset()
                for input_id in self._inputs_of(current):
                    reached |= cache[input_id]
                cache[current] = reached
        return cache[node_id]

    def _inputs_of(self, node_id: int) -> tuple[int, ...]:
        if node_id in self._sources:
            return ()
        if node_id in self._externals:
            return tuple(self._externals[node_id][1])
        return tuple(self._ops[node_id][2])

    def sourcemap(self) -> dict[int, dict[str, object]]:
        """Per-node user-source provenance (file / line / function / sub-expression text).

        The M6 sourcemap. M9's ``inspect`` renders it so every preserved node maps back to the
        analysis line that created it, without executing anything.
        """
        with self._lock:
            provenance = dict(self._provenance)
        return {
            nid: {
                "filename": p.filename,
                "lineno": p.lineno,
                "function": p.function,
                "source": p.source,
            }
            for nid, p in provenance.items()
        }

    def source_value(self, node_id: int) -> object:
        """Return the concrete data for a source node (resolving a lazy loader).

        Used by debug execution.
        """
        with self._lock:
            value = self._sources[node_id]
        return value() if callable(value) else value

    # ---- builders --------------------------------------------------------------
    def source(
        self,
        name: str,
        *,
        form: Form,
        data: object,
        identity: object = None,
        **params: ParamValue,
    ) -> Array:
        # a name is a slot of one form bound to one data object; the store interns (name, params),
        # so a name reused for other data would silently keep its first binding
        """Record a source called ``name`` of ``form``, bound to ``data``.

        A name already bound here is reused when ``data`` is the held object, or when both
        bindings have an identity and they compare equal; any other data is refused. The
        identity is ``identity``, else ``data.source_identity()`` where ``data`` defines it.
        """
        described = getattr(form, "describe", None)
        key = described() if callable(described) else form
        if identity is None:
            own = getattr(data, "source_identity", None)
            identity = own() if callable(own) else None
        prov = capture()
        fix = "pass name= to name this one, or call graphed.new_session() for a fresh Session"
        with self._lock:
            held = self._slots.setdefault(name, (key, data, identity))
            if held[0] != key:
                raise ValueError(
                    f"source {name!r} is already a slot of another form in this Session; {fix}"
                )
            if held[1] is not data:
                # `==` on the identities only: the data's own `==` may be elementwise (ndarrays)
                if identity is None or held[2] is None or not bool(held[2] == identity):
                    raise ValueError(
                        f"source {name!r} is already bound to other data in this Session; {fix}"
                    )
                data = held[1]
            node_id = self._store.add_source(name, dict(params))
            self._forms.setdefault(node_id, form)
            self._sources.setdefault(node_id, data)
            self._source_names.setdefault(node_id, name)
            self._provenance.setdefault(node_id, prov)
            self._step_reducer()
        return self._wrap(node_id)

    def record_op(
        self,
        op: str,
        inputs: Sequence[Array],
        params: Mapping[str, ParamValue] | None = None,
        *,
        reduction: bool = False,
    ) -> Array:
        """Record ``op`` over ``inputs`` and return its array."""
        params_d: dict[str, ParamValue] = dict(params or {})
        prov = capture()
        self._mine(inputs, (op, prov))
        ids = [a.node_id for a in inputs]
        form = self._located_form(op, prov, ids, params_d)
        with self._lock:
            if reduction:
                node_id = self._store.add_reduction(op, ids, params_d)
            else:
                node_id = self._store.add_op(op, ids, params_d)
            self._enter(node_id, form, prov, op=(op, params_d, ids))
        return self._wrap(node_id, inputs)

    def _forms_of(self, ids: Sequence[int]) -> list[Form]:
        with self._lock:
            return [self._forms[nid] for nid in ids]

    def _located_form(
        self, op: str, prov: Provenance, ids: Sequence[int], params: Mapping[str, object]
    ) -> Form:
        """Return the backend's ``op_form``; its type or shape error becomes a user-located one."""
        try:
            return self._backend.op_form(op, self._forms_of(ids), params)
        except GraphedTypeError:
            raise
        except Exception as exc:
            raise GraphedTypeError(op, prov, str(exc)) from exc

    def _enter(
        self,
        node_id: int,
        form: Form,
        prov: Provenance,
        *,
        op: tuple[str, dict[str, ParamValue], list[int]] | None = None,
    ) -> None:
        # the caller holds `_lock` from the store add through here
        self._forms.setdefault(node_id, form)
        if op is not None:
            self._ops.setdefault(node_id, op)
        self._provenance.setdefault(node_id, prov)
        self._step_reducer()

    def record_exchange(self, input_array: Array, params: Mapping[str, ParamValue]) -> Array:
        """Record an ``Exchange`` boundary (M39): a pure data-movement repartition of one input.

        Its ``params`` (the scheme map — ``scheme``/``key``/``parts``/``target_bytes``) are its
        structural identity; its output form is the input form (identity on the payload, §3.3a).
        """
        params_d: dict[str, ParamValue] = dict(params)
        prov = capture()
        self._mine([input_array], ("exchange", prov))
        ids = [input_array.node_id]
        form = self._backend.op_form("exchange", self._forms_of(ids), params_d)
        with self._lock:
            node_id = self._store.add_exchange(ids, params_d)
            self._enter(node_id, form, prov, op=("exchange", params_d, ids))
        return self._wrap(node_id, [input_array])

    def record_join(self, left: Array, right: Array, params: Mapping[str, ParamValue]) -> Array:
        """Record a ``Join`` boundary (M40): a two-input co-partitioned relational combine.

        Its ``params`` (``on``/``how``) + its ordered inputs ``[left, right]`` are its structural
        identity; its output form is the backend's ``op_form("join", …)`` (flat record-merge;
        §3.3).
        """
        params_d: dict[str, ParamValue] = dict(params)
        prov = capture()
        self._mine([left, right], ("join", prov))
        ids = [left.node_id, right.node_id]
        form = self._located_form("join", prov, ids, params_d)
        with self._lock:
            node_id = self._store.add_join(ids, params_d)
            self._enter(node_id, form, prov, op=("join", params_d, ids))
        return self._wrap(node_id, [left, right])

    def record_external(
        self,
        op: str,
        fn: Callable[..., object],
        inputs: Sequence[Array],
        params: Mapping[str, ParamValue] | None = None,
        *,
        descriptor: graphed.core.PayloadDescriptor | None = None,
        form: Form | None = None,
        output_type: object = None,
        form_params: Mapping[str, object] | None = None,
    ) -> Array:
        """Record an External node.

        By default the BACKEND supplies the payload descriptor and output form (the M3
        correctionlib/ONNX family); a package recording its OWN External family (M23 — e.g.
        histogram fills) passes ``descriptor=`` and ``form=`` explicitly and the backend is not
        consulted at all, so backends stay free of domain content (§A.4).

        ``output_type=`` declares the per-element type of the External's value (m71). The
        backend's optional ``canonical_output_type`` reduces it to one string, stored as the
        ``output_type`` param (so it is node identity) and honoured by the backend's ``op_form``;
        it is exclusive with ``form=``. The frontend never interprets it.

        ``form_params=`` reach ``op_form`` only, over the stored params: a default that is a
        function of the descriptor (a preserve plugin's ``output_dtype``) types the form without
        changing the node's identity or bytes.

        A declaration is checked, never cast: with a backend ``check_output_type``, the evaluator
        of a node declaring ``output_type`` or a ``form_params`` ``output_dtype`` (also beside
        ``form=``) becomes a :class:`CheckedExternal`, and a value of another type raises
        :class:`~graphed.OutputTypeError` at this call's line.
        """
        params_d: dict[str, ParamValue] = dict(params or {})
        prov = capture()
        self._mine(inputs, (op, prov))
        if (descriptor is None) != (form is None):
            raise GraphedTypeError(op, prov, "descriptor= and form= must be given together")
        # user params may carry any key, so op_form sees `output_type` only as declared here
        form_view = {k: v for k, v in params_d.items() if k != "output_type"}
        declared_type = None
        if output_type is not None:
            declared_type = self._canonical_output_type(op, prov, output_type, params_d, form)
            params_d["output_type"] = form_view["output_type"] = declared_type
        if descriptor is None:
            descriptor = self._backend.external_payload(op, params_d)
            if descriptor is None:
                raise GraphedTypeError(
                    op, prov, "backend returned no payload descriptor for external op"
                )
        ids = [a.node_id for a in inputs]
        if form is None:
            form = self._located_form(op, prov, ids, {**form_view, **(form_params or {})})
        # the declaration op_form honoured (or, beside form=, a leaf dtype) is checked on every
        # value
        check = getattr(self._backend, "check_output_type", None)
        declared = self._declaration(op, prov, declared_type, form_params)
        with self._lock:
            node_id = self._store.add_external(descriptor, ids, params_d)
            if declared is not None and check is not None:
                node = f"{op} node {node_id} ({params_d.get('fn', descriptor.kind)!r})"
                fn = CheckedExternal(fn, declared[0], declared[1], check, op, prov, node)
            prior = self._externals.get(node_id)
            if prior is not None and isinstance(prior[0], CheckedExternal) != isinstance(
                fn, CheckedExternal
            ):
                # an 'output_type' param and output_type= store the same params, so one node
                # would drop the check
                raise GraphedTypeError(
                    op,
                    prov,
                    "this External is already recorded with its type declared on one call only",
                )
            self._externals.setdefault(node_id, (fn, ids))
            self._enter(node_id, form, prov)
        return self._wrap(node_id, inputs)

    def _canonical_output_type(
        self,
        op: str,
        prov: Provenance,
        spec: object,
        params: Mapping[str, ParamValue],
        form: Form | None,
    ) -> str:
        """Return ``output_type=`` as the backend's canonical string, refusing a conflict."""
        if "output_type" in params:
            raise GraphedTypeError(op, prov, "output_type= collides with an 'output_type' param")
        if form is not None:
            raise GraphedTypeError(op, prov, "output_type= and form= are exclusive")
        canonical = getattr(self._backend, "canonical_output_type", None)
        try:
            if canonical is None:
                raise TypeError(f"{type(self._backend).__name__} cannot declare output_type")
            return str(canonical(spec))
        except Exception as exc:  # an unrepresentable declaration -> user-located error
            raise GraphedTypeError(op, prov, str(exc)) from exc

    def _declaration(
        self,
        op: str,
        prov: Provenance,
        declared_type: str | None,
        form_params: Mapping[str, object] | None,
    ) -> tuple[str, str] | None:
        """Return the ``(key, type)`` a :class:`CheckedExternal` checks, else ``None``.

        The declared ``output_type``, else a ``form_params`` leaf ``output_dtype``; none without
        the backend's ``check_output_type``.
        """
        if getattr(self._backend, "check_output_type", None) is None:
            return None
        if declared_type is not None:
            return ("output_type", declared_type)
        canonical = getattr(self._backend, "canonical_output_type", None)
        dtype = (form_params or {}).get("output_dtype")
        if canonical is None or dtype is None:
            return None
        try:
            return ("output_dtype", str(canonical(dtype)))
        except Exception as exc:  # an unrepresentable leaf dtype -> user-located error
            raise GraphedTypeError(op, prov, str(exc)) from exc

    # ---- generic graph walk (shared by materialize + projection) ----------------
    def walk(
        self,
        array: Array,
        *,
        source: Callable[[int], object],
        op: Callable[[int, str, list[object], Mapping[str, ParamValue]], object],
        external: Callable[[int, Callable[..., object], list[object]], object],
        memo: dict[int, object] | None = None,
    ) -> object:
        """Evaluate the graph from ``array`` with caller-supplied handlers for sources, ops.

        And externals. `materialize` evaluates real data; projection evaluates reporting tracers.
        Walks of several outputs that share one ``memo`` evaluate each shared node once.

        `materialize` reaches the m60 own-session guard through this one call.
        """
        self._mine([array])
        cache: dict[int, object] = {} if memo is None else memo
        inputs_of = self._inputs_of

        # explicit-stack post-order: a deeply chained recorded graph (a long selection/systematics
        # chain) would blow Python's recursion limit if this evaluated recursively. Inputs are
        # pushed reversed so the first input resolves first -- left-to-right compute order,
        # matching a depth-first recursion; the cache deduplicates shared sub-DAGs and survives
        # re-visits.
        root = array.node_id
        stack = [root]
        while stack:
            node_id = stack[-1]
            if node_id in cache:
                stack.pop()
                continue
            with self._lock:
                pending = [i for i in inputs_of(node_id) if i not in cache]
                is_source = node_id in self._sources
                external_entry = self._externals.get(node_id)
                op_entry = self._ops.get(node_id)
            if pending:
                stack.extend(reversed(pending))
                continue
            stack.pop()
            # the callbacks run unlocked: they evaluate, and may record into this Session
            if is_source:
                cache[node_id] = source(node_id)
            elif external_entry is not None:
                fn, ids = external_entry
                cache[node_id] = external(node_id, fn, [cache[i] for i in ids])
            else:
                op_name, params, ids = op_entry  # type: ignore[misc]  # neither source nor External
                cache[node_id] = op(node_id, op_name, [cache[i] for i in ids], params)
        return cache[root]

    # ---- evaluation (reference, node-by-node; the real executor is M7) ----------
    def materialize(self, array: Array, *, memo: dict[int, object] | None = None) -> object:
        """``array``'s value; walks sharing ``memo`` evaluate each shared node once."""
        return self.walk(
            array,
            source=self.source_value,
            op=lambda _nid, name, ins, params: self._backend.eval_stage(name, ins, params),
            external=lambda _nid, fn, ins: fn(*ins),
            memo=memo,
        )


#: the open ``with session:`` scopes of this context, innermost last; a tuple per context, so
#: two threads entering one Session never share a push/pop
_scopes: contextvars.ContextVar[tuple[Session, ...]] = contextvars.ContextVar(
    "graphed_session_scopes", default=()
)


@dataclass
class _Defaults:
    """The process-wide default Session and the backend factory of future defaults."""

    session: Session | None = None
    factory: Callable[[], Backend] | None = None
    # guards the two writes; never held across `Session(...)` or a factory call, which may read
    # back
    lock: threading.Lock = field(default_factory=threading.Lock)


# one object and not context variables: every thread and every context sees the default,
# including those started before it existed
_defaults = _Defaults()


def _factory_backend() -> Backend | None:
    factory = _defaults.factory
    return None if factory is None else factory()


def current_session(backend: Backend | None = None) -> Session:
    """Return the Session array-less reads record into: the innermost open scope, else the default.

    With no default yet, one is created on ``backend``, else on the factory set with
    :func:`set_default_backend`, else on ``graphed.awkward.AwkwardBackend()``. ``backend`` is
    used only then: the Session returned may record on another backend.
    """
    stack = _scopes.get()
    if stack:
        return stack[-1]
    held = _defaults.session
    if held is not None:
        return held
    made = backend if backend is not None else _factory_backend()
    if made is None:
        try:
            from .awkward import AwkwardBackend
        except ImportError as exc:
            raise ImportError(
                "graphed.current_session(): no backend for a new default Session; install "
                "graphed[awkward], or configure one with graphed.set_default_backend(...)"
            ) from exc
        made = AwkwardBackend()
    session = Session(made)
    with _defaults.lock:
        if _defaults.session is None:
            _defaults.session = session
        return _defaults.session


def new_session(backend: Backend | None = None) -> None:
    """Replace the default Session with a new one on ``backend``.

    With no ``backend``, on the factory set with :func:`set_default_backend`; with neither, the
    default is cleared and the next array-less read creates one on its own backend. An open
    ``with`` scope still wins until it exits.
    """
    made = backend if backend is not None else _factory_backend()
    set_session(None if made is None else Session(made))


def set_session(default: Session | None, /) -> Session | None:
    """Install ``default`` as the default Session (``None`` clears it); return the previous one.

    The previous one is the restore token: ``prev = set_session(s); ...; set_session(prev)``.
    """
    if default is not None and not isinstance(default, Session):
        raise TypeError(
            f"set_session takes a graphed.Session or None, not {type(default).__name__} "
            "(an array's Session is array.session)"
        )
    with _defaults.lock:
        previous, _defaults.session = _defaults.session, default
    return previous


def set_default_backend(factory: Callable[[], Backend] | None, /) -> Callable[[], Backend] | None:
    """Set the backend factory of future defaults no creator names; return the previous factory.

    Process-wide. It applies to a bare :func:`current_session` and to :func:`new_session` without
    a backend, never to a reader with its own backend, and not to a default that already exists:
    call :func:`new_session` to apply it now. ``None`` clears it.
    """
    with _defaults.lock:
        previous, _defaults.factory = _defaults.factory, factory
    return previous


R = TypeVar("R")


def reader(kind: type[Backend] | None) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Resolve an array-less reader's ``session=``: the argument, else :func:`current_session`.

    A default made here records on ``kind()``, and a Session whose backend is not a ``kind`` is
    refused at the read (``None``: any). The legacy leading positional Session still works, with a
    ``DeprecationWarning``.

    A backend package decorates each of its source readers with it, as ``@reader(MyBackend)``
    over a function taking ``session: Session | None = None``; the function always receives a
    ``Session``.
    """

    def wrap(fn: Callable[..., R]) -> Callable[..., R]:
        name = fn.__name__

        @functools.wraps(fn)
        def read(*args: Any, **kwargs: Any) -> R:
            if args and isinstance(args[0], Session):
                if "session" in kwargs:
                    raise TypeError(
                        f"{name}: the Session was passed both positionally and as session="
                    )
                warnings.warn(
                    f"{name}(session, ...) is deprecated; pass session= as a keyword",
                    DeprecationWarning,
                    stacklevel=2,
                )
                kwargs["session"], args = args[0], args[1:]
            session = kwargs.get("session")
            if session is None:
                session = current_session(None if kind is None else kind())
            elif not isinstance(session, Session):
                raise TypeError(
                    f"{name}: session= must be a graphed.Session, not {type(session).__name__}; "
                    "'session' is reserved"
                )
            if kind is not None and not isinstance(session.backend, kind):
                wanted = kind.__name__
                raise TypeError(
                    f"{name}: the Session records on {type(session.backend).__name__}, and this "
                    f"reader records on {wanted}; pass session=, call "
                    f"graphed.new_session({wanted}()), or read inside "
                    f"`with graphed.Session({wanted}()):`"
                )
            kwargs["session"] = session
            return fn(*args, **kwargs)

        return read

    return wrap
