"""AwkwardBackend: typetracer form inference + real-array evaluation (plan M3).

`op_form` runs ops on **typetracer** arrays (metadata only — no event data is read), `eval_stage`
runs the same ops on real arrays. Both go through the single `apply` dispatch in `_ops`.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast, overload

import awkward as ak
import numpy as np
from awkward.types._awkward_datashape_parser import LarkError

from graphed import Array, Session
from graphed.aggregate import resolved_backend
from graphed.backend import PYTHON_TYPES, Form
from graphed.core import PayloadDescriptor
from graphed.session import reader
from graphed.shuffle import join_blocks

from . import join, payloads, shuffle
from ._ops import apply, carrying
from .array import AwkwardArray


@dataclass(frozen=True)
class _AsRead:
    """Pickles as its array's buffers.

    So the buffers a projected read skipped reload as the same placeholders; ``ak.Array``'s own
    pickling packs the array, which reads them and raises.
    """

    array: ak.Array

    def __reduce__(self) -> tuple[Any, ...]:
        form, length, buffers = ak.to_buffers(self.array)
        # "@"-prefixed attrs are transient, which ak.Array's own pickling drops too
        attrs = {k: v for k, v in self.array.attrs.items() if not k.startswith("@")}
        rebuild = functools.partial(ak.from_buffers, behavior=self.array.behavior, attrs=attrs)
        return rebuild, (form, length, buffers)


class _BehaviorByRef(dict[Any, Any]):
    """A backend's behavior that pickles as the import ref its backend was resolved from.

    Its lambdas do not pickle; every array derived from one carrying it keeps this same object.
    """

    def __init__(self, behavior: Mapping[Any, Any], ref: str) -> None:
        super().__init__(behavior)
        self.ref = ref

    def _keyed_as(self) -> dict[Any, Any]:
        return dict(self)

    def __reduce__(self) -> tuple[Any, ...]:
        return _behavior_from, (self.ref,)


def _behavior_from(ref: str) -> Any:
    return resolved_backend(ref)._behavior


class _BehaviorByValue(dict[Any, Any]):
    """A backend's behavior that pickles as the cloudpickle blob it was loaded from.

    A process loads one object per blob, so every array a plan's tasks return carries one that
    adopts back to it, as with :class:`_BehaviorByRef`.
    """

    def __init__(self, behavior: Mapping[Any, Any], blob: bytes) -> None:
        super().__init__(behavior)
        self.blob = blob

    def _keyed_as(self) -> dict[Any, Any]:
        return dict(self)

    def __reduce__(self) -> tuple[Any, ...]:
        return _behavior_by_value, (self.blob,)


@functools.cache
def _behavior_by_value(blob: bytes) -> _BehaviorByValue:
    import cloudpickle  # noqa: PLC0415  (only a behavior shipped by value needs it)

    return _BehaviorByValue(cloudpickle.loads(blob), blob)


#: each behavior this process shipped, beside the snapshot of it that was pickled
_SHIPPED: list[tuple[dict[Any, Any], _BehaviorByValue]] = []


def shipped_behavior(behavior: Mapping[Any, Any]) -> _BehaviorByValue:
    """Return ``behavior`` as a plan ships it to workers: by value, lambdas included.

    Each distinct behavior is pickled once per process, however many plans ship it.
    """
    for seen, shipped in _SHIPPED:
        if seen == behavior:
            return shipped
    import cloudpickle  # noqa: PLC0415  (only a behavior shipped by value needs it)

    snapshot = dict(behavior)
    try:
        blob = cloudpickle.dumps(snapshot)
    except Exception as err:
        raise TypeError(
            f"the session backend's behavior does not pickle to workers ({err}); pass the plan "
            "an importable backend factory, backend='module:factory'"
        ) from err
    shipped = _behavior_by_value(blob)
    _SHIPPED.append((snapshot, shipped))
    return shipped


def _adopt(value: Any, behavior: Mapping[Any, Any] | None) -> Any:
    """Give each array in ``value`` whose behavior merely equals ``behavior`` that object.

    Awkward merges the behaviors of several arrays into a new dict, which loses its pickling.
    """
    if not behavior:
        return value
    if isinstance(value, ak.Array):
        if value.behavior is behavior or value.behavior != behavior:
            return value
        return ak.Array(value.layout, behavior=behavior, attrs=value.attrs)
    if type(value) in (list, tuple):
        return type(value)(_adopt(item, behavior) for item in value)
    if type(value) is dict:
        return {key: _adopt(item, behavior) for key, item in value.items()}
    return value


@dataclass(eq=False)
class AwkwardForm:
    """Opaque form backed by a metadata-only typetracer array (implements graphed.Form)."""

    tt: ak.Array

    def describe(self) -> str:
        # a whole-array reduction's form is a scalar typetracer, which has a dtype but no `.type`
        """Return the awkward type string of this form."""
        tt_type = getattr(self.tt, "type", None)
        return str(self.tt.dtype if tt_type is None else tt_type)

    @property
    def is_typetracer(self) -> bool:
        """Return whether this form is backed by a typetracer."""
        return ak.backend(self.tt) == "typetracer"


_BOUNDARY = frozenset(
    {"source", "external", "correction", "onnx", "map", "ak.sum", "ak.any", "ak.all", "ak.count"}
)
# ak.Array descriptors the record-time typetracer answers as the data would; it does not carry the
# source's attrs/behavior/named axes, nor a known length, so those are refused
_INTROSPECT_EAGER = frozenset({"fields", "type", "typestr", "ndim", "is_tuple", "positional_axis"})


_EXTERNAL = frozenset({"map", "correction", "onnx", "external"})


@functools.cache  # a declared type is parsed once, not on every checked call
def _type(canonical: str) -> ak.types.Type:
    """Return the awkward type of a canonical ``output_type``.

    A primitive the datashape grammar lacks (``float16`` on awkward 2.14) but awkward's dtype table
    knows is built directly.
    """
    try:
        return ak.types.from_datashape(canonical, highlevel=False)
    except LarkError:
        ak.types.numpytype.primitive_to_dtype(canonical)  # TypeError unless a known primitive
        return ak.types.NumpyType(canonical)


def canonical_output_type(spec: object) -> str:
    """Reduce an ``output_type`` spelling to awkward's type string, a fixpoint.

    The spelling is a datashape string, awkward Type/Form/ArrayType, numpy dtype-like, or Python
    type; refuse what cannot be built.
    """
    return _canonical_string(spec) if isinstance(spec, str) else _canonical_output_type(spec)


def _canonical_output_type(spec: object) -> str:
    try:
        s = PYTHON_TYPES[spec] if isinstance(spec, type) and spec in PYTHON_TYPES else spec
        if isinstance(s, ak.forms.Form):
            s = s.type
        if isinstance(s, (ak.types.ArrayType, ak.types.ScalarType)):
            s = s.content
        canonical = str(s) if isinstance(s, ak.types.Type) else None
        if isinstance(s, str):
            with contextlib.suppress(LarkError):  # datashape first: `"byte"` is awkward's byte
                canonical = str(ak.types.from_datashape(s, highlevel=False))
        if canonical is None:
            # length one: a zero-length `U` array raises on the awkward 2.6 floor
            canonical = str(
                ak.from_numpy(np.zeros(1, dtype=np.dtype(cast("Any", s)))).type.content
            )
        rebuilt = _type(canonical)
        # the installed grammar misreads it (2.14: `float16[parameters=`)
        if str(rebuilt) != canonical:
            raise TypeError(f"rebuilds as {str(rebuilt)!r}")
        ak.forms.from_type(rebuilt).length_one_array(highlevel=False)
    except (LarkError, TypeError, ValueError) as exc:
        raise TypeError(
            f"output_type {spec!r} is neither a type string the installed awkward can parse and "
            "build nor a numpy dtype it represents"
        ) from exc
    return canonical


# a string spelling (``"float64"`` on every apply_correction) is parsed once, not per record
_canonical_string = functools.cache(_canonical_output_type)


def astype_form(form: AwkwardForm, dtype: object) -> AwkwardForm:
    """Return ``form`` with its leaves cast to the primitive ``dtype``.

    ``dtype`` is any `canonical_output_type` spelling of one; a scalar stays a scalar.
    """
    t = _type(canonical_output_type(dtype))
    if not isinstance(t, ak.types.NumpyType) or t.parameters:
        raise TypeError(f"output_dtype {dtype!r} is not a primitive dtype")
    if ak.typetracer.is_unknown_scalar(form.tt):
        return AwkwardForm(ak.typetracer.create_unknown_scalar(np.dtype(t.primitive)))
    return AwkwardForm(ak.values_astype(form.tt, t.primitive))


def broadcast_form(forms: Sequence[AwkwardForm]) -> AwkwardForm:
    """Return the form ``ak.broadcast_arrays`` gives the array ``forms``.

    Scalars broadcast away; the first form when none is an array.
    """
    arrays = [f for f in forms if isinstance(f.tt, ak.Array)]
    if len(arrays) < 2:
        return (arrays or list(forms))[0]
    return AwkwardForm(ak.broadcast_arrays(*(f.tt for f in arrays))[0])


def declared_form(forms: Sequence[AwkwardForm], canonical: str) -> AwkwardForm:
    """Return the form of an External declared ``canonical``.

    That element type over an array input's length, or a scalar (a record one included) when no
    input is an array.
    """
    t = _type(canonical)
    if not any(isinstance(f.tt, ak.Array) for f in forms):
        if isinstance(t, ak.types.NumpyType) and not t.parameters:
            return AwkwardForm(ak.typetracer.create_unknown_scalar(np.dtype(t.primitive)))
        raise TypeError(f"output_type {canonical!r} over a scalar input must be a primitive dtype")
    layout = ak.forms.from_type(t).length_one_array(highlevel=False)
    return AwkwardForm(ak.Array(layout.to_typetracer(forget_length=True)))


def _fits(value: ak.types.Type, declared: ak.types.Type) -> bool:
    """``value`` is ``declared`` wherever it holds data.

    ``unknown``, the type of a buffer holding no values (an all-empty list), fits any declared
    type.
    """
    if isinstance(value, ak.types.UnknownType):
        return True
    if type(value) is not type(declared) or value.parameters != declared.parameters:
        return False
    other: Any = declared
    if isinstance(value, (ak.types.RecordType, ak.types.UnionType)):
        fields = value.fields if isinstance(value, ak.types.RecordType) else None
        return (
            fields == (other.fields if fields is not None else None)
            and len(value.contents) == len(other.contents)
            and all(map(_fits, value.contents, other.contents))
        )
    if isinstance(value, ak.types.RegularType) and value.size != other.size:
        return False
    if isinstance(value, (ak.types.ListType, ak.types.RegularType, ak.types.OptionType)):
        return _fits(value.content, other.content)
    return str(value) == str(declared)


def _leaves(t: ak.types.Type) -> list[str]:
    """Return the primitive of every leaf of ``t``.

    An ``unknown`` leaf holds no values and has none.
    """
    if isinstance(t, ak.types.NumpyType):
        return [t.primitive]
    if isinstance(t, (ak.types.RecordType, ak.types.UnionType)):
        return [p for c in t.contents for p in _leaves(c)]
    if isinstance(t, (ak.types.ListType, ak.types.RegularType, ak.types.OptionType)):
        return _leaves(t.content)
    return []


def _is_array(x: object) -> bool:
    """Array-ness at run time: a 0-d ndarray and an `ak.Record` are scalars."""
    # never np.ndim: it converts the value, and a ragged list raises
    return isinstance(x, (ak.Array, list, tuple)) or getattr(x, "ndim", 0) > 0


def check_output_type(
    value: object, inputs: Sequence[object], key: str, declared: str
) -> str | None:
    """``None`` when an External's ``value`` has its declared type, else the value's type string.

    ``output_type`` is the whole element type: the value's type string must equal it (option-ness,
    regular vs var, record names and parameters included), except that ``unknown`` inside an array
    fits anything. ``output_dtype`` is a static leaf dtype: every leaf must have it.
    """
    try:
        t = ak.type(value)
    except (TypeError, ValueError):  # not array-like at all
        return type(value).__name__
    content = t.content if isinstance(t, (ak.types.ArrayType, ak.types.ScalarType)) else t
    actual = str(content)
    if key == "output_dtype":
        return None if all(p == declared for p in _leaves(content)) else actual
    # as `declared_form`: an array when any input is one; value and inputs share the predicate
    scalar = not _is_array(value)
    if scalar == any(map(_is_array, inputs)):
        return f"scalar {actual}" if scalar else str(t)
    if actual == declared:
        return None
    if not scalar and "unknown" in actual and _fits(content, _type(declared)):
        return None
    return actual


class AwkwardBackend:
    #: the backend's versioned shuffle-format token (folded into the V2 task ids, §7.2)
    """The awkward backend: typetracer forms at record time, awkward arrays in eval."""

    identity = "graphed-awkward/0"

    #: M59: this backend evaluates the shared inner-axis tuple key (`subscript`)
    subscript_keys = True

    def __init__(self, behavior: Mapping[str, object] | None = None) -> None:
        # M18: a registered behavior dict (e.g. vector's) makes behavior PROPERTIES work through
        # plain attribute access — on the typetracer at record time and on real arrays in eval.
        """Use ``behavior``, when given, for behavior properties and methods."""
        self._behavior: dict[Any, Any] | None = (
            behavior
            if isinstance(behavior, _BehaviorByValue)
            else dict(behavior)
            if behavior
            else None
        )

    def array_type(self) -> type[Array]:
        """Return the awkward-idiom proxy.

        Sessions return ``AwkwardArray``, so ``ak.<fn>`` records ``gak.<fn>``.
        """
        return AwkwardArray

    def op_form(
        self, op: str, inputs: Sequence[Form], params: Mapping[str, object]
    ) -> AwkwardForm:
        # `inputs` is the protocol-wide `Sequence[Form]`, not `Sequence[AwkwardForm]`: narrowing a
        # parameter would make AwkwardBackend structurally incompatible with `graphed.Backend`
        # (contravariance), so `Session(AwkwardBackend())` would not type-check. A session only
        # ever feeds a backend the forms that same backend produced.
        """Infer the form ``op`` gives ``inputs`` from their typetracers."""
        forms = cast("Sequence[AwkwardForm]", inputs)
        if op == "exchange":
            # a pure data-movement boundary is identity on the payload form (§3.3a)
            return forms[0]
        if op == "join":
            # M40 (§3.3): flat relational record-merge form; how=left/outer ⇒ missing side
            # option-typed
            return AwkwardForm(join.join_form([f.tt for f in forms], params))
        if op in _EXTERNAL:
            # an External's value type is not derivable from its inputs: a declared `output_type`
            # is recorded as given, a plugin's static leaf `output_dtype` casts the leaves of the
            # inputs' broadcast, and anything else records the first input's form
            declared = params.get("output_type")
            if declared is not None:
                return declared_form(forms, str(declared))
            leaf = params.get("output_dtype")
            if leaf is not None:
                return astype_form(broadcast_form(forms), leaf)
            return forms[0]
        operands = [f.tt for f in forms]
        return AwkwardForm(apply(op, operands, params, behavior=self._behavior))

    def eval_stage(
        self, op: str, inputs: Sequence[object], params: Mapping[str, object]
    ) -> object:
        """Evaluate ``op`` over real awkward ``inputs``."""
        if op == "join":  # needs `self` (the shared kernel routes through JoinBackend primitives)
            return self._eval_join(inputs, params)
        return self.adopt(apply(op, inputs, params, behavior=self._behavior))

    # ---- M54: behavior methods with arguments ------------------------------------------------
    canonical_output_type = staticmethod(canonical_output_type)
    check_output_type = staticmethod(check_output_type)

    def attribute_kind(self, form: AwkwardForm, name: str) -> str:
        """Classify `arr.<name>` as a field, method, introspection or property.

        A record FIELD shadows the behavior (as `apply`'s `field` branch resolves it), a behavior
        function is a "method" the frontend hands back as a callable, ak.Array's own descriptors
        (`fields`, `ndim`, ...) are "introspection" answered by `introspect`, and anything else
        that resolves is a "property" recorded as a `field` op. An unresolved name raises
        `AttributeError` and the frontend keeps today's path.
        """
        tt = self._with_behavior(form.tt)
        if name in tt.fields:
            return "field"
        static = inspect.getattr_static(tt, name)
        # the PROPERTY side is the closed set (a data descriptor or a cached_property is read
        # like a field); anything else that is callable or a descriptor is a method, so a method
        # descriptor the stdlib adds later (partialmethod, singledispatchmethod, ...) still counts
        if hasattr(type(static), "__set__") or isinstance(static, functools.cached_property):
            # ak.Array's own descriptors describe the array, they are not per-element data
            return (
                "introspection"
                if static is inspect.getattr_static(ak.Array, name, None)
                else "property"
            )
        return "method" if callable(static) or hasattr(static, "__get__") else "property"

    def introspect(self, form: AwkwardForm, name: str) -> object:
        """Answer an `introspection` attribute from the record-time typetracer.

        Raise `AttributeError` when the typetracer cannot answer it as the data would.
        """
        if name not in _INTROSPECT_EAGER:
            hint = (
                "use gak.mask(array, condition)"
                if name == "mask"
                else "materialize the array and read it there"
            )
            raise AttributeError(f"a deferred graphed array cannot answer {name!r}; {hint}")
        return getattr(form.tt, name)

    def method_outputs(
        self, forms: Sequence[AwkwardForm], params: Mapping[str, object]
    ) -> tuple[object, ...] | None:
        """Run the call on the typetracers BEFORE anything is recorded.

        `None` for one awkward array, and for a tuple of them the same NESTING with `None` at every
        leaf (M59 — the shape `metric_table(return_combinations=True)` answers in). Anything else
        raises.
        """
        result = apply(
            "method", [self._with_behavior(f.tt) for f in forms], params, behavior=self._behavior
        )
        return self._output_shape(result, params)

    def _output_shape(
        self, result: object, params: Mapping[str, object]
    ) -> tuple[object, ...] | None:
        if isinstance(result, ak.Array):
            return None
        if isinstance(result, tuple) and result:
            return tuple(self._output_shape(item, params) for item in result)
        raise TypeError(
            f"{params['method']}() returned {type(result).__name__}, which is not an awkward "
            "array or a tuple of awkward arrays, so it cannot be recorded"
        )

    def worker_factory(self) -> Callable[[], AwkwardBackend]:
        """Return the factory a plan's workers rebuild this backend with, behavior included."""
        if not self._behavior:
            return type(self)
        return functools.partial(type(self), behavior=shipped_behavior(self._behavior))

    def resolved_from(self, ref: str) -> AwkwardBackend:
        """Return this backend as a worker resolved it from the import ``ref``.

        Its behavior, and so every partial carrying it, pickles as ``ref``.
        """
        if self._behavior and not isinstance(self._behavior, _BehaviorByRef):
            self._behavior = _BehaviorByRef(self._behavior, ref)
        return self

    def adopt(self, value: Any) -> Any:
        """Return ``value`` carrying this backend's behavior object.

        ``value`` is an array, or a list, tuple or dict of them; it carries the object wherever it
        carries an equal copy.
        """
        return _adopt(value, self._behavior)

    def capturable(self, chunk: ak.Array) -> _AsRead:
        """Return ``chunk`` in a form that pickles back to the chunk as read.

        It pickles with cloudpickle, as behaviors hold lambdas.
        """
        return _AsRead(chunk)

    def _with_behavior(self, tt: ak.Array) -> ak.Array:
        return carrying(tt, self._behavior) if self._behavior else tt

    def _eval_join(self, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        left, right = inputs[0], inputs[1]
        on = join.on_from_params(params)
        how = str(params.get("how", "inner"))
        # projection replay: structural merge, no matching/data read
        if ak.backend(left) == "typetracer":
            return join.merge_records(left, right, on=on)
        # gak.join(grouped=True): awkward-only regroup post-op
        if bool(params.get("grouped", False)):
            return join.join_grouped(left, right, on=on, how=how)
        # the shared radix-hash kernel (JoinBackend prims only)
        return join_blocks(self, left, right, on=on, how=how)

    def boundary_ops(self) -> frozenset[str]:
        """Return the op names that end a fused stage."""
        return _BOUNDARY

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        """Return ``used`` unchanged (M5)."""
        return used  # M5

    def broadcast_like(self, value: object, factor: object) -> object:
        """§6.1d's broadcast seam for this idiom: `factor` given `value`'s jagged structure.

        A rectilinear idiom needs nothing here, which is why the seam is optional and
        `graphed.broadcast_like` falls back to the identity when a backend omits it.
        """
        from .functions import broadcast_arrays  # noqa: PLC0415  (functions imports this module)

        return broadcast_arrays(value, factor)[1]  # type: ignore[arg-type]

    def external_payload(self, op: str, params: Mapping[str, object]) -> PayloadDescriptor | None:
        """Return the payload descriptor an External ``op`` carries, if any."""
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

    # --- ShuffleBackend exchange half (M39 §3.0) — thin delegates to the pure `shuffle` module ---
    def partition(
        self,
        block: Any,
        key_field: str,
        parts: int,
        *,
        salt: int = 0,
        boundaries: object = None,
    ) -> tuple[Any, ...]:
        """Route each row of ``block`` to one of ``parts`` sub-blocks by ``key_field``."""
        return shuffle.partition(block, key_field, parts, salt=salt, boundaries=boundaries)

    def concat(self, blocks: Sequence[Any]) -> Any:
        """Concatenate ``blocks`` row-wise."""
        return shuffle.concat(blocks)

    def slice_rows(self, block: Any, start: int, stop: int) -> Any:
        """Return rows ``[start, stop)`` of ``block`` at event boundaries."""
        return shuffle.slice_rows(block, start, stop)

    def estimated_bytes(self, block_or_form: object) -> int:
        """Estimate the bytes of a block or form."""
        return shuffle.estimated_bytes(block_or_form)

    def to_wire(self, block: Any) -> bytes:
        """Serialize ``block`` deterministically."""
        return shuffle.to_wire(block)

    def from_wire(self, data: bytes) -> Any:
        """Deserialize a block :meth:`to_wire` wrote."""
        return shuffle.from_wire(data)

    # ---- JoinBackend join half (M40 §3.3) — thin delegates to the pure `join` module ----
    def match_indices(
        self, build: Any, probe: Any, *, on: Sequence[str], how: str = "inner"
    ) -> tuple[Any, Any]:
        """Match ``build`` and ``probe`` rows on ``on`` as aligned index arrays."""
        return join.match_indices(build, probe, on=on, how=how)

    def take(self, block: Any, index: Any) -> Any:
        """Gather ``block``'s rows at ``index``; ``-1`` gives a missing row."""
        return join.take(block, index)

    def merge_records(self, left: Any, right: Any, *, on: Sequence[str]) -> Any:
        """Merge two aligned record blocks, keeping the ``on`` fields once."""
        return join.merge_records(left, right, on=on)


def _typetracer(array: ak.Array) -> ak.Array:
    return ak.Array(ak.Array(array).layout.to_typetracer(forget_length=True))


@overload
def from_awkward(session: Session, name: str, array: object, **descriptor: object) -> Any: ...
@overload
def from_awkward(
    name: str, array: object, *, session: Session | None = None, **descriptor: object
) -> Any: ...
@reader(AwkwardBackend)
def from_awkward(
    name: str, array: object, *, session: Session | None = None, **descriptor: object
) -> Any:
    """Create a metadata-only source from an in-memory awkward array.

    It records into ``session``, else the current Session. The form comes from the typetracer;
    the real array is retained only for evaluation.
    """
    real = array if isinstance(array, ak.Array) else ak.Array(array)
    return cast("Session", session).source(name, form=AwkwardForm(_typetracer(real)), data=real)


# from_parquet moved to io.py (M15): the multi-file, blind-partition specialization of the
# graphed.parquet base; the M3 single-file shape is a special case of it.
