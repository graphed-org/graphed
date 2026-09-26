"""AwkwardBackend: typetracer form inference + real-array evaluation (plan M3).

`op_form` runs ops on **typetracer** arrays (metadata only — no event data is read), `eval_stage`
runs the same ops on real arrays. Both go through the single `apply` dispatch in `_ops`.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import awkward as ak
import numpy as np
from awkward.types._awkward_datashape_parser import LarkError

from graphed import Session
from graphed.backend import PYTHON_TYPES, Form
from graphed.core import PayloadDescriptor

from . import join, payloads, shuffle
from ._ops import apply


@dataclass(frozen=True)
class _AsRead:
    """Pickles as its array's buffers, so the buffers a projected read skipped reload as the same
    placeholders; ``ak.Array``'s own pickling packs the array, which reads them and raises."""

    array: ak.Array

    def __reduce__(self) -> tuple[Any, ...]:
        form, length, buffers = ak.to_buffers(self.array)
        # "@"-prefixed attrs are transient, which ak.Array's own pickling drops too
        attrs = {k: v for k, v in self.array.attrs.items() if not k.startswith("@")}
        rebuild = functools.partial(ak.from_buffers, behavior=self.array.behavior, attrs=attrs)
        return rebuild, (form, length, buffers)


@dataclass(eq=False)
class AwkwardForm:
    """Opaque form backed by a metadata-only typetracer array (implements graphed.Form)."""

    tt: ak.Array

    def describe(self) -> str:
        # a whole-array reduction's form is a scalar typetracer, which has a dtype but no `.type`
        tt_type = getattr(self.tt, "type", None)
        return str(self.tt.dtype if tt_type is None else tt_type)

    @property
    def is_typetracer(self) -> bool:
        return ak.backend(self.tt) == "typetracer"


_BOUNDARY = frozenset(
    {"source", "external", "correction", "onnx", "map", "ak.sum", "ak.any", "ak.all", "ak.count"}
)
# ak.Array descriptors the record-time typetracer answers as the data would; it does not carry the
# source's attrs/behavior/named axes, nor a known length, so those are refused
_INTROSPECT_EAGER = frozenset({"fields", "type", "typestr", "ndim", "is_tuple", "positional_axis"})


_EXTERNAL = frozenset({"map", "correction", "onnx", "external"})


def _type(canonical: str) -> ak.types.Type:
    """The awkward type of a canonical ``output_type``; a primitive the datashape grammar lacks
    (``float16`` on awkward 2.14) but awkward's dtype table knows is built directly."""
    try:
        return ak.types.from_datashape(canonical, highlevel=False)
    except LarkError:
        ak.types.numpytype.primitive_to_dtype(canonical)  # TypeError unless a known primitive
        return ak.types.NumpyType(canonical)


def canonical_output_type(spec: object) -> str:
    """Reduce an ``output_type`` spelling (datashape string, awkward Type/Form/ArrayType, numpy
    dtype-like, Python type) to awkward's type string, a fixpoint; refuse what cannot be built."""
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
            canonical = str(ak.from_numpy(np.zeros(1, dtype=np.dtype(cast("Any", s)))).type.content)
        rebuilt = _type(canonical)
        if str(rebuilt) != canonical:  # the installed grammar misreads it (2.14: `float16[parameters=`)
            raise TypeError(f"rebuilds as {str(rebuilt)!r}")
        ak.forms.from_type(rebuilt).length_one_array(highlevel=False)
    except (LarkError, TypeError, ValueError) as exc:
        raise TypeError(
            f"output_type {spec!r} is neither a type string the installed awkward can parse and "
            "build nor a numpy dtype it represents"
        ) from exc
    return canonical


def astype_form(form: AwkwardForm, dtype: object) -> AwkwardForm:
    """``form`` with its leaves cast to the primitive ``dtype`` (any `canonical_output_type`
    spelling of one); a scalar stays a scalar."""
    t = _type(canonical_output_type(dtype))
    if not isinstance(t, ak.types.NumpyType) or t.parameters:
        raise TypeError(f"output_dtype {dtype!r} is not a primitive dtype")
    if ak.typetracer.is_unknown_scalar(form.tt):
        return AwkwardForm(ak.typetracer.create_unknown_scalar(np.dtype(t.primitive)))
    return AwkwardForm(ak.values_astype(form.tt, t.primitive))


def declared_form(first: AwkwardForm, canonical: str) -> AwkwardForm:
    """The form of an External declared ``canonical``: that element type over ``first``'s length."""
    t = _type(canonical)
    if ak.typetracer.is_unknown_scalar(first.tt):
        if isinstance(t, ak.types.NumpyType) and not t.parameters:
            return AwkwardForm(ak.typetracer.create_unknown_scalar(np.dtype(t.primitive)))
        raise TypeError(f"output_type {canonical!r} over a scalar input must be a primitive dtype")
    layout = ak.forms.from_type(t).length_one_array(highlevel=False)
    return AwkwardForm(ak.Array(layout.to_typetracer(forget_length=True)))


def _fits(value: ak.types.Type, declared: ak.types.Type) -> bool:
    """``value`` is ``declared`` wherever it holds data; ``unknown``, the type of a buffer holding no
    values (an all-empty list), fits any declared type."""
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
    """The primitive of every leaf of ``t``; an ``unknown`` leaf holds no values and has none."""
    if isinstance(t, ak.types.NumpyType):
        return [t.primitive]
    if isinstance(t, (ak.types.RecordType, ak.types.UnionType)):
        return [p for c in t.contents for p in _leaves(c)]
    if isinstance(t, (ak.types.ListType, ak.types.RegularType, ak.types.OptionType)):
        return _leaves(t.content)
    return []


def check_output_type(value: object, inputs: Sequence[object], key: str, declared: str) -> str | None:
    """``None`` when an External's ``value`` has its declared type, else the value's type string.

    ``output_type`` is the whole element type: the value's type string must equal it (option-ness,
    regular vs var, record names and parameters included), except that ``unknown`` inside an array
    fits anything. ``output_dtype`` is a static leaf dtype: every leaf must have it."""
    try:
        t = ak.type(value)
    except (TypeError, ValueError):  # not array-like at all
        return type(value).__name__
    content = t.content if isinstance(t, (ak.types.ArrayType, ak.types.ScalarType)) else t
    actual = str(content)
    if key == "output_dtype":
        return None if all(p == declared for p in _leaves(content)) else actual
    if actual == declared:
        return None
    if isinstance(t, ak.types.ArrayType) and "unknown" in actual and _fits(content, _type(declared)):
        return None
    return actual


class AwkwardBackend:
    #: the backend's versioned shuffle-format token (folded into the V2 task ids, §7.2)
    identity = "graphed-awkward/0"

    #: M59: this backend evaluates the shared inner-axis tuple key (`subscript`)
    subscript_keys = True

    def __init__(self, behavior: Mapping[str, object] | None = None) -> None:
        # M18: a registered behavior dict (e.g. vector's) makes behavior PROPERTIES work through
        # plain attribute access — on the typetracer at record time and on real arrays in eval.
        self._behavior = dict(behavior) if behavior else None

    def op_form(self, op: str, inputs: Sequence[Form], params: Mapping[str, object]) -> AwkwardForm:
        # `inputs` is the protocol-wide `Sequence[Form]`, not `Sequence[AwkwardForm]`: narrowing a
        # parameter would make AwkwardBackend structurally incompatible with `graphed.Backend`
        # (contravariance), so `Session(AwkwardBackend())` would not type-check. A session only ever
        # feeds a backend the forms that same backend produced.
        forms = cast("Sequence[AwkwardForm]", inputs)
        if op == "exchange":
            return forms[0]  # a pure data-movement boundary is identity on the payload form (§3.3a)
        if op == "join":
            # M40 (§3.3): flat relational record-merge form; how=left/outer ⇒ missing side option-typed
            return AwkwardForm(join.join_form([f.tt for f in forms], params))
        if op in _EXTERNAL:
            # an External's value type is not derivable from its inputs: a declared `output_type`
            # is recorded as given, a plugin's static leaf `output_dtype` casts the first input's
            # leaves, and anything else records the first input's form
            declared = params.get("output_type")
            if declared is not None:
                return declared_form(forms[0], str(declared))
            leaf = params.get("output_dtype")
            if leaf is not None:
                return astype_form(forms[0], leaf)
            return forms[0]
        operands = [f.tt for f in forms]
        return AwkwardForm(apply(op, operands, params, behavior=self._behavior))

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        if op == "join":  # needs `self` (the shared kernel routes through JoinBackend primitives)
            return self._eval_join(inputs, params)
        return apply(op, inputs, params, behavior=self._behavior)

    # ---- M54: behavior methods with arguments ------------------------------------------------
    canonical_output_type = staticmethod(canonical_output_type)
    check_output_type = staticmethod(check_output_type)

    def attribute_kind(self, form: AwkwardForm, name: str) -> str:
        """Classify `arr.<name>`: a record FIELD shadows the behavior (as `apply`'s `field` branch
        resolves it), a behavior function is a "method" the frontend hands back as a callable, ak.Array's
        own descriptors (`fields`, `ndim`, ...) are "introspection" answered by `introspect`, and
        anything else that resolves is a "property" recorded as a `field` op. An unresolved name
        raises `AttributeError` and the frontend keeps today's path."""
        tt = self._with_behavior(form.tt)
        if name in tt.fields:
            return "field"
        static = inspect.getattr_static(tt, name)
        # the PROPERTY side is the closed set (a data descriptor or a cached_property is read
        # like a field); anything else that is callable or a descriptor is a method, so a method
        # descriptor the stdlib adds later (partialmethod, singledispatchmethod, ...) still counts
        if hasattr(type(static), "__set__") or isinstance(static, functools.cached_property):
            # ak.Array's own descriptors describe the array, they are not per-element data
            return "introspection" if static is inspect.getattr_static(ak.Array, name, None) else "property"
        return "method" if callable(static) or hasattr(static, "__get__") else "property"

    def introspect(self, form: AwkwardForm, name: str) -> object:
        """Answer an `introspection` attribute from the record-time typetracer, or raise
        `AttributeError` when the typetracer cannot answer it as the data would."""
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
        """Run the call on the typetracers BEFORE anything is recorded: `None` for one awkward
        array, and for a tuple of them the same NESTING with `None` at every leaf (M59 — the shape
        `metric_table(return_combinations=True)` answers in). Anything else raises."""
        result = apply("method", [self._with_behavior(f.tt) for f in forms], params, behavior=self._behavior)
        return self._output_shape(result, params)

    def _output_shape(self, result: object, params: Mapping[str, object]) -> tuple[object, ...] | None:
        if isinstance(result, ak.Array):
            return None
        if isinstance(result, tuple) and result:
            return tuple(self._output_shape(item, params) for item in result)
        raise TypeError(
            f"{params['method']}() returned {type(result).__name__}, which is not an awkward array "
            "or a tuple of awkward arrays, so it cannot be recorded"
        )

    def capturable(self, chunk: ak.Array) -> _AsRead:
        """``chunk`` in a form that pickles (with cloudpickle, as behaviors hold lambdas) back to
        the chunk as read."""
        return _AsRead(chunk)

    def _with_behavior(self, tt: ak.Array) -> ak.Array:
        return ak.Array(tt.layout, behavior=self._behavior, attrs=tt.attrs) if self._behavior else tt

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

    def broadcast_like(self, value: object, factor: object) -> object:
        """§6.1d's broadcast seam for this idiom: `factor` given `value`'s jagged structure.

        A rectilinear idiom needs nothing here, which is why the seam is optional and
        `graphed.broadcast_like` falls back to the identity when a backend omits it."""
        from .functions import broadcast_arrays  # noqa: PLC0415  (functions imports this module)

        return broadcast_arrays(value, factor)[1]  # type: ignore[arg-type]

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
