"""graphed-numpy: a numpy backend proving the graphed backend seam (plans M2 + M5).

Operates on 1-D numpy arrays (a "bag"): elementwise arithmetic, boolean filtering, sum reduction,
and arbitrary opaque Python callables via `map`. M5 adds **record sources** (named columns) + field
access and a real **necessary-buffer (column) projection** that tracks which fields each computation
touches. `external_payload` flags wrapped opaque callables as a preservation risk (plan A.3.1).
"""

from __future__ import annotations

import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from graphed import Array, Session
from graphed_core import PayloadDescriptor

from .projection import project

_ARITH = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}


@dataclass(frozen=True)
class NumpyForm:
    """Opaque form for a numpy bag (dtype + kind), or a record source (named columns)."""

    dtype: np.dtype
    kind: str = "vector"
    fields: tuple[tuple[str, str], ...] | None = None  # (column, dtype-str) for record sources

    def describe(self) -> str:
        if self.fields is not None:
            return f"record[{','.join(f for f, _ in self.fields)}]"
        return f"{self.kind}[{self.dtype}]"


def _is_numeric(form: NumpyForm) -> bool:
    return form.fields is None and np.issubdtype(form.dtype, np.number)


class NumpyBackend:
    """A `graphed.Backend` over numpy arrays (M2) + record field access (M5)."""

    def op_form(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> NumpyForm:
        forms = [f for f in inputs if isinstance(f, NumpyForm)]
        if op in _ARITH:
            if "scalar" in params:  # array OP scalar (the M3 frontend encodes the scalar in params)
                (a,) = forms
                if not _is_numeric(a):
                    raise TypeError(f"{op} requires a numeric operand, got {a.describe()}")
                return NumpyForm(np.promote_types(a.dtype, np.asarray(params["scalar"]).dtype))
            a, b = forms
            if not (_is_numeric(a) and _is_numeric(b)):
                raise TypeError(f"{op} requires numeric operands, got {a.describe()} and {b.describe()}")
            return NumpyForm(np.promote_types(a.dtype, b.dtype))
        if op == "field":
            (rec,) = forms
            if rec.fields is None:
                raise TypeError(f"field access requires a record source, got {rec.describe()}")
            fd = dict(rec.fields)
            name = str(params["field"])
            if name not in fd:
                raise TypeError(f"record has no field {name!r}; fields are {sorted(fd)}")
            return NumpyForm(np.dtype(fd[name]))
        if op == "filter":
            data, mask = forms
            if mask.dtype != np.bool_:
                raise TypeError(f"filter mask must be boolean, got {mask.describe()}")
            return NumpyForm(data.dtype)
        if op == "sum":
            (a,) = forms
            if not _is_numeric(a):
                raise TypeError(f"sum requires a numeric array, got {a.describe()}")
            return NumpyForm(a.dtype, kind="scalar")
        if op == "map":
            return NumpyForm(np.dtype(object))  # opaque callable: result form unknown
        raise TypeError(f"unsupported op {op!r}")

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        if op in _ARITH:
            if "scalar" in params:
                s, x = params["scalar"], inputs[0]
                return _ARITH[op](s, x) if params.get("side") == "l" else _ARITH[op](x, s)
            return _ARITH[op](inputs[0], inputs[1])
        if op == "field":
            return np.asarray(inputs[0][str(params["field"])])  # type: ignore[index]
        if op == "filter":
            return np.asarray(inputs[0])[np.asarray(inputs[1])]
        if op == "sum":
            return np.sum(inputs[0])
        raise TypeError(f"unsupported op {op!r}")

    def boundary_ops(self) -> frozenset[str]:
        return frozenset({"source", "sum", "map"})

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        # Vestigial M2 per-op stub; the real projection is the module-level `project` (M5).
        return used

    def external_payload(self, op: str, params: Mapping[str, object]) -> PayloadDescriptor | None:
        if op != "map":
            return None
        fn_name = str(params.get("fn", "lambda"))
        return PayloadDescriptor(
            kind="opaque_callable",
            content_hash=f"unhashed-opaque:{fn_name}",
            framework="python",
            version=platform.python_version(),
            io_schema="opaque->opaque",
            preprocessing_ref=None,
        )


def from_array(session: Session, name: str, values: object) -> Array:
    """Create a flat source Array from a numpy array (or anything array-like)."""
    arr = np.asarray(values)
    return session.source(name, form=NumpyForm(arr.dtype), data=arr)


def from_record(session: Session, name: str, **columns: object) -> Array:
    """Create a record source (named columns) — the buffer-projection target."""
    cols = {k: np.asarray(v) for k, v in columns.items()}
    fields = tuple((k, v.dtype.str) for k, v in cols.items())
    return session.source(name, form=NumpyForm(np.dtype(object), kind="record", fields=fields), data=cols)


__all__ = ["NumpyBackend", "NumpyForm", "from_array", "from_record", "project"]
__version__ = "0.0.1"
