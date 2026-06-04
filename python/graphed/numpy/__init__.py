"""graphed-numpy: a trivial numpy backend proving the graphed backend seam (plan M2).

Operates on 1-D numpy arrays (a "bag"): elementwise arithmetic, boolean filtering, sum reduction,
and arbitrary opaque Python callables via `map`. No HEP here. `external_payload` returns a payload
descriptor for any wrapped opaque callable, flagged as a preservation risk (plan A.3.1).
"""

from __future__ import annotations

import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from graphed import Array, Session
from graphed_core import PayloadDescriptor

_ARITH = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}


@dataclass(frozen=True)
class NumpyForm:
    """Opaque form for a numpy bag: a dtype + a kind (``vector`` or ``scalar``)."""

    dtype: np.dtype
    kind: str = "vector"

    def describe(self) -> str:
        return f"{self.kind}[{self.dtype}]"


def _is_numeric(form: NumpyForm) -> bool:
    return np.issubdtype(form.dtype, np.number)


class NumpyBackend:
    """A `graphed.Backend` over numpy arrays."""

    def op_form(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> NumpyForm:
        forms = [f for f in inputs if isinstance(f, NumpyForm)]
        if op in _ARITH:
            a, b = forms
            if not (_is_numeric(a) and _is_numeric(b)):
                raise TypeError(f"{op} requires numeric operands, got {a.describe()} and {b.describe()}")
            return NumpyForm(np.promote_types(a.dtype, b.dtype))
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
            # opaque callable: result form is unknown; treat as an object vector.
            return NumpyForm(np.dtype(object))
        raise TypeError(f"unsupported op {op!r}")

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        if op in _ARITH:
            return _ARITH[op](inputs[0], inputs[1])
        if op == "filter":
            return np.asarray(inputs[0])[np.asarray(inputs[1])]
        if op == "sum":
            return np.sum(inputs[0])
        raise TypeError(f"unsupported op {op!r}")

    def boundary_ops(self) -> frozenset[str]:
        return frozenset({"source", "sum", "map"})

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        # Column projection is M5; trivial backend reads everything.
        return used

    def external_payload(self, op: str, params: Mapping[str, object]) -> PayloadDescriptor | None:
        if op != "map":
            return None
        fn_name = str(params.get("fn", "lambda"))
        # M2 cannot content-hash the callable body; flag it as a preservation risk. M9 does the
        # real hashing. The fn name keeps distinct callables interning to distinct nodes (the
        # node's params also carry it).
        return PayloadDescriptor(
            kind="opaque_callable",
            content_hash=f"unhashed-opaque:{fn_name}",
            framework="python",
            version=platform.python_version(),
            io_schema="opaque->opaque",
            preprocessing_ref=None,
        )


def from_array(session: Session, name: str, values: object) -> Array:
    """Create a source Array from a numpy array (or anything array-like)."""
    arr = np.asarray(values)
    return session.source(name, form=NumpyForm(arr.dtype), data=arr)


__all__ = ["NumpyBackend", "NumpyForm", "from_array"]
__version__ = "0.0.1"
