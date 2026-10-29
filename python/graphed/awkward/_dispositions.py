"""§2.3c: each public gak function's dispatch classification, and §2.3e's argument fixtures.

An UNDISPOSED function does not fail loudly — `Varied`'s label-mapping `getattr` turns an
unhandled duck-typed read into a recorded op and silently compiles nonsense — so the table below
is exhaustive over :mod:`graphed.awkward.functions`' public surface and a frozen gate enumerates
it dynamically. A new gak function is fixed HERE, never by editing that test.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

#   *broadcast*             elementwise/structural, one recorded call per universe
#   *container-traversing*  the arrays live INSIDE a Mapping/Sequence argument
#   *tuple-returning*       the wrapper rebuilds its result, so it emits one Varied per position
#   *eager-metadata*        answers on the nominal member (sound by §2.1's form compatibility)
#   *refusing*              §5.4 boundary verbs
_GAK_EAGER_METADATA = ("backend_of", "fields", "head", "sample", "to_list", "type_of")
_GAK_REFUSING = ("join",)
_GAK_TUPLE_RETURNING = ("broadcast_arrays", "unzip")
_GAK_CONTAINER_TRAVERSING = (
    "apply_correction", "argcartesian", "cartesian", "concatenate", "onnx_inference", "zip",
)  # fmt: skip
_GAK_BROADCAST = (
    "all", "any", "argcombinations", "argmax", "argmin", "argsort", "combinations", "corr",
    "count", "count_nonzero", "covar", "drop_none", "fill_none", "firsts", "flatten",
    "from_regular", "full_like", "is_none", "isclose", "linear_fit", "local_index", "mask", "max",
    "mean", "min", "moment", "nan_to_num", "num", "ones_like", "pad_none", "prod", "ptp", "ravel",
    "run_lengths", "singletons", "softmax", "sort", "std", "sum", "to_regular", "unflatten",
    "values_astype", "var", "where", "with_field", "with_name", "with_parameter", "without_field",
    "without_parameters", "zeros_like",
)  # fmt: skip

GAK_DISPOSITIONS: dict[str, str] = {
    **dict.fromkeys(_GAK_BROADCAST, "broadcast"),
    **dict.fromkeys(_GAK_CONTAINER_TRAVERSING, "container-traversing"),
    **dict.fromkeys(_GAK_TUPLE_RETURNING, "tuple-returning"),
    **dict.fromkeys(_GAK_EAGER_METADATA, "eager-metadata"),
    **dict.fromkeys(_GAK_REFUSING, "refusing"),
}


@dataclass(frozen=True)
class GakSlot:
    """A substitution SLOT in an argument fixture, naming the operand KIND it needs.

    graphed type-checks the primary operand at RECORD time through the backend's `op_form`, so one
    array cannot serve every function; §2.3e's propagation gate owns one contexted array per kind
    and substitutes the one a slot names. The vocabulary is frozen at m48: flat numeric, jagged
    numeric, record, boolean mask, option type.
    """

    kind: str


@dataclass(frozen=True)
class GakArgFixture:
    """The AUXILIARY arguments §2.3e's propagation gate needs to call one function.

    It lives here beside the classification, not in the frozen test, so a new gak function arrives
    with both and the test stays untouched. The CONTEXTED operand is the test's, substituted into
    the slots below — a fixture-supplied context-free primary would degrade the check to
    `None == None`.
    """

    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)


def _echo(*values: object, **_kwargs: object) -> object:
    """Return a stand-in payload evaluator for the two External-recording fixtures below."""
    return values[0]


_FLAT = GakSlot("flat")
_JAGGED = GakSlot("jagged")
_RECORD = GakSlot("record")
_MASK = GakSlot("mask")
_OPTION = GakSlot("option")
_INNER = {"axis": 1}

GAK_ARG_FIXTURES: dict[str, GakArgFixture] = {
    "all": GakArgFixture((_MASK,), _INNER),
    "any": GakArgFixture((_MASK,), _INNER),
    "apply_correction": GakArgFixture(
        (b'{"schema_version":2,"corrections":[]}', "sf", [_FLAT], _echo), {"args": ["$0"]}
    ),
    "argcartesian": GakArgFixture(([_JAGGED, _JAGGED],)),
    "argcombinations": GakArgFixture((_JAGGED, 2)),
    "argmax": GakArgFixture((_JAGGED,), _INNER),
    "argmin": GakArgFixture((_JAGGED,), _INNER),
    "argsort": GakArgFixture((_JAGGED,)),
    "broadcast_arrays": GakArgFixture((_FLAT, _JAGGED)),
    "cartesian": GakArgFixture(([_JAGGED, _JAGGED],)),
    "combinations": GakArgFixture((_JAGGED, 2)),
    "concatenate": GakArgFixture(([_FLAT, _FLAT],)),
    "corr": GakArgFixture((_JAGGED, _JAGGED), _INNER),
    "count": GakArgFixture((_JAGGED,), _INNER),
    "count_nonzero": GakArgFixture((_JAGGED,), _INNER),
    "covar": GakArgFixture((_JAGGED, _JAGGED), _INNER),
    "drop_none": GakArgFixture((_OPTION,)),
    "fill_none": GakArgFixture((_OPTION, 0.0)),
    "firsts": GakArgFixture((_JAGGED,)),
    "flatten": GakArgFixture((_JAGGED,)),
    "from_regular": GakArgFixture((_JAGGED,)),
    "full_like": GakArgFixture((_JAGGED, 1.0)),
    "is_none": GakArgFixture((_OPTION,)),
    "isclose": GakArgFixture((_JAGGED, _JAGGED)),
    "linear_fit": GakArgFixture((_JAGGED, _JAGGED), _INNER),
    "local_index": GakArgFixture((_JAGGED,)),
    "mask": GakArgFixture((_JAGGED, _MASK)),
    "max": GakArgFixture((_JAGGED,), _INNER),
    "mean": GakArgFixture((_JAGGED,), _INNER),
    "min": GakArgFixture((_JAGGED,), _INNER),
    "moment": GakArgFixture((_JAGGED, 2), _INNER),
    "nan_to_num": GakArgFixture((_JAGGED,)),
    "num": GakArgFixture((_JAGGED,)),
    "ones_like": GakArgFixture((_JAGGED,)),
    # b"" is a valid (empty) ModelProto: the weights-identity path needs a PARSEABLE model, not a
    # real one, and the path-based form would need a file on disk.
    "onnx_inference": GakArgFixture((b"", [_FLAT], _echo), {"args": [["$0"]]}),
    "pad_none": GakArgFixture((_JAGGED, 2)),
    "prod": GakArgFixture((_JAGGED,), _INNER),
    "ptp": GakArgFixture((_JAGGED,), _INNER),
    "ravel": GakArgFixture((_JAGGED,)),
    "run_lengths": GakArgFixture((_FLAT,)),
    "singletons": GakArgFixture((_OPTION,)),
    "softmax": GakArgFixture((_JAGGED,)),
    "sort": GakArgFixture((_JAGGED,)),
    "std": GakArgFixture((_JAGGED,), _INNER),
    "sum": GakArgFixture((_JAGGED,), _INNER),
    "to_regular": GakArgFixture((_JAGGED,)),
    "unflatten": GakArgFixture((_JAGGED, 2), _INNER),
    "unzip": GakArgFixture((_RECORD,)),
    "values_astype": GakArgFixture((_JAGGED, "float32")),
    "var": GakArgFixture((_JAGGED,), _INNER),
    "where": GakArgFixture((_MASK, 1.0, 0.0)),
    "with_field": GakArgFixture((_RECORD, _JAGGED, "extra")),
    "with_name": GakArgFixture((_RECORD, "Thing")),
    "with_parameter": GakArgFixture((_RECORD, "key", "value")),
    "without_field": GakArgFixture((_RECORD, "pt")),
    "without_parameters": GakArgFixture((_RECORD,)),
    "zeros_like": GakArgFixture((_JAGGED,)),
    "zip": GakArgFixture(({"pt": _JAGGED, "eta": _JAGGED},)),
}
