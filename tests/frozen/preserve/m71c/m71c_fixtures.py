"""Fixtures for the m71c suite: byte-literal correctionlib/ONNX payloads and two awkward sessions.

The `output_dtype=` plugin field is m71c-new, so every plugin carrying it is built inside a test
body; this module collects on a base without it.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import awkward as ak
import numpy as np

from graphed import GraphedTypeError, Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.numpy import NumpyBackend, from_array
from graphed.preserve.externals import ExternalPlugin, sha256_bytes

CSET = (
    b'{"schema_version": 2, "corrections": [{"name": "sf", "version": 1, "inputs": [{"name": '
    b'"systematic", "type": "string"}, {"name": "x", "type": "real"}], "output": {"name": "sf", '
    b'"type": "real"}, "data": {"nodetype": "category", "input": "systematic", "content": [{"key": '
    b'"nominal", "value": 1.0}]}}]}'
)
MODEL = (
    b'\x08\t:]\n\x12\n\x01x\n\x01W\n\x01B\x12\x01y"\x04Gemm\x12\x01m*\x0f\x08\x01\x08\x01\x10\x01B'
    b"\x01WJ\x04\x00\x00\x00?*\r\x08\x01\x10\x01B\x01BJ\x04\x00\x00\x00\x00Z\x11\n\x01x\x12\x0c\n\n"
    b"\x08\x01\x12\x06\n\x00\n\x02\x08\x01b\x11\n\x01y\x12\x0c\n\n\x08\x01\x12\x06\n\x00\n\x02\x08"
    b"\x01B\x04\n\x00\x10\r"
)
SF = {"name": "sf", "args": ["nominal", "$0"]}

EVENTS = ak.zip(
    {
        "run": np.array([1, 2, 1, 3], dtype=np.uint32),
        "x": np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32),
    }
)

#: every numpy numerical dtype (``np.typecodes`` "AllInteger" + "AllFloat" + bool, fixed-width only)
NUMERIC = (
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
)


def _evaluator(*values: object) -> object:
    raise AssertionError("the args= template path never calls the evaluator")


def _samples() -> list[bytes]:
    return [b"a", b"b"]


def recorded() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    return session, from_awkward(session, "events", EVENTS)


def numpy_recorded() -> tuple[Session, Any]:
    session = Session(NumpyBackend())
    return session, from_array(session, "x", np.array([0.5, 1.5], dtype=np.float32))


def plugin(kind: str, dtype: str, **fields: Any) -> ExternalPlugin:
    """A plugin whose value casts its first input to `dtype`; `fields` carries `output_dtype=`."""

    def evaluate(resource: Any, params: Mapping[str, Any], inputs: Sequence[Any]) -> Any:
        return ak.values_astype(inputs[0], dtype)

    return ExternalPlugin(kind=kind, content_hash=sha256_bytes, evaluate=evaluate, samples=_samples, **fields)


def describe(session: Session, array: Any) -> str:
    return str(session.form(array).describe())


def same(value: Any, expected: Any) -> bool:
    """Equal values AND equal types."""
    return bool(ak.to_list(value) == ak.to_list(expected) and str(ak.type(value)) == str(ak.type(expected)))


def refusal(call: Callable[[], object]) -> GraphedTypeError:
    try:
        call()
    except GraphedTypeError as exc:
        return exc
    raise AssertionError("expected a GraphedTypeError")


def here() -> int:
    """The caller's NEXT line: the line a one-line refused call sits on."""
    return sys._getframe(1).f_lineno + 1
