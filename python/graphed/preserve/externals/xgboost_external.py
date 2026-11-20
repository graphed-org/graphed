"""The ``xgboost_model`` plugin (M26): XGBoost's JSON or UBJSON model; canonical-JSON hash."""

from __future__ import annotations

import struct
from collections.abc import Mapping
from typing import Any

from ..errors import PreserveError
from ._base import ExternalPlugin
from ._helpers import (
    _as_event_array,
    _canonical_json_hash_impl,
    _stack_feature_columns,
    canonical_document_hash,
    json_bytes,
    memoized_model_hash,
    ml_matrix,
    parse_call_template,
)

_DOMAIN = b"xgboost-json-v1"


def xgboost_content_hash(payload: bytes) -> str:
    """Canonical JSON of the model document, from its JSON or UBJSON bytes — dependency-free.

    One booster saved as ``.json`` and as ``.ubj`` hashes the same: a UBJSON float32 reads as the
    shortest decimal that round-trips it, which is what XGBoost's JSON writer prints.
    """
    return memoized_model_hash(_DOMAIN.decode(), payload, _xgboost_hash_impl)


def _xgboost_hash_impl(payload: bytes) -> str:
    if not _is_ubjson(payload):
        try:
            return _canonical_json_hash_impl(_DOMAIN, payload)
        except PreserveError as err:
            raise PreserveError(
                f"an xgboost payload is JSON, gzipped JSON or UBJSON; this is none ({err})"
            ) from err
    try:
        return canonical_document_hash(_DOMAIN, _ubjson(payload))
    except (ValueError, IndexError, UnicodeDecodeError, struct.error, RecursionError) as err:
        raise PreserveError(f"an xgboost payload is not valid UBJSON ({err})") from err


def _is_ubjson(payload: bytes) -> bool:
    # a UBJSON object opens with a typed key length or a container marker, JSON with a quote
    return payload[:1] == b"{" and payload[1:2] in (b"i", b"U", b"I", b"l", b"L", b"$", b"#")


#: UBJSON's fixed-size numeric markers (big-endian)
_NUMBERS = {"i": ">b", "U": ">B", "I": ">h", "l": ">i", "L": ">q", "d": ">f", "D": ">d"}
_CONSTANTS = {"Z": None, "T": True, "F": False}


def _ubjson(buf: bytes) -> Any:
    """Decode UBJSON (as XGBoost writes it) to the document ``json.loads`` gives for its JSON."""
    import numpy as np

    pos = 0

    def marker() -> str:
        nonlocal pos
        pos += 1
        return chr(buf[pos - 1])

    def size() -> int:
        kind = marker()
        if kind not in "iUIlL":
            raise ValueError(f"UBJSON length at byte {pos} is not an integer")
        n: int = value(kind)
        if n < 0:
            raise ValueError(f"UBJSON length at byte {pos} is negative")
        return n

    def text() -> str:
        nonlocal pos
        n = size()
        pos += n
        return buf[pos - n : pos].decode("utf-8")

    def value(kind: str) -> Any:
        nonlocal pos
        if kind in _NUMBERS:
            (number,) = struct.unpack_from(_NUMBERS[kind], buf, pos)
            pos += struct.calcsize(_NUMBERS[kind])
            return float(str(np.float32(number))) if kind == "d" else number
        if kind == "S":
            return text()
        if kind in "[{":
            return container(kind == "{")
        if kind in _CONSTANTS:
            return _CONSTANTS[kind]
        raise ValueError(f"UBJSON marker {kind!r} at byte {pos - 1}")

    def container(is_object: bool) -> Any:
        nonlocal pos
        typed = count = None
        if buf[pos : pos + 1] == b"$":
            pos += 1
            typed = marker()
        if buf[pos : pos + 1] == b"#":
            pos += 1
            count = size()
        if not is_object and count is not None and typed in _NUMBERS:  # a typed array, in bulk
            numbers: np.ndarray[Any, Any] = np.frombuffer(buf, _NUMBERS[typed], count, pos)
            pos += numbers.nbytes
            return (numbers.astype(str).astype(float) if typed == "d" else numbers).tolist()
        items: list[Any] = []
        while count is None or len(items) < count:
            if count is None and buf[pos : pos + 1] == (b"}" if is_object else b"]"):
                pos += 1
                break
            key = text() if is_object else None
            items.append((key, value(typed or marker())))
        return dict(items) if is_object else [item for _, item in items]

    document = value(marker())
    if pos != len(buf):
        raise ValueError(f"{len(buf) - pos} trailing bytes")
    return document


def load_xgboost(payload: bytes, params: Mapping[str, Any]) -> Any:
    """Load an XGBoost booster from its JSON or UBJSON payload."""
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(bytearray(json_bytes(payload)))  # UBJSON passes through ungzipped
    return booster


def eval_xgboost(booster: Any, params: Mapping[str, Any], inputs: list[Any]) -> Any:
    """Predict with the booster over ``inputs``, one value per event."""
    template = parse_call_template(params, len(inputs), allow_kwargs=False)
    if template is not None:
        args, _ = template
        if len(args) != 1 or args[0][0] not in ("group", "slot"):
            raise PreserveError(
                "xgboost models take a single tabular matrix: the call template must be "
                'exactly one positional group (e.g. [["$0", "$1"]])'
            )
        matrix = ml_matrix(args[0], inputs)
    else:
        matrix = _stack_feature_columns(inputs)
    import numpy as np
    import xgboost as xgb

    # xgboost reads absent trailing columns as missing values instead of refusing them
    expected = booster.num_features()
    if matrix.shape[1] != expected:
        raise PreserveError(
            f"the xgboost model takes {expected} features; the call gives {matrix.shape[1]}"
        )
    names = booster.feature_names  # its own, so a model trained on named columns runs as shipped
    if not len(matrix):  # an empty DMatrix predicts shape (0, 0), losing the output width
        one = booster.predict(xgb.DMatrix(np.zeros((1, expected)), feature_names=names))
        return _as_event_array(one[:0])
    return _as_event_array(booster.predict(xgb.DMatrix(matrix, feature_names=names)))


def _xgboost_samples() -> list[bytes]:
    # hash validation needs structurally-plausible distinct JSON documents, not trained models
    return [
        b'{"learner":{"attributes":{},"gradient_booster":{"model":{"trees":[{"split":6.0}]}}},"version":[2,0,0]}',
        b'{"learner":{"attributes":{},"gradient_booster":{"model":{"trees":[{"split":9.5}]}}},"version":[2,0,0]}',
    ]


XGBOOST_PLUGIN = ExternalPlugin(
    kind="xgboost_model",
    content_hash=xgboost_content_hash,
    evaluate=eval_xgboost,
    samples=_xgboost_samples,
    load=load_xgboost,
    framework="xgboost",
    output_dtype="float64",
)
