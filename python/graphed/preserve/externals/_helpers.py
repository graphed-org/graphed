"""Shared helpers for the ML-framework plugins (M26)."""

from __future__ import annotations

import gzip
import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, TypeVar

from ..errors import PreserveError


def json_bytes(payload: bytes) -> bytes:
    """Return a JSON payload's bytes, gunzipped when gzipped.

    CMS POGs ship correction sets as .json.gz.
    """
    return gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload


def _canonical_json_hash(domain: bytes, payload: bytes) -> str:
    """sha256 over domain-separated, canonicalized JSON (key order + whitespace = formatting)."""
    return memoized_model_hash(
        domain.decode(), payload, lambda p: _canonical_json_hash_impl(domain, p)
    )


def _canonical_json_hash_impl(domain: bytes, payload: bytes) -> str:
    try:
        parsed = json.loads(json_bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, ValueError, OSError, EOFError) as err:
        raise PreserveError(f"payload is not valid JSON ({err})") from err
    canon = json.dumps(parsed, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(domain + canon.encode("utf-8")).hexdigest()


def _stack_feature_columns(inputs: list[Any]) -> Any:
    """Per-event feature arrays -> one (n_events, n_features) matrix in their common dtype.

    A framework that needs one dtype casts it to the model's own.
    """
    import awkward as ak
    import numpy as np

    return np.stack([np.asarray(ak.to_numpy(ak.Array(i))) for i in inputs], axis=1)


def _as_event_array(out: Any) -> Any:
    """Return a model's (n,) or (n, 1) output as one value per event.

    (n, k, ...) as ``ak.from_numpy`` gives it, regular rows of k.
    """
    import awkward as ak
    import numpy as np

    values = np.asarray(out, dtype="float64")
    return ak.from_numpy(values.reshape(-1) if values.shape[1:] == (1,) else values)


def _strip_config_names(obj: Any) -> Any:
    """Drop auto-generated layer ``name`` entries from a keras config: incidental, not content."""
    if isinstance(obj, dict):
        return {k: _strip_config_names(v) for k, v in obj.items() if k != "name"}
    if isinstance(obj, list):
        return [_strip_config_names(v) for v in obj]
    return obj


# ---- M27: variadic call templates ---------------------------------------------------------------
# params["args"] (positional) / params["kwargs"] (keyword) route a node's graph inputs to the
# callee's REAL signature. Entries: "$i" (input slot i), ["$i", "$j"] (a group — for ML plugins,
# stacked into one feature matrix), or — where a plugin allows it — a constant (correctionlib's
# systematic names). Stored in the IR as canonical-JSON strings (the ParamMap is scalar-typed);
# accepted here as either the decoded structure or the JSON string. Replay obeys the template
# exactly; an unknown shape is a loud PreserveError, never a guess.

TemplateEntry = tuple[str, Any]  # ("slot", i) | ("group", [i, ...]) | ("const", value)


def _decode_spec(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError as err:
            raise PreserveError(f"call template is not valid JSON: {value!r}") from err
    return value


def _parse_entry(
    entry: Any, n_inputs: int, *, allow_constants: bool, allow_groups: bool
) -> TemplateEntry:
    if isinstance(entry, str) and entry.startswith("$"):
        i = int(entry[1:])
        if not 0 <= i < n_inputs:
            raise PreserveError(
                f"call template slot {entry} out of range (node has {n_inputs} inputs)"
            )
        return ("slot", i)
    if isinstance(entry, list):
        if not allow_groups:
            raise PreserveError(
                "this plugin's callee takes scalar/array arguments, not stacked groups"
            )
        return (
            "group",
            [
                _parse_entry(e, n_inputs, allow_constants=False, allow_groups=False)[1]
                for e in entry
            ],
        )
    if allow_constants and isinstance(entry, (str, int, float, bool)):
        return ("const", entry)
    raise PreserveError(f"call template entry {entry!r} is not a slot, group, or allowed constant")


def parse_call_template(
    params: Any,
    n_inputs: int,
    *,
    allow_constants: bool = False,
    allow_groups: bool = True,
    allow_kwargs: bool = True,
) -> tuple[list[TemplateEntry], dict[str, TemplateEntry]] | None:
    """Parse the node's call template; ``None`` when absent (callers use their legacy shape)."""
    raw_args = _decode_spec(params.get("args")) if params.get("args") is not None else None
    raw_kwargs = _decode_spec(params.get("kwargs")) if params.get("kwargs") is not None else None
    if raw_args is None and raw_kwargs is None:
        return None
    if raw_kwargs and not allow_kwargs:
        raise PreserveError("this plugin's callee does not take keyword arguments")
    kwargs: dict[str, TemplateEntry] = {
        str(k): _parse_entry(
            v, n_inputs, allow_constants=allow_constants, allow_groups=allow_groups
        )
        for k, v in (raw_kwargs or {}).items()
    }
    if isinstance(raw_args, dict):
        # the NAMED form (onnx feeds, triton InferInputs): named protocol inputs, not python kwargs
        named = {
            str(k): _parse_entry(
                v, n_inputs, allow_constants=allow_constants, allow_groups=allow_groups
            )
            for k, v in raw_args.items()
        }
        return ([("named", named)], kwargs)
    args = [
        _parse_entry(e, n_inputs, allow_constants=allow_constants, allow_groups=allow_groups)
        for e in (raw_args or [])
    ]
    return (args, kwargs)


def ml_matrix(entry: TemplateEntry, inputs: list[Any]) -> Any:
    """Materialize one template entry as the ML convention: an (n_events, k) matrix."""
    kind, value = entry
    if kind == "slot":
        return _stack_feature_columns([inputs[value]])
    if kind == "group":
        return _stack_feature_columns([inputs[i] for i in value])
    raise PreserveError(f"constants are not valid model inputs (got {value!r})")


# ---- M36: memoize the EXPENSIVE (model-parsing) content hashes ----------------------------------
# A model's content hash is computed at record time AND re-verified at bundle-build time; each call
# re-parses the model (onnx.load, torch.jit.load, keras load, jax deserialize). Cache the result by
# the payload bytes so the same payload parses once. Bounded FIFO; keyed per (domain, bytes) so
# distinct plugins never collide. Cold in a fresh process, so the by-value validate subprocess
# still computes correctly.
# keyed by the payload itself: bytes cache their hash, so a repeat with the same object is O(1)
_HASH_MEMO: OrderedDict[tuple[str, bytes], Any] = OrderedDict()
_HASH_MEMO_CAP = 64  # also bounds the payloads the keys keep alive
_HASH_MEMO_LOCK = threading.Lock()
_T = TypeVar("_T")


def memoized_model_hash(domain: str, payload: bytes, compute: Callable[[bytes], _T]) -> _T:
    key = (domain, bytes(payload))
    with _HASH_MEMO_LOCK:  # get + move_to_end must not interleave with another thread's eviction
        if key in _HASH_MEMO:
            _HASH_MEMO.move_to_end(key)
            cached: _T = _HASH_MEMO[key]
            return cached
    result = compute(payload)
    with _HASH_MEMO_LOCK:
        _HASH_MEMO[key] = result
        while len(_HASH_MEMO) > _HASH_MEMO_CAP:
            _HASH_MEMO.popitem(last=False)
    return result
