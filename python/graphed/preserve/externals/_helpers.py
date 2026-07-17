"""Shared helpers for the ML-framework plugins (M26)."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from ..errors import PreserveError


def _canonical_json_hash(domain: bytes, payload: bytes) -> str:
    """sha256 over domain-separated, canonicalized JSON (key order + whitespace = formatting)."""
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as err:
        raise PreserveError(f"payload is not valid JSON ({err})") from err
    canon = json.dumps(parsed, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(domain + canon.encode("utf-8")).hexdigest()


def _stack_feature_columns(inputs: list[Any]) -> Any:
    """Per-event feature arrays -> one float32 (n_events, n_features) matrix (the M9 convention)."""
    import awkward as ak  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    cols = [np.asarray(ak.to_numpy(ak.Array(i)), dtype="float32") for i in inputs]
    return np.stack(cols, axis=1)


def _as_event_array(out: Any) -> Any:
    import awkward as ak  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    return ak.Array(np.asarray(out, dtype="float64").reshape(-1))


def _strip_config_names(obj: Any) -> Any:
    """Drop auto-generated layer ``name`` entries from a keras config: incidental, not content."""
    if isinstance(obj, dict):
        return {k: _strip_config_names(v) for k, v in obj.items() if k != "name"}
    if isinstance(obj, list):
        return [_strip_config_names(v) for v in obj]
    return obj


# ---- M27: variadic call templates ----------------------------------------------------------------
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


def _parse_entry(entry: Any, n_inputs: int, *, allow_constants: bool, allow_groups: bool) -> TemplateEntry:
    if isinstance(entry, str) and entry.startswith("$"):
        i = int(entry[1:])
        if not 0 <= i < n_inputs:
            raise PreserveError(f"call template slot {entry} out of range (node has {n_inputs} inputs)")
        return ("slot", i)
    if isinstance(entry, list):
        if not allow_groups:
            raise PreserveError("this plugin's callee takes scalar/array arguments, not stacked groups")
        return (
            "group",
            [_parse_entry(e, n_inputs, allow_constants=False, allow_groups=False)[1] for e in entry],
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
        str(k): _parse_entry(v, n_inputs, allow_constants=allow_constants, allow_groups=allow_groups)
        for k, v in (raw_kwargs or {}).items()
    }
    if isinstance(raw_args, dict):
        # the NAMED form (onnx feeds, triton InferInputs): named protocol inputs, not python kwargs
        named = {
            str(k): _parse_entry(v, n_inputs, allow_constants=allow_constants, allow_groups=allow_groups)
            for k, v in raw_args.items()
        }
        return ([("named", named)], kwargs)
    args = [
        _parse_entry(e, n_inputs, allow_constants=allow_constants, allow_groups=allow_groups)
        for e in (raw_args or [])
    ]
    return (args, kwargs)


def ml_matrix(entry: TemplateEntry, inputs: list[Any]) -> Any:
    """Materialize one template entry as the ML convention: a float32 (n_events, k) matrix."""
    kind, value = entry
    if kind == "slot":
        return _stack_feature_columns([inputs[value]])
    if kind == "group":
        return _stack_feature_columns([inputs[i] for i in value])
    raise PreserveError(f"constants are not valid model inputs (got {value!r})")


# ---- M36: memoize the EXPENSIVE (model-parsing) content hashes -----------------------------------
# A model's content hash is computed at record time AND re-verified at bundle-build time; each call
# re-parses the model (onnx.load, torch.jit.load, keras load, jax deserialize). Cache the result by
# a cheap raw-bytes digest so the same payload parses once. Bounded FIFO; keyed per (domain, bytes)
# so distinct plugins never collide. Cold in a fresh process, so the by-value validate subprocess
# still computes correctly.
_HASH_MEMO: OrderedDict[str, str] = OrderedDict()
_HASH_MEMO_CAP = 64


def memoized_model_hash(domain: str, payload: bytes, compute: Callable[[bytes], str]) -> str:
    key = domain + ":" + hashlib.sha256(payload).hexdigest()
    cached = _HASH_MEMO.get(key)
    if cached is not None:
        _HASH_MEMO.move_to_end(key)
        return cached
    result = compute(payload)
    _HASH_MEMO[key] = result
    while len(_HASH_MEMO) > _HASH_MEMO_CAP:
        _HASH_MEMO.popitem(last=False)
    return result
