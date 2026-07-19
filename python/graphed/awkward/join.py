"""The awkward ``JoinBackend`` relational-join primitives (plan M40 §3.0, §3.3, §2.1).

The M39 exchange half lives in :mod:`graphed.awkward.shuffle`; this is the M40 join half the generic
radix-hash kernel calls, plus the ``pack_key`` column-builder and the ``join``/``pack_key`` form and
evaluation dispatch. Semantics are RELATIONAL (SQL/pandas ``merge``): a probe row with k build
matches yields k aligned pairs — NOT a list-of-matches, NOT a grouped shape (grouped is the
awkward-only ``gak.join(grouped=True)`` post-op).

Load-bearing invariants:
  * ``take``'s ``-1`` sentinel is a MISS → an awkward OPTION ``None`` row (``ak.mask``), never the last
    row (an ``np.take``/negative-index gather is plan M40 trap #1);
  * ``pack_key`` is a deterministic big-endian integer bit-packing — ``run`` most-significant …
    ``event`` least-significant, 20-bit stride so zero high fields contribute zero bits — NEVER Python
    ``hash()`` (trap #6, reproducibility §A.3.1); both backends compute it identically;
  * ``merge_records`` is the flat field-union minus the duplicate ``on`` key (left-wins on any other
    name collision, so joining on the natural keys never collides the carried key columns).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import awkward as ak
import numpy as np

#: field-width stride (bits) of the big-endian pack — 20 bits/field fits run<2^24, lumi/event<2^20
#: into a u64 (plan §2.1; overflow→fixed-width-bytes is Phase-2).
_PACK_STRIDE = 20


def on_from_params(params: Mapping[str, Any]) -> list[str]:
    """The key field names from an op's ``on`` param (comma-encoded, the §A.4 ParamValue rule)."""
    return [f for f in str(params.get("on", "")).split(",") if f]


def _key_rows(block: ak.Array, on: Sequence[str]) -> list[tuple[Any, ...]]:
    """One hashable key tuple per row, over the ``on`` fields (multi-field keys join as a tuple)."""
    cols = [np.asarray(ak.to_numpy(block[c])).tolist() for c in on]
    return list(zip(*cols, strict=True))


def match_indices(
    build: ak.Array, probe: ak.Array, *, on: Sequence[str], how: str = "inner"
) -> tuple[np.ndarray, np.ndarray]:
    """Relationally match two co-partitioned blocks on ``on`` → aligned ``(build_idx, probe_idx)``
    WITH duplication: a build row with k probe matches yields k aligned pairs (SQL/pandas ``merge``).
    Pairs are emitted in build-row order (then probe order) — deterministic, and non-decreasing in the
    build index so ``gak.join(grouped=True)`` can regroup by run-length. ``how=left``/``outer`` keep an
    unmatched BUILD row as ``(i, -1)``; ``how=right``/``outer`` keep an unmatched PROBE row as
    ``(-1, j)``. A ``-1`` feeds :func:`take` as an option ``None`` (plan M40 §3.3)."""
    bkeys = _key_rows(build, on)
    pkeys = _key_rows(probe, on)
    pmap: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for j, kv in enumerate(pkeys):
        pmap[kv].append(j)
    b_out: list[int] = []
    p_out: list[int] = []
    matched: set[int] = set()
    for i, kv in enumerate(bkeys):
        ms = pmap.get(kv)
        if ms:
            for j in ms:
                b_out.append(i)
                p_out.append(j)
                matched.add(j)
        elif how in ("left", "outer"):
            b_out.append(i)
            p_out.append(-1)
    if how in ("right", "outer"):
        for j in range(len(pkeys)):
            if j not in matched:
                b_out.append(-1)
                p_out.append(j)
    return np.asarray(b_out, dtype=np.int64), np.asarray(p_out, dtype=np.int64)


def take(block: ak.Array, index: np.ndarray) -> ak.Array:
    """Gather rows of ``block`` by ``index``. A ``-1`` entry is a MISS → an awkward OPTION ``None``
    row (``ak.mask``), NEVER the last row — a plain ``block[index]`` would read ``block[-1]`` and pass a
    row-count check while silently corrupting the value (plan M40 trap #1)."""
    idx = np.asarray(index).astype(np.int64)
    valid = idx >= 0
    gathered = block[np.where(valid, idx, np.int64(0))]  # clamp misses to 0, then mask them out
    if bool(valid.all()):
        return gathered
    return ak.mask(gathered, valid)


def merge_records(left: ak.Array, right: ak.Array, *, on: Sequence[str]) -> ak.Array:
    """Flat field-union of two aligned taken blocks, deduped by field-name INTERSECTION (LEFT wins):
    LEFT contributes every field; RIGHT contributes each field LEFT does not already carry. A join on
    the natural keys packs run/lumi/event + ``__joinkey__`` onto BOTH sides, so only intersection-dedup
    yields a clean flat record; ``on`` is unused (protocol-required; :func:`join_form` mirrors this).
    One flat record per aligned row; option rows from :func:`take` survive as option fields (M40 §3.3).
    ponytail: no SQL-style suffixing of a shared NON-key column — add if an analysis ever joins tables
    sharing a payload field name."""
    fields: dict[str, ak.Array] = {f: left[f] for f in ak.fields(left)}
    for f in ak.fields(right):
        if f not in fields:
            fields[f] = right[f]
    return ak.zip(fields, depth_limit=1)


def pack_key(rec: ak.Array, on: Sequence[str]) -> ak.Array:
    """Add a flat u64 ``__joinkey__`` column to ``rec``, big-endian bit-packed over ``on`` (first field
    most-significant). Integer bit-ops only — deterministic, process-independent, injective on in-range
    keys, and ``pack_key(0,…,0,e) == e``. NEVER Python ``hash()`` (plan M40 trap #6; §A.3.1)."""
    n = len(on)
    if n == 0:
        raise ValueError("pack_key needs at least one key field")
    key: ak.Array | None = None
    for i, f in enumerate(on):
        shifted = ak.values_astype(rec[f], np.uint64) << np.uint64((n - 1 - i) * _PACK_STRIDE)
        key = shifted if key is None else key | shifted
    return ak.with_field(rec, ak.values_astype(key, np.uint64), where="__joinkey__")


def _optional(arr: ak.Array) -> ak.Array:
    """Option-wrap a leaf so its form carries a ``?`` (how=left/outer makes the missing side option-
    typed). Form-only: on op_form's non-reporting typetracers this reads no data."""
    return ak.mask(arr, arr == arr)


def join_form(inputs: Sequence[ak.Array], params: Mapping[str, Any]) -> ak.Array:
    """The op_form("join") output form: the relational field-union (see :func:`merge_records`) with the
    missing side option-typed under ``how=left``/``right``/``outer`` (plan M40 §3.3, a3). Built
    structurally on typetracers — no matching, no data read."""
    left, right = inputs[0], inputs[1]
    how = str(params.get("how", "inner"))
    left_opt = how in ("right", "outer")
    right_opt = how in ("left", "outer")
    fields: dict[str, ak.Array] = {f: (_optional(left[f]) if left_opt else left[f]) for f in ak.fields(left)}
    for f in ak.fields(right):
        if f in fields:  # intersection-dedup, mirroring merge_records (`on` ignored)
            continue
        fields[f] = _optional(right[f]) if right_opt else right[f]
    return ak.zip(fields, depth_limit=1)


def join_grouped(left: ak.Array, right: ak.Array, *, on: Sequence[str], how: str = "inner") -> ak.Array:
    """``gak.join(grouped=True)``: the relational result regrouped by a DETERMINISTIC ``ak.unflatten``
    whose run-lengths are the per-build-row match counts (``match_indices`` emits build-major, so the
    aligned build-index array run-lengths ARE those counts). Flattening it reproduces the relational
    result exactly; the outer length is the number of build rows that matched (plan M40 a4). Awkward-
    only (numpy has no ``gak``). ponytail: grouping is per build row (inner/left); an outer join's
    trailing unmatched-probe rows group together — fine until an outer grouped join is a real need."""
    bi, pi = match_indices(left, right, on=on, how=how)
    flat = merge_records(take(left, bi), take(right, pi), on=on)
    counts = ak.run_lengths(ak.Array(bi))
    return ak.unflatten(flat, counts)
