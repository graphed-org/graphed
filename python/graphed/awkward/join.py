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
  * ``merge_records`` COALESCES the shared ``on`` key (present side wins, kept NON-option since a
    coalesced key is never null) and REJECTS any shared non-key column (F7 — no silent SQL suffixing);
    only the absent side's non-key fields become option ``None``.
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
    row-count check while silently corrupting the value (plan M40 trap #1). A ZERO-ROW ``block`` (a
    schema-only carrier for a one-sided join dest, or a partition an upstream cut emptied) has no row 0
    for the clamp-to-0 below to land on, and a gather from an empty block can only be misses — so it
    short-circuits to ``len(index)`` typed ``None`` rows (the null-fill a left/right/outer join needs)."""
    idx = np.asarray(index).astype(np.int64)
    if len(block) == 0:  # empty carrier -> all-None option column of block's type, length len(idx)
        return ak.Array(
            ak.contents.IndexedOptionArray(ak.index.Index64(np.full(len(idx), -1, dtype=np.int64)), block.layout)
        )
    valid = idx >= 0
    gathered = block[np.where(valid, idx, np.int64(0))]  # clamp misses to 0, then mask them out
    if bool(valid.all()):
        return gathered
    return ak.mask(gathered, valid)


def merge_records(left: ak.Array, right: ak.Array, *, on: Sequence[str]) -> ak.Array:
    """Flat field-union of two aligned taken blocks. The shared ``on`` key columns are **COALESCED** —
    the present (non-``None``) side wins, so a left/right/outer miss row keeps the REAL key value, never
    null (F3); the absent side's NON-key fields stay option ``None``. LEFT contributes its non-shared
    fields, RIGHT contributes each field LEFT lacks. A shared NON-key column is **rejected** (F7 — no
    silent SQL-style suffixing; rename it or add it to ``on``)."""
    on_set = {str(c) for c in on}
    lf, rf = ak.fields(left), ak.fields(right)
    shared_nonkey = sorted(f for f in rf if f in lf and f not in on_set)
    if shared_nonkey:
        raise ValueError(f"merge_records: shared non-key column(s) {shared_nonkey} — rename or add to `on`")
    fields: dict[str, ak.Array] = {}
    for f in lf:
        if f in on_set and f in rf:  # coalesce the shared key: the present (non-None) side wins
            # ...then STRIP the option: a row is emitted only when its present side carries the key, so the
            # coalesced key is never null — drop_none removes the never-taken option layer WITHOUT dropping
            # a row (length-preserving here), keeping the key non-option == op_form == numpy (the F4 class).
            fields[f] = ak.drop_none(ak.where(ak.is_none(left[f]), right[f], left[f]))
        else:
            fields[f] = left[f]
    for f in rf:
        if f not in lf:
            fields[f] = right[f]
    return ak.zip(fields, depth_limit=1)


def pack_key(rec: ak.Array, on: Sequence[str]) -> ak.Array:
    """Add a flat u64 ``__joinkey__`` column to ``rec``, big-endian bit-packed over ``on`` (first field
    most-significant). Integer bit-ops only — deterministic, process-independent, injective on in-range
    keys, and ``pack_key(0,…,0,e) == e``. NEVER Python ``hash()`` (plan M40 trap #6; §A.3.1)."""
    n = len(on)
    if n == 0:
        raise ValueError("pack_key needs at least one key field")
    if n * _PACK_STRIDE > 64:  # the packing itself cannot fit u64 (overflow→fixed-width bytes is Phase-2)
        raise ValueError(
            f"pack_key: {n} fields x {_PACK_STRIDE} bits exceeds u64 (overflow->bytes is Phase-2)"
        )
    limit = np.uint64(1) << np.uint64(_PACK_STRIDE)
    key: ak.Array | None = None
    for i, f in enumerate(on):
        col = ak.values_astype(rec[f], np.uint64)
        # F8: a field value >= 2**stride bleeds into the adjacent field and silently collides — raise
        # loudly instead (guard skipped on the typetracer, which carries no data to check).
        if ak.backend(col) != "typetracer" and bool(ak.any(col >= limit)):
            raise ValueError(
                f"pack_key: field {f!r} has a value >= 2**{_PACK_STRIDE} (overflow->bytes is Phase-2)"
            )
        shifted = col << np.uint64((n - 1 - i) * _PACK_STRIDE)
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
    on_set = {str(c) for c in on_from_params(params)}
    lf, rf = ak.fields(left), ak.fields(right)
    shared_nonkey = sorted(f for f in rf if f in lf and f not in on_set)
    if shared_nonkey:
        raise ValueError(f"op_form(join): shared non-key column(s) {shared_nonkey} — rename or add to `on`")
    left_opt = how in ("right", "outer")  # left's NON-key fields can be null on a right-only row
    right_opt = how in ("left", "outer")  # right's NON-key fields can be null on a left-only row
    fields: dict[str, ak.Array] = {}
    for f in lf:  # a coalesced shared key is NON-optional (always present); other left fields per `how`
        fields[f] = left[f] if (f in on_set and f in rf) else (_optional(left[f]) if left_opt else left[f])
    for f in rf:
        if f not in lf:
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
