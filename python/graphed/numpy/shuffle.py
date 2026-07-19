"""The numpy ``ShuffleBackend`` exchange primitives (plan M39 §3.0, §3.3, B-r5.3).

A SECOND exchange backend ships in M39 (not M40) so the generic engine's backend-agnosticism is
witnessed by EXECUTION, not merely the §A.4 import lint. numpy's rectilinear primitives are trivial
— ``partition``/``concat``/``slice_rows``/``estimated_bytes``/``to_wire``/``from_wire`` over
structured (record) arrays — but they obey the SAME pinned §4 routing rule the awkward backend does,
proven by the shared golden vectors (a numpy impl using any other hash fails them identically).

The route is ``int.from_bytes(sha256(key.to_bytes(8, "big") [+ salt]).digest()[:8], "big") % P`` on
the packed-u64 ``__joinkey__`` column — process-independent (no Python ``hash()``, which is
PYTHONHASHSEED-salted) so the same key lands in the same dest in every producer process (B2).
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from typing import Any

import numpy as np

#: the pinned §4 route hash (the golden vectors require exactly this; measured in exec-local's
#: benchmark against a non-crypto alternative).
PINNED_ROUTING_HASH = "sha256"


def route(key: int, parts: int, *, salt: int = 0) -> int:
    """The pinned §4/§3.0 route: sha256 of the 8-byte big-endian key (salt=0 == no bytes appended)."""
    key_bytes = int(key).to_bytes(8, "big") + (int(salt).to_bytes(8, "big") if salt else b"")
    return int.from_bytes(hashlib.sha256(key_bytes).digest()[:8], "big") % parts


def _dests(keys: np.ndarray, parts: int, salt: int) -> np.ndarray:
    """Per-row destination indices under the pinned route (one sha256 per row)."""
    return np.fromiter((route(int(k), parts, salt=salt) for k in keys), dtype=np.intp, count=len(keys))


def partition(
    block: np.ndarray,
    key_field: str,
    parts: int,
    *,
    salt: int = 0,
    boundaries: object = None,
) -> tuple[np.ndarray, ...]:
    """Route each record to one of ``parts`` sub-blocks by the pinned hash of ``block[key_field]``.
    Row-conserving and order-preserving within each dest; deterministic (§4/B2)."""
    dest = _dests(np.asarray(block[key_field]).astype(np.uint64), parts, salt)
    return tuple(block[dest == d] for d in range(parts))


def concat(blocks: Sequence[np.ndarray]) -> np.ndarray:
    """Vertically concatenate record blocks in order (the ascending-src_pid merge). The engine only
    ever merges a dest's >=1 contributing blocks, so an empty list is a caller error (np raises).

    M40/E4: if any block is an option (masked) block, the per-field validity mask is preserved via
    ``np.ma.concatenate``; a pure-exchange (all-plain) merge stays the byte-identical M39 path."""
    if any(isinstance(b, np.ma.MaskedArray) for b in blocks):
        return np.ma.concatenate([np.ma.asanyarray(b) for b in blocks])
    return np.concatenate([np.asarray(b) for b in blocks])


def slice_rows(block: np.ndarray, start: int, stop: int) -> np.ndarray:
    """A contiguous half-open record slice (mask-preserving for M40 option blocks; M39-identical
    for plain blocks)."""
    if isinstance(block, np.ma.MaskedArray):
        return block[start:stop]
    return np.asarray(block)[start:stop]


def estimated_bytes(block_or_form: object) -> int:
    """Measured payload bytes: the record array's ``nbytes`` (scales with itemsize)."""
    return int(np.asarray(block_or_form).nbytes)


def to_wire(block: np.ndarray) -> bytes:
    """Deterministic serialization — content-addressing hashes these bytes.

    Plain (exchange) blocks use the M39 ``.npy`` path BYTE-IDENTICALLY. Option (masked) join blocks
    carry per-field validity, which ``.npy`` cannot represent, so they go over Arrow native validity
    (E4); :func:`from_wire` dispatches on the leading magic bytes (``\\x93NUMPY`` vs Arrow IPC)."""
    if isinstance(block, np.ma.MaskedArray):
        return _mask_to_wire(block)
    buf = io.BytesIO()
    np.save(buf, np.asarray(block), allow_pickle=False)
    return buf.getvalue()


def from_wire(data: bytes) -> np.ndarray:
    """Inverse of :func:`to_wire`: ``.npy`` bytes -> plain array, Arrow bytes -> option block."""
    if data[:6] == b"\x93NUMPY":
        return np.load(io.BytesIO(data), allow_pickle=False)
    return _mask_from_wire(data)


# ---- M40 join half: relational primitives over structured (record) blocks --------------------
# The engine composes these into the generic radix-hash join; they are pure (no ``graphed`` imports)
# so the §A.4-clean engine can call them through the ``JoinBackend`` protocol. Option (nullable) rows
# are carried as numpy MASKED arrays: a missing field reads ``mask=True`` (E4). ``take`` NEVER uses
# ``np.take`` — a ``-1`` index is a MISS (invalid row), not "gather the last row" (M40 trap #1).


def _key_array(block: np.ndarray, on: Sequence[str]) -> np.ndarray:
    """The 1-D comparable join key for ``argsort``/``searchsorted``. Single field (the pinned
    ``__joinkey__`` flow) is the plain column; multiple fields fall back to a lexicographic void
    view (numpy orders structured arrays by field order)."""
    data = np.ma.getdata(np.ma.asanyarray(block))
    names = [str(c) for c in on]
    if len(names) == 1:
        return np.asarray(data[names[0]])
    return np.ascontiguousarray(data[names])


def match_indices(
    build: np.ndarray, probe: np.ndarray, *, on: Sequence[str], how: str = "inner"
) -> tuple[np.ndarray, np.ndarray]:
    """Relationally match two co-partitioned blocks on ``on`` (argsort + searchsorted, NEVER
    ``hash()``) and return aligned ``(build_idx, probe_idx)`` WITH duplication: a probe row with k
    build matches yields k pairs (SQL/pandas semantics). ``how`` in {inner,left,right,outer}; an
    unmatched side is emitted with a ``-1`` sentinel (→ :func:`take` yields a null/option row)."""
    bkey = _key_array(build, on)
    pkey = _key_array(probe, on)
    order = np.argsort(bkey, kind="stable")  # build positions sorted by key (stable ⇒ deterministic)
    skey = bkey[order]
    lo = np.searchsorted(skey, pkey, side="left")
    hi = np.searchsorted(skey, pkey, side="right")
    counts = (hi - lo).astype(np.intp)  # build matches per probe row
    total = int(counts.sum())
    # vectorized cross product: for each probe p, pair every build row in order[lo[p]:hi[p]] with p.
    starts: np.ndarray = np.repeat(lo, counts)
    group_base: np.ndarray = np.repeat(np.cumsum(counts) - counts, counts)
    build_idx = order[starts + (np.arange(total, dtype=np.intp) - group_base)]
    probe_idx: np.ndarray = np.repeat(np.arange(len(pkey), dtype=np.intp), counts)
    parts_b, parts_p = [build_idx.astype(np.intp)], [probe_idx]
    if how in ("left", "outer"):  # build rows with no probe match -> (build_row, -1)
        spk = np.sort(pkey)
        no_match = np.searchsorted(spk, bkey, side="right") == np.searchsorted(spk, bkey, side="left")
        ub: np.ndarray = np.flatnonzero(no_match).astype(np.intp)
        parts_b.append(ub)
        parts_p.append(np.full(len(ub), -1, dtype=np.intp))
    if how in ("right", "outer"):  # probe rows with no build match -> (-1, probe_row)
        up: np.ndarray = np.flatnonzero(counts == 0).astype(np.intp)
        parts_b.append(np.full(len(up), -1, dtype=np.intp))
        parts_p.append(up)
    return np.concatenate(parts_b), np.concatenate(parts_p)


def _masked_gather(block: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Gather rows by ``index`` into a structured MASKED array; ``index < 0`` rows are invalid.
    Data at a miss is row 0's (clip-to-0), never the last row — so a masked value is not a real one."""
    src = np.ma.asanyarray(block)
    data = np.ma.getdata(src)
    idx = np.asarray(index)
    safe = np.where(idx < 0, 0, idx).astype(np.intp)
    gathered = data[safe]
    miss = idx < 0
    prior = np.ma.getmaskarray(src)  # structured bool (all-False for a plain input)
    out_mask: np.ndarray = np.zeros(len(gathered), dtype=np.ma.make_mask_descr(gathered.dtype))
    any_masked = False
    for name in gathered.dtype.names or ():
        m = miss | prior[name][safe]
        out_mask[name] = m
        any_masked = any_masked or bool(m.any())
    if not any_masked:  # all-valid gather of a plain block -> stay PLAIN (M39 .npy wire, no Arrow)
        return gathered
    return np.ma.MaskedArray(gathered, mask=out_mask)


def take(block: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Positional record gather; a ``-1`` entry is a null/option row (M40 trap #1), NOT ``np.take``."""
    return _masked_gather(block, index)


def merge_records(left: np.ndarray, right: np.ndarray, *, on: Sequence[str]) -> np.ndarray:
    """Flat relational record-merge of two row-aligned (taken) blocks: the union of both sides'
    fields with the shared ``on`` key kept ONCE (coalesced from whichever side is present, so an
    outer row's key survives). Per-field validity masks are preserved (E4)."""
    on_set = {str(c) for c in on}
    lm, rm = np.ma.asanyarray(left), np.ma.asanyarray(right)
    ldata, lmask = np.ma.getdata(lm), np.ma.getmaskarray(lm)
    rdata, rmask = np.ma.getdata(rm), np.ma.getmaskarray(rm)
    lnames = list(ldata.dtype.names or ())
    shared_nonkey = sorted(n for n in (rdata.dtype.names or ()) if n in lnames and n not in on_set)
    if shared_nonkey:  # F7: no silent SQL-style suffixing — rename or add to `on`
        raise ValueError(f"merge_records: shared non-key column(s) {shared_nonkey} — rename or add to `on`")
    right_extra = [n for n in (rdata.dtype.names or ()) if n not in on_set and n not in lnames]
    out_names = lnames + right_extra
    out_dt = np.dtype([(n, (ldata if n in lnames else rdata).dtype[n]) for n in out_names])
    n = len(ldata)
    data = np.zeros(n, dtype=out_dt)
    mask: np.ndarray = np.zeros(n, dtype=np.ma.make_mask_descr(out_dt))
    for name in lnames:
        if name in on_set:  # coalesce the shared key: take the present side, null only if both miss
            data[name] = np.where(~lmask[name], ldata[name], rdata[name])
            mask[name] = lmask[name] & rmask[name]
        else:
            data[name], mask[name] = ldata[name], lmask[name]
    for name in right_extra:
        data[name], mask[name] = rdata[name], rmask[name]
    if not any(bool(mask[name].any()) for name in out_names):  # no nulls (inner) -> PLAIN (.npy wire)
        return data
    return np.ma.MaskedArray(data, mask=mask)


def pack_key(record: object, on: Sequence[str]) -> np.ndarray:
    """Add a flat u64 ``__joinkey__`` column to a record from the ``on`` fields by big-endian integer
    bit-packing — ``run`` (first) most-significant … the last field least-significant. Deterministic,
    process-independent, injective on in-range keys; NEVER Python ``hash()`` (spec pin, trap #6).

    Returns a structured ndarray (all original columns + ``__joinkey__``) so the exchange/join path
    can route on it directly. ponytail: pinned 20-bit-per-field HEP layout (``run<<40|lumi<<20|event``
    for the 3-field key); wider keys / overflow→fixed-width-bytes is Phase-2."""
    cols = _as_columns(record)
    names = [str(c) for c in on]
    if len(names) * 20 > 64:  # the packing cannot fit u64 (overflow->fixed-width bytes is Phase-2)
        raise ValueError(f"pack_key: {len(names)} fields x 20 bits exceeds u64 (overflow->bytes is Phase-2)")
    for nm in names:  # F8: a field value >= 2**20 bleeds into the adjacent field and silently collides
        if np.any(np.asarray(cols[nm]).astype(np.uint64) >= np.uint64(1 << 20)):
            raise ValueError(
                f"pack_key: field {nm!r} has a value >= 2**20 (overflow->fixed-width bytes is Phase-2)"
            )
    key = _pack_key_column([cols[nm] for nm in names])
    cols["__joinkey__"] = key
    out_dt = np.dtype([(nm, np.asarray(v).dtype) for nm, v in cols.items()])
    out = np.zeros(len(key), dtype=out_dt)
    for nm, v in cols.items():
        out[nm] = np.asarray(v)
    return out


def _pack_key_column(columns: Sequence[np.ndarray]) -> np.ndarray:
    """``(c0 << 20*(n-1)) | ... | c_{n-1}`` as uint64 — the pinned big-endian pack."""
    n = len(columns)
    key: np.ndarray = np.zeros(len(columns[0]), dtype=np.uint64)
    for j, c in enumerate(columns):
        key = key | (np.asarray(c).astype(np.uint64) << np.uint64(20 * (n - 1 - j)))
    return key


def _as_columns(record: object) -> dict[str, np.ndarray]:
    """A mutable name->column dict from a record (dict of columns or a structured ndarray)."""
    if isinstance(record, dict):
        return {str(k): np.asarray(v) for k, v in record.items()}
    arr = np.asarray(record)
    return {name: np.asarray(arr[name]) for name in (arr.dtype.names or ())}


# ---- E4 Arrow-native-validity wire codec (option blocks only) --------------------------------
def _pa() -> Any:
    try:
        import pyarrow as pa  # noqa: PLC0415  (lazy: pyarrow is the optional [parquet] extra)
    except ImportError as exc:  # pragma: no cover - exercised where pyarrow is absent
        raise ImportError(
            "option (nullable join) blocks serialize over Arrow — install: pip install 'graphed[parquet]'"
        ) from exc
    return pa


def _mask_to_wire(block: np.ndarray) -> bytes:
    """Serialize a structured masked block to Arrow IPC, carrying each field's mask as native
    validity (``pa.array(values, mask=...)``) — the E4 pin."""
    pa = _pa()
    src = np.ma.asanyarray(block)
    data, mask = np.ma.getdata(src), np.ma.getmaskarray(src)
    cols = {
        name: pa.array(np.asarray(data[name]), mask=np.asarray(mask[name]))
        for name in (data.dtype.names or ())
    }
    table = pa.table(cols)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return bytes(sink.getvalue().to_pybytes())


def _mask_from_wire(data: bytes) -> np.ndarray:
    """Inverse of :func:`_mask_to_wire`: rebuild the structured masked block from Arrow validity."""
    pa = _pa()
    table = pa.ipc.open_stream(pa.BufferReader(data)).read_all()
    names = table.column_names
    values: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for name in names:
        arr = table.column(name).combine_chunks()
        masks[name] = np.asarray(arr.is_null())
        # drop the validity bitmap so to_numpy never trips on a null; masked values are placeholders.
        bare = pa.Array.from_buffers(arr.type, len(arr), [None, arr.buffers()[1]], offset=arr.offset)
        values[name] = np.asarray(bare.to_numpy(zero_copy_only=False))
    out_dt = np.dtype([(name, values[name].dtype) for name in names])
    n = len(next(iter(values.values())))
    out_data = np.zeros(n, dtype=out_dt)
    out_mask: np.ndarray = np.zeros(n, dtype=np.ma.make_mask_descr(out_dt))
    for name in names:
        out_data[name], out_mask[name] = values[name], masks[name]
    return np.ma.MaskedArray(out_data, mask=out_mask)
