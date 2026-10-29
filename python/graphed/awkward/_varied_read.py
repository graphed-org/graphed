"""The §6.4c/e read-back: ``read_varied`` rebuilds every universe of a variation-aware skim."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import awkward as ak
import numpy as np

from graphed import parquet as gpq
from graphed.errors import GraphedError

from ._varied_write import (
    _BASE_FIELD,
    _MANIFEST_KEY,
    _RECORD_TYPE,
    _VARY_PREFIX,
    _filled,
    _flat_field,
    _uint_view,
)


def _label_order(fields: Sequence[str]) -> list[str]:
    """Universes in §2.4 order (nominal first).

    Recovered from the on-disk vary-column order: value deltas precede masks, so a first-occurrence
    walk yields the variations in registration order; the reserved `nominal` label (present only in
    mask columns) is forced to the front.
    """
    seen: list[str] = []
    for name in fields:
        if name.startswith(_VARY_PREFIX):
            label = name[len(_VARY_PREFIX) :].split("__", 1)[0]
            if label not in seen:
                seen.append(label)
    return ["nominal", *(label for label in seen if label != "nominal")]


def _dotted_for_flat(record: ak.Array, flat: str) -> str:
    """Return the record's leaf path whose `_`-flattening equals `flat`.

    It is unique — §6.4b refuses collisions.
    """
    for path in record.layout.form.columns():
        if _flat_field(path) == flat:
            return str(path)
    raise GraphedError(f"no stored leaf flattens to the manifest field {flat!r}")


def _get_leaf(record: ak.Array, path: str) -> ak.Array:
    leaf = record
    for part in path.split("."):
        leaf = leaf[part]
    return leaf


def _apply_xor(nominal_value: ak.Array, delta: ak.Array, jagged: bool) -> ak.Array:
    """§6.4c inverse: the label value is `nominal XOR delta` on the shared uint view.

    It is reinterpreted back to nominal's dtype (XOR is its own inverse, so this is exact), `None`
    where the delta is.
    """
    nom, _ = _filled(nominal_value, jagged)
    del_, mask = _filled(delta, jagged)
    restored = (_uint_view(nom) ^ _uint_view(del_)).view(nom.dtype)
    out = restored if mask is None else np.ma.array(restored, mask=mask)
    return ak.unflatten(out, np.asarray(ak.num(nominal_value))) if jagged else ak.Array(out)


def _decode_level0(packed: ak.Array) -> np.ndarray:
    """Return a flat level-0 validity mask: one packbits byte per row → one bool per row."""
    bits = [
        bool(np.unpackbits(np.asarray(row, dtype=np.uint8), count=1)[0])
        for row in ak.to_list(packed)
    ]
    out: np.ndarray = np.array(bits, dtype=bool)
    return out


def _decode_jagged(packed: ak.Array, counts: np.ndarray) -> ak.Array:
    """Return a level->=1 validity mask: each row's packbits bytes → its `count` object bits."""
    return ak.Array(
        [
            np.unpackbits(np.asarray(byte_row, dtype=np.uint8), count=int(n)).astype(bool).tolist()
            for byte_row, n in zip(ak.to_list(packed), counts, strict=True)
        ]
    )


def _entry_key(entry: int | list[Any]) -> tuple[str | None, int]:
    """Return a manifest `entry` as `(field_path|None, depth)`.

    The entry is a bare `int` depth, or `[flat_field, depth]`.
    """
    return (None, entry) if isinstance(entry, int) else (entry[0], entry[1])


def _reconstruct_universe(
    base: ak.Array, full: ak.Array, manifest: Mapping[str, Any], label: str
) -> ak.Array:
    """§6.4c: universe `label`'s post-selection record.

    Nominal `base` (on the superset rows) with the label's XOR value deltas applied, then its
    stored level-0 row mask AND every level->=1 object mask (each stored at superset granularity,
    restricted to the kept rows).
    """
    columns = manifest[label]
    rec = base
    for col, info in columns.items():
        if info["representation"] in ("xor", "base") and col.startswith(_VARY_PREFIX):
            path = _dotted_for_flat(rec, info["field"])
            if info["representation"] == "base":
                value = full[col]
            else:
                nominal_value = _get_leaf(rec, path)
                value = _apply_xor(nominal_value, full[col], nominal_value.ndim > 1)
            rec = ak.with_field(rec, value, where=path.split("."))

    packed = {  # entry -> the label's mask column, still at superset granularity
        _entry_key(info["entry"]): (info["representation"], full[col])
        for col, info in columns.items()
        if info["representation"] in ("packbits", "option")
    }
    # level 0 selects rows; a higher level's mask is decoded over the PRE-selection field counts,
    # then restricted to the kept rows — `X[m0][m_k[m0]]` == the analyst's `X[event][object]`
    # (§6.4c).
    higher: list[tuple[str | None, ak.Array]] = []
    m0: Any = None
    for entry in manifest["levels"]:
        key = _entry_key(entry)
        field_path, depth = key
        representation, column = packed[key]
        if representation == "option":  # indexes as ak does: a None entry gives a None
            mask: Any = column
        elif depth == 0:
            mask = _decode_level0(column)
        else:
            target = rec if field_path is None else rec[field_path]
            mask = _decode_jagged(column, np.asarray(ak.num(target)))
        if depth == 0:
            m0 = mask
        else:
            higher.append((field_path, mask))

    if m0 is not None:
        rec = rec[m0]
    for field_path, mask in higher:
        inner = mask if m0 is None else mask[m0]
        if field_path is None:
            rec = rec[inner]
        else:
            rec = ak.with_field(rec, rec[field_path][inner], where=field_path.split("."))
    # the XOR rebuild carries no parameters: impose the universe's record type (nominal's, unless
    # stored)
    stored = columns.get(f"{_RECORD_TYPE}{label}") or manifest["nominal"].get(
        f"{_RECORD_TYPE}nominal"
    )
    # an option level-0 mask gave None rows
    option = isinstance(rec.type.content, ak.types.OptionType)
    target = rec.type.content.content if option else rec.type.content
    if stored is not None:
        want = ak.forms.from_dict(stored["type"]).type
    elif target.is_equal_to(base.type.content):
        want = base.type.content
    else:
        return rec
    return ak.enforce_type(rec, ak.types.OptionType(want) if option else want)


def read_varied(path: str) -> dict[str, ak.Array]:
    """Reconstruct every universe of a §6.4 variation-aware skim.

    `{label: ak.Array}` in §2.4 order (nominal first). Each value is universe `label`'s
    post-selection record — its rows passing the stored level-0 mask and, at every supplied level,
    its objects passing the stored mask — bit-exact vs the in-memory varied run. The symmetric
    partner of `to_parquet(..., select=…)`.
    """
    meta = gpq._pq().ParquetFile(path).metadata.metadata or {}
    blob = meta.get(_MANIFEST_KEY)
    if blob is None:
        raise GraphedError(
            f"{path!r} carries no {_MANIFEST_KEY!r} manifest — not a variation-aware skim"
        )
    manifest = json.loads(blob)

    full = ak.from_parquet(path)
    if _BASE_FIELD in full.fields:
        base = full[_BASE_FIELD]  # a bare jagged-collection skim (the record IS the collection)
    else:
        base_fields = [f for f in full.fields if not f.startswith(_VARY_PREFIX)]
        base = ak.zip({f: full[f] for f in base_fields}, depth_limit=1)
    return {
        label: _reconstruct_universe(base, full, manifest, label)
        for label in _label_order(full.fields)
    }
