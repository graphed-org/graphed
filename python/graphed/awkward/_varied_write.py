"""The §6.4 variation-aware write's record-time half: entry checks, encodings, the manifest.

The awkward idiom's `to_parquet(record, select=…)` materializes the SUPERSET of rows passing any
universe's level-0 selection and augments the nominal record with same-dtype XOR value deltas +
packbits validity masks, so every universe's post-selection values and row set reconstruct
bit-for-bit via `read_varied`. The write task and the §7.2 node-id unpack live in `io`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import awkward as ak
import numpy as np

import graphed.core
from graphed import Array, CompiledGraph, Session, context_of
from graphed import parquet as gpq
from graphed import write as gw
from graphed.errors import GraphedError
from graphed.systematics.lineage import is_ancestor, links_below
from graphed.varied import Varied, member_of, members_of, most_derived_context

from .backend import AwkwardForm

if TYPE_CHECKING:
    from .io import _MaskSpec, _ValueSpec


_VARY_PREFIX = "__vary_"
#: a bare jagged-collection record (`to_parquet(events.Jet, …)`) is a LIST of records, not a
#: row-level record, so it cannot carry sibling columns; it is wrapped under this reserved field.
_BASE_FIELD = "__base__"
#: §6.4e: the manifest lives in parquet key-value file metadata (greenfield — graphed writes none).
_MANIFEST_KEY = b"graphed.variations"
#: prefix of a manifest entry that names no column (outside `_VARY_PREFIX`, so no leaf name
#: collides): an `xor` entry whose `field` matches no stored leaf, so a reader that predates `base`
#: refuses the file.
_REQUIRES_BASE = "requires_base__"
#: prefix of the entry carrying a universe's record form when the reader cannot rebuild its type
#: from the columns (a parameter such as `__doc__`, a top record's own parameters); it names no
#: column either.
_RECORD_TYPE = "type__"

_SelectKey = Any  # `int` (bare depth) or `tuple[str, int]` (field-scoped, depth >= 1)


def _flat_field(path: str) -> str:
    """§6.4b: the on-disk field spelling.

    A nested path flattened per level with `_` (`Jet.pt` -> `Jet_pt`; a bare-field skim's `pt` ->
    `pt`).
    """
    return path.replace(".", "_")


def _entry_depth(key: _SelectKey) -> int:
    return key if isinstance(key, int) else key[1]


def _entry_flat(key: _SelectKey) -> str:
    """Return the mask column's `entry` token (`…__mask__0`, `…__mask__1`, `…__mask__Jet_1`)."""
    if isinstance(key, int):
        return str(key)
    field_name, depth = key
    return f"{_flat_field(field_name)}_{depth}"


def _entry_manifest(key: _SelectKey) -> int | list[Any]:
    """Return the manifest `entry`/levels element: a bare depth `int`, or `[flat_field, depth]`."""
    return key if isinstance(key, int) else [_flat_field(key[0]), key[1]]


def _level_sort_key(key: _SelectKey) -> tuple[int, str]:
    """§6.4e's explicit total order over the heterogeneous levels list.

    `(depth, flat_field or "")` — bare-depth before field-scoped of the same depth (`sorted()`
    cannot order the mixed list).
    """
    return (_entry_depth(key), "" if isinstance(key, int) else _flat_field(key[0]))


def _normalize_select(select: Any) -> dict[_SelectKey, Any]:
    """§6.4a's key space.

    A single `Varied`/Array row mask (⇔ `{0: mask}`), or a mapping keyed by `int` (bare depth) or
    `(field_name, depth>=1)` (field-scoped).
    """
    if isinstance(select, Mapping):
        out: dict[_SelectKey, Any] = {}
        for key, mask in select.items():
            field_scoped = (
                isinstance(key, tuple)
                and len(key) == 2
                and isinstance(key[0], str)
                and isinstance(key[1], int)
                and key[1] >= 1
            )
            if not (isinstance(key, int) or field_scoped):
                raise GraphedError(
                    "select= keys are a bare depth `int` or a `(field_name, depth>=1)` tuple, "
                    f"not {key!r}"
                )
            out[key] = mask
        return out
    return {0: select}


def _is_option(session: Session, array: Any) -> bool:
    """Whether a mask's type holds an option at any depth.

    For example `?bool`, `var * ?bool`, `?var * bool`.
    """
    t = _tt(session, array).type.content
    while t is not None:
        if isinstance(t, ak.types.OptionType):
            return True
        t = getattr(t, "content", None)
    return False


def _tt(session: Session, array: Any) -> ak.Array:
    """Return the typetracer of an array's form.

    The awkward-idiom depth/leaf oracle (§6.4a's `.tt.ndim`).
    """
    form = session.form(array)
    assert isinstance(form, AwkwardForm)
    return form.tt


def _type_of(session: Session, array: Any) -> ak.types.Type:
    return _tt(session, array).layout.form.type


def _record_context(value: Any) -> Any:
    """Return the row-space handle of a write operand.

    A plain value's own handle, a container's most-derived member handle (§2.3e).
    """
    if isinstance(value, Varied):
        return most_derived_context(*members_of(value).values())
    return context_of(value)


def _lineage_refusal(record_ctx: Any, select_ctx: Any) -> None:
    """Refuse a §6.4a(2a) lineage or §6.4b row-space mismatch, decided record-time.

    It is decided by CONTEXT-HANDLE reachability over the level-0 mask. Vary IDENTITY links do not
    move the row space (§6.1d kind (2)); a mask or project link does. Direction picks the message:
    a record selected BELOW its mask is §6.4b (row-space, NEVER offsets); a mask more derived than
    the record is §6.4a(2a).
    """
    if record_ctx is None:
        return  # absent-operand (i): a loose record carries no handle, so (2a) is skipped
    if select_ctx is None:
        raise GraphedError(
            f"the record read through {record_ctx!r} was given a select= mask that carries no "
            "context handle, so its lineage cannot be checked against the record (§6.4a(2a·ii))"
        )
    if record_ctx is select_ctx:
        return
    if is_ancestor(select_ctx, record_ctx):
        if all(kind == "vary" for kind, _ in links_below(record_ctx, select_ctx)):
            return
        raise GraphedError(
            f"the record read through {record_ctx!r} is selected into a narrower ROW SPACE than "
            f"its select= mask ({select_ctx!r}) defines: a stored field would live below the "
            "superset the mask spans, and a mask has no inverse to lift it back — read the "
            "record at the mask's row space (§6.4b)"
        )
    if is_ancestor(record_ctx, select_ctx):
        if all(kind == "vary" for kind, _ in links_below(select_ctx, record_ctx)):
            return
        raise GraphedError(
            f"the record read through {record_ctx!r} is not the context its select= mask "
            f"({select_ctx!r}) derives from: the mask is more derived than the record's rows "
            "(§6.4a(2a))"
        )
    raise GraphedError(
        f"the record ({record_ctx!r}) and its select= mask ({select_ctx!r}) are on divergent "
        "branches; the mask does not select the record's rows (§6.4a(2a))"
    )


def _resolve_field(
    nodes: list[dict[str, Any]], session: Session, node_id: int, name: str
) -> int | str:
    """Return the canonical value node for `field(node, name)`.

    The field is followed back to the node that produced it through every op that passes fields
    through unchanged (zip, with_field, with_name, with_parameter, without_field, a field
    selection, a field projection), and any other op ends the walk at its own projection. The M4
    optimizer pushes no projection through these, so without the walk an untouched leaf of a varied
    record would look varied (§6.4b).
    """
    if node_id >= len(nodes):  # newer than the snapshot: a projection we recorded, never a zip
        return int(getattr(Array(session, node_id), name).node_id)
    nd = nodes[node_id]
    op = nd["name"] if nd["kind"] == "op" else None
    if op == "ak.zip":
        fields = str(nd["params"]["fields"]).split(",")
        return int(nd["inputs"][fields.index(name)])
    if op == "field":
        record, field = int(nd["inputs"][0]), str(nd["params"]["field"])
        if field in _tt(session, Array(session, record)).fields:  # else a behavior property
            inner = _resolve_field(nodes, session, record, field)
            if isinstance(inner, int) and inner != node_id:  # a source's field is its own fixpoint
                return _resolve_field(nodes, session, inner, name)
    if op in ("ak.with_name", "ak.with_parameter", "ak.without_field", "fields"):
        return _resolve_field(nodes, session, int(nd["inputs"][0]), name)
    if op == "ak.with_field":
        if str(nd["params"]["field"]) == name:
            if len(nd["inputs"]) > 1:
                return int(nd["inputs"][1])
            # a scalar field has no value node: its value and dtype are its identity
            return json.dumps(
                {k: v for k, v in nd["params"].items() if k != "field"}, sort_keys=True
            )
        return _resolve_field(nodes, session, int(nd["inputs"][0]), name)
    return int(getattr(Array(session, node_id), name).node_id)


def _canonical_leaf(
    nodes: list[dict[str, Any]], session: Session, member: Any, path: str
) -> int | str:
    key: int | str = member.node_id
    for part in path.split("."):
        # a scalar key is always a path's last
        key = _resolve_field(nodes, session, int(key), part)
    return key


def _varied_leaves(session: Session, record: Any, labels: tuple[str, ...]) -> list[str]:
    """Return the leaf paths whose canonical value node DIFFERS across labels.

    Those are structurally varied, §6.4b; an all-labels-equal leaf (unvaried, or coupled only
    through a zip) contributes no delta column.
    """
    nodes = session._nodes()
    leaves = _tt(session, member_of(record, "nominal")).layout.form.columns()
    varied = []
    for path in leaves:
        canon = {
            _canonical_leaf(nodes, session, member_of(record, label), path) for label in labels
        }
        if len(canon) > 1:
            varied.append(path)
    return varied


def _check_collision(varied: Sequence[str]) -> None:
    """§6.4b: refuse two varied leaves that flatten to one on-disk name, naming both.

    For example a nested `Jet.pt` beside a flat `Jet_pt`, both -> `__vary_L__Jet_pt`.
    """
    by_flat: dict[str, list[str]] = {}
    for path in varied:
        by_flat.setdefault(_flat_field(path), []).append(path)
    for flat, sources in by_flat.items():
        if len(sources) > 1:
            raise GraphedError(
                f"stored fields {sources} both flatten to the on-disk name {_VARY_PREFIX}<label>__"
                f"{flat}; a variation-aware skim cannot tell them apart — rename one (§6.4b)"
            )


def _check_bare_key(session: Session, record: Any, keys: Sequence[_SelectKey]) -> None:
    """§6.4a: a bare depth-`k>=1` key is legal only when the record is ITSELF jagged at depth `k`.

    A flat record with two or more independently jagged fields refuses it, naming the ambiguity.
    """
    nominal = member_of(record, "nominal")
    record_ndim = _tt(session, nominal).ndim
    for key in keys:
        if isinstance(key, int) and key >= 1 and record_ndim <= key:
            form = _tt(session, nominal)
            jagged = [f for f in form.fields if form[f].ndim > key]
            raise GraphedError(
                f"the bare depth-{key} select= key needs the record to be jagged at depth {key}; "
                f"this record is a flat record whose jagged fields {jagged} each have their own "
                f"offsets — name the field with a ({jagged[0]!r}, {key}) key instead (§6.4a)"
            )


def _check_depth(session: Session, entries: Mapping[_SelectKey, Any]) -> None:
    """§6.4a(2c): a mask supplied at level `k` must have depth `k` over the record's structure.

    Typetracer ndim `k + 1` (flat = ndim 1 at level 0), at EVERY supplied level, record-time.
    """
    for key, mask in entries.items():
        ndim = _tt(session, member_of(mask, "nominal")).ndim
        depth = _entry_depth(key)
        if ndim != depth + 1:
            raise GraphedError(
                f"the select= mask at level {key!r} has depth {ndim - 1} (ndim {ndim}); level "
                f"{depth} requires a depth-{depth} mask (ndim {depth + 1}) over the record "
                "(§6.4a(2c))"
            )


def _uint_view(a: np.ndarray) -> np.ndarray:
    return a.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize])


def _counts(values: ak.Array) -> np.ndarray:
    """Per-row list lengths, -1 for a `None` list."""
    counts: np.ndarray = np.asarray(ak.fill_none(ak.num(values), -1))
    return counts


def _xor_representable(leaf_type: ak.types.Type) -> bool:
    """Return the leaf types `_encode_delta` round-trips exactly.

    `T`, `?T`, `var * T`, `var * ?T` over a primitive, none of them a string-like `__array__`, nor
    `datetime64[s]`, which parquet stores as `[ms]` so nominal's bits on disk differ from the ones
    the delta was taken against.
    """
    if isinstance(leaf_type, ak.types.ListType):
        leaf_type = leaf_type.content
    if isinstance(leaf_type, ak.types.OptionType):
        leaf_type = leaf_type.content
    return (
        isinstance(leaf_type, ak.types.NumpyType)
        and leaf_type.parameter("__array__") is None
        and leaf_type.primitive != "datetime64[s]"
    )


def _filled(values: ak.Array, jagged: bool) -> tuple[np.ndarray, Any]:
    """Return a leaf's flat values with `None` read as 0 (so both sides XOR the same bits).

    And its `None` mask (numpy's `nomask` when it holds none, which keeps a no-`None` delta an
    `UnmaskedArray` on disk).
    """
    flat = ak.to_numpy(ak.flatten(values) if jagged else values)
    if isinstance(flat, np.ma.MaskedArray):
        return flat.filled(0), np.ma.getmask(flat)
    return flat, None


def _encode_delta(nominal: ak.Array, label: ak.Array, jagged: bool) -> ak.Array:
    """§6.4c: the same-dtype XOR bit-delta vs nominal.

    Zero wherever a label equals nominal, masked where the label is `None`.
    """
    nominal_flat, _ = _filled(nominal, jagged)
    label_flat, mask = _filled(label, jagged)
    delta = _uint_view(label_flat) ^ _uint_view(nominal_flat)
    out = delta if mask is None else np.ma.array(delta, mask=mask)
    return ak.unflatten(out, ak.num(nominal)) if jagged else ak.Array(out)


def _encode_mask(mask: ak.Array) -> ak.Array:
    """§6.4c: a per-row `packbits` validity mask stored as `var * uint8`.

    Row-aligned; a flat level-0 mask packs one bit per row, a jagged level-k>=1 mask packs its
    row's objects.
    """
    return ak.Array(
        [
            list(
                np.packbits(np.asarray(row, dtype=bool) if isinstance(row, list) else [bool(row)])
            )
            for row in ak.to_list(mask)
        ]
    )


def _write_augmented(payload: ak.Array, path: str, manifest: Mapping[str, Any] | None) -> None:
    """Write the augmented record, merging the §6.4e manifest into the parquet file KV metadata.

    `ak.to_parquet` writes awkward's own form KV (needed so `read_varied`/`ak.from_parquet` still
    reconstruct the record); we then re-stamp the file-level metadata through the PUBLIC arrow
    route (`Table.replace_schema_metadata`, preserving awkward's entries and every field's
    metadata), never `awkward._connect`. `manifest is None` is the unvaried byte-golden path —
    untouched (§6.4g).
    """
    ak.to_parquet(payload, path)
    if manifest is None:
        return
    pq = gpq._pq()
    fs, where = gw.part_fs(path)
    table = pq.read_table(where, filesystem=fs)
    meta = dict(table.schema.metadata or {})
    meta[_MANIFEST_KEY] = json.dumps(manifest, sort_keys=True).encode()
    pq.write_table(table.replace_schema_metadata(meta), where, filesystem=fs)


def _build_manifest(
    labels: Sequence[str],
    value_specs: Sequence[_ValueSpec],
    mask_specs: Sequence[_MaskSpec],
    record_types: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """§6.4e: the driver-derived reconstruction manifest.

    Every label's stored columns (value columns keyed `xor` or `base`/field, masks keyed
    `packbits`/entry) plus the reserved ordered `levels` list. Mirrors exactly the columns
    `_VariedWritePart.__call__` writes, plus one `_REQUIRES_BASE` marker entry per `base` column
    and one `_RECORD_TYPE` entry per `record_types` label, neither naming a column, so it is
    data-independent and deterministic (serialized `json.dumps(sort_keys=True)`; `levels` order is
    the explicit key `sorted()` cannot give).
    """
    manifest: dict[str, Any] = {label: {} for label in labels}
    for spec in value_specs:
        for label in labels:
            if label == "nominal":
                continue
            manifest[label][f"{_VARY_PREFIX}{label}__{spec.flat}"] = {
                "representation": "base" if label in spec.verbatim else "xor",
                "field": spec.flat,
                "entry": None,
            }
            if label in spec.verbatim:
                manifest[label][f"{_REQUIRES_BASE}{label}__{spec.flat}"] = {
                    "representation": "xor",
                    "field": (
                        f"graphed>=0.0.7 needed: {label} stores {spec.flat} as representation base"
                    ),
                    "entry": None,
                }
    for ms in mask_specs:
        for label in labels:
            manifest[label][f"{_VARY_PREFIX}{label}__mask__{ms.entry_flat}"] = {
                "representation": "option" if label in ms.option else "packbits",
                "field": None,
                "entry": ms.manifest,
            }
    for label, record_type in record_types.items():
        manifest[label][f"{_RECORD_TYPE}{label}"] = {
            "representation": "base",
            "field": None,
            "entry": None,
            "type": record_type,
        }
    manifest["levels"] = [ms.manifest for ms in mask_specs]
    return manifest


def _refuse_optimizer_merge(
    session: Session, outputs: Sequence[Any], compiled: CompiledGraph
) -> None:
    """§6.4f binds §7.2's optimizer-merge shortfall onto the write path (record-time, at the call).

    The varied unpack resolves BY NODE ID, so the compiled output count must match the DISTINCT
    marked ids; a `w * 1.0` label merges two distinct record ids into one compiled output, and the
    node-id table can then no longer tell the labels apart.
    """
    marked = len(dict.fromkeys(out.node_id for out in outputs))
    compiled_outputs = len(graphed.core.GraphStore.deserialize(compiled.ir).outputs())
    if compiled_outputs >= marked:
        return
    raise GraphedError(
        f"the optimizer merged outputs that record as distinct nodes ({marked} marked, "
        f"{compiled_outputs} compiled), so this varied write's labels can no longer be told "
        "apart. "
        'Spell a label whose value equals another with the SAME expression (variations={"1": w}, '
        "not w * 1.0), which routes it through the supported record-time dedup instead "
        "(§6.4f/§7.2)"
    )
