"""Partitioned parquet I/O for the awkward backend (M15.2, dask-awkward parity plan).

Specializes the backend-agnostic `graphed.parquet` base: the awkward pieces are exactly two —
the FORM comes from the arrow schema alone (`ak.from_arrow_schema`; no event data is read at
construction) and the per-partition codec is `ak.from_parquet`/`ak.to_parquet`.

`to_parquet` follows the R15.4/R15.5 contract proven by the uproot integration: compute-disabled
returns a task graph of write tasks; each task evaluates the array's graph through the COMPILED
IR (R7.8 — compiled once at the driver, never re-recorded per partition), reads only the
PROJECTED columns (R15.3 — the read list is wired from the projection), derives its own output
part index from its partition (R15.9), and writes one parquet part.
"""

from __future__ import annotations

import functools
import operator
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import awkward as ak
import numpy as np

import graphed.core
from graphed import Array, Backend, CompiledGraph, Session, compile_ir, context_of, evaluate_ir
from graphed import parquet as gpq
from graphed import write as gw
from graphed.core import Partition
from graphed.core.execution import Plan, SequentialRunner, WorkerResources
from graphed.errors import GraphedError
from graphed.varied import Varied, member_of, most_derived_context, union_labels
from graphed.write import PartitionedSource

from .backend import AwkwardBackend, AwkwardForm
from .projection import project_buffers


def _schema_form(paths: Sequence[str], columns: Sequence[str] | None) -> AwkwardForm:
    """The dataset's form from the arrow SCHEMA alone (first file authoritative; nothing decoded)."""
    form = ak.from_arrow_schema(gpq.schema_of(paths))
    if columns:
        form = form.select_columns(list(columns))
    tt = ak.Array(form.length_zero_array(highlevel=False).to_typetracer(forget_length=True))
    return AwkwardForm(tt)


@dataclass(frozen=True)
class _DatasetLoader:
    """Lazy whole-dataset loader for the reference ``materialize`` — AND a
    ``graphed.write.PartitionedSource``, so the generic writer reads it partition by partition
    (the whole-dataset path is only ever the reference ``materialize``)."""

    paths: tuple[str, ...]
    columns: tuple[str, ...] | None

    def __call__(self) -> ak.Array:
        return ak.from_parquet(list(self.paths), columns=list(self.columns) if self.columns else None)

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return gpq.make_partitions(self.paths, steps_per_file=steps_per_file, open_files=False)

    def read_partition(self, partition: Partition, columns: Sequence[str] | None, resources: Any) -> ak.Array:
        return read_parquet_partition(partition, columns if columns else (self.columns or None))


def from_parquet(
    session: Session,
    name: str,
    path: str | Sequence[str],
    *,
    columns: Sequence[str] | None = None,
    steps_per_file: int = 1,
    open_files: bool = True,
) -> Any:
    """A deferred awkward array over a parquet dataset (file / directory / glob / list).

    The form is built from METADATA alone; no event data is read here. ``steps_per_file`` and
    ``open_files`` shape the dataset's default partitioning (``partitions_of``); with
    ``open_files=False`` no file is opened at all (blind partitions, R7.9)."""
    paths = gpq.discover(path)
    form = _schema_form(paths, columns)
    if steps_per_file < 1:
        raise ValueError(f"steps_per_file must be >= 1, got {steps_per_file}")
    if not open_files:
        gpq.make_partitions(paths, steps_per_file=steps_per_file, open_files=False)  # validated blind
    loader = _DatasetLoader(paths, tuple(columns) if columns else None)
    return gpq.deferred_source(session, name, paths=paths, form=form, loader=loader)


def read_parquet_partition(partition: Partition, columns: Sequence[str] | None = None) -> ak.Array:
    """Read one partition (resolving blind ones at read time), restricted to ``columns``."""
    part = gpq.resolve_partition(partition)
    arr = ak.from_parquet(part.uri, columns=list(columns) if columns else None)
    return arr[part.entry_start : part.entry_stop]


# ---- deferred writing ------------------------------------------------------------------------
@dataclass(frozen=True)
class _WritePart:
    """The picklable per-partition write task: compiled IR in, one parquet part out."""

    compiled: CompiledGraph
    source_name: str
    columns: tuple[str, ...]
    destination: str
    prefix: str
    steps_per_file: int
    bases: tuple[tuple[Any, int], ...]
    column: str = "data"
    reader: Any = None  # a graphed.write.PartitionedSource (picklable)
    behavior: Any = None  # a behavior dict or an importable "module:attr" reference
    memory_data: ak.Array | None = None  # in-memory source payload (bounded by the dataset)
    memory_rows: int = 0

    def __call__(self, partition: Partition, resources: WorkerResources) -> list[str]:
        if self.reader is not None:
            chunk = self.reader.read_partition(partition, self.columns or None, resources)
            index = gw.blind_part_index(partition, dict(self.bases))
        elif self.memory_data is not None:
            chunk = self.memory_data[partition.entry_start : partition.entry_stop]
            index = _memory_step(partition, self.memory_rows, self.steps_per_file)
        else:  # pragma: no cover - every source is a protocol reader or in-memory
            raise TypeError("write task has neither a partition reader nor in-memory data")
        backend = AwkwardBackend(behavior=_resolve_behavior(self.behavior))
        (out,) = evaluate_ir(self.compiled, cast("Backend", backend), {self.source_name: chunk})
        result = ak.Array(out)
        payload = result if result.fields else ak.Array({self.column: result})
        os.makedirs(self.destination, exist_ok=True)
        path = gpq.part_path(self.destination, index, prefix=self.prefix)
        ak.to_parquet(payload, path)
        return [path]


def _syntactic_fields(array: Any, source_node_id: int) -> set[str] | None:
    """The top-level source fields the recorded graph ACCESSES (a session walk). ``None`` means
    the whole source is consumed (a bare source, or a non-field op applied to it directly)."""
    found: set[str] = set()
    whole = [False]
    sentinel = object()

    def on_source(nid: int) -> object:
        return (sentinel, nid)

    def touches_source(ins: list[object]) -> bool:
        return any(
            isinstance(x, tuple) and len(x) == 2 and x[0] is sentinel and x[1] == source_node_id for x in ins
        )

    def on_op(_nid: int, name: str, ins: list[object], params: Any) -> object:
        if touches_source(ins):
            if name == "field":
                found.add(str(params["field"]))
            elif name == "fields":
                found.update(f for f in str(params["fields"]).split(",") if f)
            else:
                whole[0] = True
        return None

    def on_external(_nid: int, _fn: Any, ins: list[object]) -> object:
        """An External handed the source RECORD replays against every column, so the read list
        cannot narrow — the same rule `on_op` applies to a non-field op."""
        if touches_source(ins):
            whole[0] = True
        return None

    array.session.walk(array, source=on_source, op=on_op, external=on_external)
    return None if (whole[0] or not found) else found


def _evaluation_columns(
    array: Any, source_node_id: int, source_name: str, source_form: AwkwardForm
) -> tuple[str, ...]:
    """The per-task read list: the graph's SYNTACTIC source-field accesses, refined per field by
    the buffer projection.

    Evaluation replays EVERY recorded node — including field accesses whose buffers the output
    never touches (a zip's untouched legs) — so the syntactic set decides WHICH fields must
    exist; the buffer view then decides which LEAVES to read for each: DATA needs read their
    leaves, an offsets-only field reads its CHEAPEST CARRIER leaf (parquet has no standalone
    counter column — R15.8's translation). An empty tuple means "everything" (the whole source
    is consumed)."""
    accessed = _syntactic_fields(array, source_node_id)
    if accessed is None:
        return ()
    leaves = source_form.tt.layout.form.columns()
    needs = project_buffers(array).buffers_for(source_name)
    out: set[str] = set()
    for f in sorted(accessed):
        data_paths = [
            p for p, need in needs.items() if need.value == "data" and (p == f or p.startswith(f + "."))
        ]
        if data_paths:
            for p in data_paths:
                out.update(c for c in leaves if c == p or c.startswith(p + "."))
        else:
            under = sorted(c for c in leaves if c == f or c.startswith(f + "."))
            out.add(under[0] if under else f)
    return tuple(sorted(out))


def _resolve_behavior(behavior: Any) -> Any:
    """A behavior dict, or an importable "module:attr" reference (behavior dicts often contain
    lambdas, which do not pickle to process workers)."""
    if isinstance(behavior, str):
        import importlib  # noqa: PLC0415

        mod_name, _, attr = behavior.partition(":")
        return getattr(importlib.import_module(mod_name), attr)
    return behavior


def _memory_step(partition: Partition, n: int, steps: int) -> int:
    for s in range(steps):
        if ((s * n) // steps, ((s + 1) * n) // steps) == (partition.entry_start, partition.entry_stop):
            return s
    raise ValueError(f"{partition} does not match any of {steps} steps over {n} rows")


# ==== variation-aware write-out (§6.4) =========================================================
# The awkward idiom's `to_parquet(record, select=…)` materializes the SUPERSET of rows passing any
# universe's level-0 selection and augments the nominal record with same-dtype XOR value deltas +
# packbits validity masks, so every universe's post-selection values and row set reconstruct
# bit-for-bit via `read_varied`. The manifest that resolves it lands in C4 (§6.4e); this section is
# the row rule, the entry checks, the augmentation, and the §7.2 node-id unpack.

_VARY_PREFIX = "__vary_"
#: a bare jagged-collection record (`to_parquet(events.Jet, …)`) is a LIST of records, not a
#: row-level record, so it cannot carry sibling columns; it is wrapped under this reserved field.
_BASE_FIELD = "__base__"

_SelectKey = Any  # `int` (bare depth) or `tuple[str, int]` (field-scoped, depth >= 1)


def _flat_field(path: str) -> str:
    """§6.4b: the on-disk field spelling — a nested path flattened per level with `_` (`Jet.pt` ->
    `Jet_pt`; a bare-field skim's `pt` -> `pt`)."""
    return path.replace(".", "_")


def _entry_depth(key: _SelectKey) -> int:
    return key if isinstance(key, int) else key[1]


def _entry_flat(key: _SelectKey) -> str:
    """The mask column's `entry` token (`…__mask__0`, `…__mask__1`, `…__mask__Jet_1`)."""
    if isinstance(key, int):
        return str(key)
    field_name, depth = key
    return f"{_flat_field(field_name)}_{depth}"


def _entry_manifest(key: _SelectKey) -> int | list[Any]:
    """The manifest `entry`/levels element: a bare depth `int`, or `[flat_field, depth]`."""
    return key if isinstance(key, int) else [_flat_field(key[0]), key[1]]


def _level_sort_key(key: _SelectKey) -> tuple[int, str]:
    """§6.4e's explicit total order over the heterogeneous levels list: `(depth, flat_field or "")`
    — bare-depth before field-scoped of the same depth (`sorted()` cannot order the mixed list)."""
    return (_entry_depth(key), "" if isinstance(key, int) else _flat_field(key[0]))


def _normalize_select(select: Any) -> dict[_SelectKey, Any]:
    """§6.4a's key space: a single `Varied`/Array row mask (⇔ `{0: mask}`), or a mapping keyed by
    `int` (bare depth) or `(field_name, depth>=1)` (field-scoped)."""
    if isinstance(select, Mapping):
        out: dict[_SelectKey, Any] = {}
        for key, mask in select.items():
            field_scoped = (
                isinstance(key, tuple) and len(key) == 2
                and isinstance(key[0], str) and isinstance(key[1], int) and key[1] >= 1
            )
            if not (isinstance(key, int) or field_scoped):
                raise GraphedError(
                    f"select= keys are a bare depth `int` or a `(field_name, depth>=1)` tuple, not {key!r}"
                )
            out[key] = mask
        return out
    return {0: select}


def _tt(session: Session, array: Any) -> ak.Array:
    """The typetracer of an array's form — the awkward-idiom depth/leaf oracle (§6.4a's `.tt.ndim`)."""
    form = session.form(array)
    assert isinstance(form, AwkwardForm)
    return form.tt


def _record_context(value: Any) -> Any:
    """The row-space handle of a write operand: a plain value's own handle, a container's
    most-derived member handle (§2.3e)."""
    if isinstance(value, Varied):
        return most_derived_context(*value._members.values())
    return context_of(value)


def _lineage_refusal(record_ctx: Any, select_ctx: Any) -> None:
    """§6.4a(2a) lineage / §6.4b row-space, decided record-time by CONTEXT-HANDLE reachability over
    the level-0 mask. Vary IDENTITY links do not move the row space (§6.1d kind (2)); a mask or
    project link does. Direction picks the message: a record selected BELOW its mask is §6.4b
    (row-space, NEVER offsets); a mask more derived than the record is §6.4a(2a)."""
    if record_ctx is None:
        return  # absent-operand (i): a loose record carries no handle, so (2a) is skipped
    if select_ctx is None:
        raise GraphedError(
            f"the record read through {record_ctx!r} was given a select= mask that carries no "
            "context handle, so its lineage cannot be checked against the record (§6.4a(2a·ii))"
        )
    if record_ctx is select_ctx:
        return
    if select_ctx._is_ancestor_of(record_ctx):
        if all(kind == "vary" for kind, _ in record_ctx._links_below(select_ctx)):
            return
        raise GraphedError(
            f"the record read through {record_ctx!r} is selected into a narrower ROW SPACE than its "
            f"select= mask ({select_ctx!r}) defines: a stored field would live below the superset the "
            "mask spans, and a mask has no inverse to lift it back — read the record at the mask's "
            "row space (§6.4b)"
        )
    if record_ctx._is_ancestor_of(select_ctx):
        if all(kind == "vary" for kind, _ in select_ctx._links_below(record_ctx)):
            return
        raise GraphedError(
            f"the record read through {record_ctx!r} is not the context its select= mask "
            f"({select_ctx!r}) derives from: the mask is more derived than the record's rows (§6.4a(2a))"
        )
    raise GraphedError(
        f"the record ({record_ctx!r}) and its select= mask ({select_ctx!r}) are on divergent "
        "branches; the mask does not select the record's rows (§6.4a(2a))"
    )


def _resolve_field(nodes: list[dict[str, Any]], session: Session, node_id: int, name: str) -> int:
    """The CANONICAL value node for `field(node, name)`, seeing through zip/with_field projection —
    the M4 optimizer does not push field projection through a `zip`, so the raw per-label field node
    always differs even when the value does not. Distinguishes a genuinely varying leaf (§6.4b) from
    an untouched one that only the record's construction couples to the shift."""
    nd = nodes[node_id]
    op = nd["name"] if nd["kind"] == "op" else None
    if op == "ak.zip":
        fields = str(nd["params"]["fields"]).split(",")
        return int(nd["inputs"][fields.index(name)])
    if op == "ak.with_field":
        if str(nd["params"]["field"]) == name:
            return int(nd["inputs"][1])
        return _resolve_field(nodes, session, int(nd["inputs"][0]), name)
    return int(getattr(Array(session, node_id), name).node_id)


def _canonical_leaf(nodes: list[dict[str, Any]], session: Session, member: Any, path: str) -> int:
    node_id: int = member.node_id
    for part in path.split("."):
        node_id = _resolve_field(nodes, session, node_id, part)
    return node_id


def _varied_leaves(session: Session, record: Any, labels: tuple[str, ...]) -> list[str]:
    """The leaf paths whose canonical value node DIFFERS across labels (structurally varied, §6.4b);
    an all-labels-equal leaf (unvaried, or coupled only through a zip) contributes no delta column."""
    nodes = session._store.nodes()
    leaves = _tt(session, member_of(record, "nominal")).layout.form.columns()
    varied = []
    for path in leaves:
        canon = {_canonical_leaf(nodes, session, member_of(record, label), path) for label in labels}
        if len(canon) > 1:
            varied.append(path)
    return varied


def _check_collision(varied: Sequence[str]) -> None:
    """§6.4b: two varied leaves that flatten to one on-disk name are refused naming both (a nested
    `Jet.pt` beside a flat `Jet_pt`, both -> `__vary_L__Jet_pt`)."""
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
    """§6.4a: a bare depth-`k>=1` key is legal only when the record is ITSELF jagged at depth `k`; a
    flat record with two or more independently jagged fields refuses it, naming the ambiguity."""
    nominal = member_of(record, "nominal")
    record_ndim = _tt(session, nominal).ndim
    for key in keys:
        if isinstance(key, int) and key >= 1 and record_ndim <= key:
            form = _tt(session, nominal)
            jagged = [f for f in form.fields if form[f].ndim > key]
            raise GraphedError(
                f"the bare depth-{key} select= key needs the record to be jagged at depth {key}; this "
                f"record is a flat record whose jagged fields {jagged} each have their own offsets — "
                f"name the field with a ({jagged[0]!r}, {key}) key instead (§6.4a)"
            )


def _check_depth(session: Session, entries: Mapping[_SelectKey, Any]) -> None:
    """§6.4a(2c): a mask supplied at level `k` must have depth `k` over the record's structure —
    typetracer ndim `k + 1` (flat = ndim 1 at level 0), at EVERY supplied level, record-time."""
    for key, mask in entries.items():
        ndim = _tt(session, member_of(mask, "nominal")).ndim
        depth = _entry_depth(key)
        if ndim != depth + 1:
            raise GraphedError(
                f"the select= mask at level {key!r} has depth {ndim - 1} (ndim {ndim}); level "
                f"{depth} requires a depth-{depth} mask (ndim {depth + 1}) over the record (§6.4a(2c))"
            )


def _uint_view(a: np.ndarray) -> np.ndarray:
    return a.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize])


def _encode_delta(nominal: ak.Array, label: ak.Array, jagged: bool) -> ak.Array:
    """§6.4c: the same-dtype XOR bit-delta vs nominal (zero wherever a label equals nominal)."""
    if jagged:
        fn = _uint_view(ak.to_numpy(ak.flatten(nominal)))
        fl = _uint_view(ak.to_numpy(ak.flatten(label)))
        return ak.unflatten(fl ^ fn, ak.num(nominal))
    return ak.Array(_uint_view(ak.to_numpy(label)) ^ _uint_view(ak.to_numpy(nominal)))


def _encode_mask(mask: ak.Array) -> ak.Array:
    """§6.4c: a per-row `packbits` validity mask stored as `var * uint8` (row-aligned; a flat level-0
    mask packs one bit per row, a jagged level-k>=1 mask packs its row's objects)."""
    return ak.Array([
        list(np.packbits(np.asarray(row, dtype=bool) if isinstance(row, list) else [bool(row)]))
        for row in ak.to_list(mask)
    ])


@dataclass(frozen=True)
class _ValueSpec:
    """A structurally-varied leaf: its on-disk name, jaggedness (record-time), and per-label output
    node ids (nominal is the XOR reference AND, on an all-zero-delta collapse, the §7.2 dedup source)."""

    path: str
    flat: str
    jagged: bool
    members: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _MaskSpec:
    """A supplied selection level: its on-disk `entry` token, manifest value, depth, the field its
    level->=1 mask filters (`None` = the record's own axis), and per-label output node ids."""

    entry_flat: str
    manifest: int | list[Any]
    depth: int
    field_path: str | None
    members: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _VariedWritePart:
    """The picklable per-partition varied-write task (§6.4). Widens `_WritePart`: one `compile_ir`
    over base + per-label values + per-label masks, resolved BY NODE ID through `rank` (§7.2), XOR/
    packbits-encoded on the evaluated buffers, and written as one augmented row-level record."""

    compiled: CompiledGraph
    source_name: str
    columns: tuple[str, ...]
    destination: str
    prefix: str
    steps_per_file: int
    bases: tuple[tuple[Any, int], ...]
    labels: tuple[str, ...]
    #: §7.2: record node id -> position in `evaluate_ir`'s DEDUPLICATED output list. Driver-derived,
    #: shipped in the closure — the worker cannot recompute it from the reduced bytes alone.
    rank: tuple[tuple[int, int], ...]
    base_id: int
    value_specs: tuple[_ValueSpec, ...]
    mask_specs: tuple[_MaskSpec, ...]
    wrapped: bool
    reader: Any = None
    behavior: Any = None
    memory_data: ak.Array | None = None
    memory_rows: int = 0
    manifest: Mapping[str, Any] | None = None  # C4 fills the KV manifest; None writes plain parquet

    def _chunk(self, partition: Partition, resources: WorkerResources) -> tuple[Any, int]:
        if self.reader is not None:
            return self.reader.read_partition(partition, self.columns or None, resources), gw.blind_part_index(
                partition, dict(self.bases)
            )
        if self.memory_data is not None:
            return self.memory_data[partition.entry_start : partition.entry_stop], _memory_step(
                partition, self.memory_rows, self.steps_per_file
            )
        raise TypeError("write task has neither a partition reader nor in-memory data")

    def __call__(self, partition: Partition, resources: WorkerResources) -> list[str]:
        chunk, index = self._chunk(partition, resources)
        backend = AwkwardBackend(behavior=_resolve_behavior(self.behavior))
        values = evaluate_ir(self.compiled, cast("Backend", backend), {self.source_name: chunk})
        rank = dict(self.rank)

        def resolved(node_id: int) -> ak.Array:
            return ak.Array(values[rank[node_id]])

        base = resolved(self.base_id)
        cols: dict[str, ak.Array] = {}
        for spec in self.value_specs:
            nominal_value = resolved(dict(spec.members)["nominal"])
            if spec.jagged:
                nominal_counts = np.asarray(ak.num(nominal_value))
                for label, node_id in spec.members:
                    if label == "nominal":
                        continue
                    if not np.array_equal(np.asarray(ak.num(resolved(node_id))), nominal_counts):
                        # §6.4d: differing per-label offsets have no representable XOR delta —
                        # execution-time (offsets are data), naming the label and the field.
                        raise GraphedError(
                            f"variation {label!r} changes the multiplicity of field {spec.path!r}: its "
                            "per-label offsets differ from nominal's, so it has no same-shaped delta "
                            "(§6.4d — a multiplicity-changing stored variation is Phase 2)"
                        )
            for label, node_id in spec.members:
                if label == "nominal":
                    continue
                cols[f"{_VARY_PREFIX}{label}__{spec.flat}"] = _encode_delta(
                    nominal_value, resolved(node_id), spec.jagged
                )
        for mask_spec in self.mask_specs:
            field_values = base if mask_spec.field_path is None else base[mask_spec.field_path]
            for label, node_id in mask_spec.members:
                mask = resolved(node_id)
                if len(mask) != len(base):  # §6.4a(2b) row-count, execution-time
                    raise GraphedError(
                        f"the level-{mask_spec.entry_flat} select= mask for {label!r} spans {len(mask)} "
                        f"rows but the record spans {len(base)} (§6.4a(2b))"
                    )
                if mask_spec.depth >= 1 and not np.array_equal(
                    np.asarray(ak.num(mask)), np.asarray(ak.num(field_values))
                ):  # level->=1 STRUCTURAL: the mask's offsets must match the field's (§6.4a)
                    raise GraphedError(
                        f"the level-{mask_spec.entry_flat} select= mask for {label!r} does not share the "
                        "offsets of the field it filters (§6.4a level->=1 structural check)"
                    )
                cols[f"{_VARY_PREFIX}{label}__mask__{mask_spec.entry_flat}"] = _encode_mask(mask)

        if self.wrapped:
            fields: dict[str, ak.Array] = {_BASE_FIELD: base}
        else:
            fields = {name: base[name] for name in base.fields}
        payload = ak.zip({**fields, **cols}, depth_limit=1)
        os.makedirs(self.destination, exist_ok=True)
        path = gpq.part_path(self.destination, index, prefix=self.prefix)
        _write_augmented(payload, path, self.manifest)
        return [path]


def _write_augmented(payload: ak.Array, path: str, manifest: Mapping[str, Any] | None) -> None:
    """Write the augmented record. C4 merges the §6.4e manifest through the PUBLIC arrow route; until
    then a plain `ak.to_parquet` (the unvaried byte-golden path is a separate branch, never touched)."""
    ak.to_parquet(payload, path)


def _evaluation_columns_union(
    outputs: Sequence[Any], source_node_id: int, source_name: str, source_form: AwkwardForm
) -> tuple[str, ...]:
    """The read list widened over EVERY marked output (§6.4f): the union of each output's projected
    source-field accesses, `()` (read everything) absorbing when any output consumes the whole source."""
    union: set[str] = set()
    for out in outputs:
        cols = _evaluation_columns(out, source_node_id, source_name, source_form)
        if not cols:
            return ()
        union.update(cols)
    return tuple(sorted(union))


def _write_varied(
    record: Any,
    destination: str,
    select: Any,
    *,
    steps_per_file: int,
    compute: bool,
    executor: Any | None,
    prefix: str,
    behavior: Any,
) -> list[str] | Plan[list[str]]:
    """§6.4a-f: the superset row rule, the record-time entry checks, and the augmented plan."""
    entries = _normalize_select(select)
    masks = tuple(entries.values())
    labels = union_labels(record, *masks)
    nominal_record = member_of(record, "nominal")
    session: Session = nominal_record.session

    # --- record-time entry checks (raise at the call; `compute=False` still raises) ---
    level0 = entries.get(0)
    _lineage_refusal(_record_context(record), _record_context(level0) if level0 is not None else None)
    _check_depth(session, entries)
    _check_bare_key(session, record, tuple(entries))
    varied = _varied_leaves(session, record, labels)
    _check_collision(varied)

    # --- superset rows: the level-0 OR over the per-label masks (ordinary graph ops, §6.4a) ---
    if level0 is not None:
        superset = functools.reduce(operator.or_, (member_of(level0, label) for label in labels))
    else:
        superset = None
    base = nominal_record if superset is None else nominal_record[superset]

    nodes = session._store.nodes()
    value_specs: list[_ValueSpec] = []
    outputs: list[Any] = [base]
    for path in varied:
        jagged = _tt(session, Array(session, _canonical_leaf(nodes, session, nominal_record, path))).ndim > 1
        members: list[tuple[str, int]] = []
        for label in labels:
            leaf = Array(session, _canonical_leaf(nodes, session, member_of(record, label), path))
            leaf = leaf if superset is None else leaf[superset]
            members.append((label, leaf.node_id))
            outputs.append(leaf)
        value_specs.append(_ValueSpec(path, _flat_field(path), jagged, tuple(members)))

    mask_specs: list[_MaskSpec] = []
    for key in sorted(entries, key=_level_sort_key):
        mask = entries[key]
        field_path = None if isinstance(key, int) else key[0]
        members = []
        for label in labels:
            member = member_of(mask, label)
            member = member if superset is None else member[superset]
            members.append((label, member.node_id))
            outputs.append(member)
        mask_specs.append(
            _MaskSpec(_entry_flat(key), _entry_manifest(key), _entry_depth(key), field_path, tuple(members))
        )

    compiled = compile_ir(session, *outputs)
    _refuse_optimizer_merge(session, outputs, compiled)

    rank = {node_id: i for i, node_id in enumerate(dict.fromkeys(out.node_id for out in outputs))}
    wrapped = _tt(session, nominal_record).ndim > 1

    ((source_id, data),) = session.sources().items()
    source_name = session.source_name(source_id)
    source_form = session.form_of(source_id)
    assert isinstance(source_form, AwkwardForm)
    columns = _evaluation_columns_union(outputs, source_id, source_name, source_form)

    common: dict[str, Any] = {
        "compiled": compiled,
        "source_name": source_name,
        "columns": columns,
        "destination": destination,
        "prefix": prefix,
        "steps_per_file": steps_per_file,
        "labels": labels,
        "rank": tuple(rank.items()),
        "base_id": base.node_id,
        "value_specs": tuple(value_specs),
        "mask_specs": tuple(mask_specs),
        "wrapped": wrapped,
        "behavior": behavior,
    }
    if isinstance(data, PartitionedSource):
        partitions = data.partitions(steps_per_file)
        keys = list(dict.fromkeys((p.uri, p.tree) if p.tree else p.uri for p in partitions))
        writer = _VariedWritePart(
            bases=tuple(gw.file_bases(keys, steps_per_file).items()), reader=data, **common
        )
    else:
        whole = ak.Array(data() if callable(data) else data)
        n = len(whole)
        partitions = tuple(
            Partition(f"memory://{source_name}", "", (s * n) // steps_per_file, ((s + 1) * n) // steps_per_file)
            for s in range(steps_per_file)
        )
        writer = _VariedWritePart(bases=(), memory_data=whole, memory_rows=n, **common)

    plan = gw.write_plan(partitions, writer)
    if not compute:
        return plan
    runner = executor if executor is not None else SequentialRunner()
    return list(runner.run(plan).value)


def _refuse_optimizer_merge(session: Session, outputs: Sequence[Any], compiled: CompiledGraph) -> None:
    """§6.4f binds §7.2's optimizer-merge shortfall onto the write path (record-time, at the call):
    the varied unpack resolves BY NODE ID, so the compiled output count must match the DISTINCT
    marked ids; a `w * 1.0` label merges two distinct record ids into one compiled output, and the
    node-id table can then no longer tell the labels apart."""
    marked = len(dict.fromkeys(out.node_id for out in outputs))
    compiled_outputs = len(graphed.core.GraphStore.deserialize(compiled.ir).outputs())
    if compiled_outputs >= marked:
        return
    raise GraphedError(
        f"the optimizer merged outputs that record as distinct nodes ({marked} marked, "
        f"{compiled_outputs} compiled), so this varied write's labels can no longer be told apart. "
        'Spell a label whose value equals another with the SAME expression (variations={"1": w}, '
        "not w * 1.0), which routes it through the supported record-time dedup instead (§6.4f/§7.2)"
    )


def to_parquet(
    array: Any,
    destination: str,
    *,
    select: Any = None,
    steps_per_file: int = 1,
    compute: bool = True,
    executor: Any | None = None,
    prefix: str = "part",
    column: str = "data",
    behavior: Any = None,
) -> list[str] | Plan[list[str]]:
    """Write the deferred array to parquet parts, one per partition (R15.4 semantics).

    With ``compute=False`` returns the task graph of write tasks; with ``compute=True`` runs that
    SAME plan (``SequentialRunner`` by default; pass any R7 executor). The array must be recorded
    over exactly one source; the per-task read list comes from the recorded graph's projection.

    ``select=`` turns this into a §6.4 variation-aware write: ``array`` is the PRE-selection record
    and ``select`` carries the per-level ``Varied`` mask(s) (a single row mask ⇔ ``{0: mask}``, or a
    ``{0: event_mask, ("Jet", 1): jet_mask}`` mapping). The writer materializes the SUPERSET of rows
    passing any universe's selection, augments the nominal record with reconstruction data, and
    returns the ordinary part-path list — a §2.3d *accepting* verb. Without ``select=`` the path is
    unchanged (byte-identical, §6.4g)."""
    if select is not None:
        return _write_varied(
            array, destination, select, steps_per_file=steps_per_file, compute=compute,
            executor=executor, prefix=prefix, behavior=behavior,
        )
    session: Session = array.session
    sources = session.sources()
    if len(sources) != 1:
        raise TypeError(f"to_parquet needs an array recorded over exactly one source, got {len(sources)}")
    ((node_id, data),) = sources.items()
    source_name = session.source_name(node_id)
    source_form = session.form_of(node_id)
    assert isinstance(source_form, AwkwardForm)  # this backend recorded the source
    columns = _evaluation_columns(array, node_id, source_name, source_form)
    compiled = compile_ir(session, array)

    if isinstance(data, PartitionedSource):
        # the generic path: ANY source describing its own partitioning (parquet datasets, the
        # ROOT reader integration, ...) is written partition by partition — its whole-dataset
        # loader is never invoked
        partitions = data.partitions(steps_per_file)
        keys = list(dict.fromkeys((p.uri, p.tree) if p.tree else p.uri for p in partitions))
        writer = _WritePart(
            compiled=compiled,
            source_name=source_name,
            columns=columns,
            destination=destination,
            prefix=prefix,
            steps_per_file=steps_per_file,
            bases=tuple(gw.file_bases(keys, steps_per_file).items()),
            column=column,
            reader=data,
            behavior=behavior,
        )
    else:
        whole = ak.Array(data() if callable(data) else data)
        n = len(whole)
        partitions = tuple(
            Partition(
                f"memory://{source_name}", "", (s * n) // steps_per_file, ((s + 1) * n) // steps_per_file
            )
            for s in range(steps_per_file)
        )
        writer = _WritePart(
            compiled=compiled,
            source_name=source_name,
            columns=columns,
            destination=destination,
            prefix=prefix,
            steps_per_file=steps_per_file,
            bases=(),
            column=column,
            behavior=behavior,
            memory_data=whole,
            memory_rows=n,
        )

    plan = gw.write_plan(partitions, writer)
    if not compute:
        return plan
    runner = executor if executor is not None else SequentialRunner()
    return list(runner.run(plan).value)
