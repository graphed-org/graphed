"""Partitioned parquet I/O for the awkward backend (M15.2, dask-awkward parity plan).

Specializes the backend-agnostic `graphed.parquet` base: the awkward pieces are exactly two —
the FORM comes from the arrow schema alone (awkward's own reader on a zero-row file of that
schema; no event data is read at construction) and the per-partition codec is
`ak.from_parquet`/`ak.to_parquet`.

`to_parquet` follows the R15.4/R15.5 contract proven by the uproot integration: compute-disabled
returns a task graph of write tasks; each task evaluates the array's graph through the COMPILED
IR (R7.8 — compiled once at the driver, never re-recorded per partition), reads only the
PROJECTED columns (R15.3 — the read list is wired from the projection), derives its own output
part index from its partition (R15.9), and writes one parquet part.
"""

from __future__ import annotations

import functools
import importlib
import operator
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Literal, cast, overload

import awkward as ak
import numpy as np

from graphed import (
    Array,
    Backend,
    CompiledGraph,
    Session,
    compile_ir,
    evaluate_ir,
    refuse_chunk_partials,
)
from graphed import parquet as gpq
from graphed import write as gw
from graphed.aggregate import plan_services, shipped_evaluators
from graphed.core import Partition
from graphed.core.execution import Plan, SequentialRunner, WorkerResources
from graphed.errors import GraphedError
from graphed.services import bind_externals
from graphed.session import reader
from graphed.varied import member_of, refuse_container, union_labels
from graphed.write import PartitionedSource

from ._read_list import _evaluation_columns, _evaluation_columns_union
from ._varied_read import read_varied as read_varied
from ._varied_write import (
    _BASE_FIELD,
    _VARY_PREFIX,
    _build_manifest,
    _canonical_leaf,
    _check_bare_key,
    _check_collision,
    _check_depth,
    _counts,
    _encode_delta,
    _encode_mask,
    _entry_depth,
    _entry_flat,
    _entry_manifest,
    _flat_field,
    _is_option,
    _level_sort_key,
    _lineage_refusal,
    _normalize_select,
    _record_context,
    _refuse_optimizer_merge,
    _SelectKey,
    _tt,
    _type_of,
    _varied_leaves,
    _write_augmented,
    _xor_representable,
)
from .backend import AwkwardBackend, AwkwardForm, _canonical, shipped_behavior
from .functions import fill_none


def _schema_form(paths: Sequence[str], columns: Sequence[str] | None) -> AwkwardForm:
    """Return the dataset's form from the arrow SCHEMA alone.

    The first file is authoritative; nothing is decoded.

    The form is what awkward's OWN reader gives a ZERO-ROW parquet file of that schema, not what
    `ak.from_arrow_schema` gives the schema: the latter sees arrow types only and drops the record
    names and node parameters awkward stored beside them, so a deferred read of a file awkward
    wrote would not type like an eager read of it and a behavior keyed on a record name would
    never resolve. The zero-row file is written OUTSIDE the dataset, and no event data is read.
    """
    with tempfile.TemporaryDirectory() as scratch:
        empty = os.path.join(scratch, "schema.parquet")
        gpq._pq().write_table(gpq.schema_of(paths).empty_table(), empty)
        form = ak.from_parquet(empty).layout.form
    if columns:
        form = form.select_columns(list(columns))
    tt = ak.Array(form.length_zero_array(highlevel=False).to_typetracer(forget_length=True))
    return AwkwardForm(tt)


@dataclass(frozen=True)
class _DatasetLoader:
    """Lazy whole-dataset loader for the reference ``materialize``.

    AND a ``graphed.write.PartitionedSource``, so the generic writer reads it partition by
    partition (the whole-dataset path is only ever the reference ``materialize``).
    """

    paths: tuple[str, ...]
    columns: tuple[str, ...] | None
    steps_per_file: int = 1
    open_files: bool = False

    def source_identity(self) -> object:
        """Return what this loader reads and how it partitions it."""
        return (type(self), self.paths, self.columns, self.steps_per_file, self.open_files)

    def __call__(self) -> ak.Array:
        return ak.from_parquet(
            list(self.paths), columns=list(self.columns) if self.columns else None
        )

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return gpq.make_partitions(
            self.paths, steps_per_file=steps_per_file, open_files=self.open_files
        )

    def read_partition(
        self, partition: Partition, columns: Sequence[str] | None, resources: Any
    ) -> ak.Array:
        return read_parquet_partition(partition, columns if columns else (self.columns or None))


@overload
def from_parquet(
    session: Session,
    name: str,
    path: str | Sequence[str],
    *,
    columns: Sequence[str] | None = None,
    steps_per_file: int = 1,
    open_files: bool = False,
) -> Any: ...
@overload
def from_parquet(
    name: str,
    path: str | Sequence[str],
    *,
    columns: Sequence[str] | None = None,
    steps_per_file: int = 1,
    open_files: bool = False,
    session: Session | None = None,
) -> Any: ...
@reader(AwkwardBackend)
def from_parquet(
    name: str,
    path: str | Sequence[str],
    *,
    columns: Sequence[str] | None = None,
    steps_per_file: int = 1,
    open_files: bool = False,
    session: Session | None = None,
) -> Any:
    """Return a deferred awkward array over a parquet dataset (file / directory / glob / list).

    It records into ``session``, else the current Session.

    The form is built from METADATA alone; no event data is read here. ``steps_per_file`` and
    ``open_files`` shape the dataset's default partitioning, which a plan takes unless its own
    ``steps_per_file=`` is given: each file splits into ``steps_per_file`` partitions, blind by
    default (no file is opened to plan, R7.9) or, with ``open_files=True``, with exact entry
    ranges from each file's metadata row count, read when the plan is built. Both are source
    params, so they are part of the source's identity.
    """
    paths = gpq.discover(path)
    form = _schema_form(paths, columns)
    if steps_per_file < 1:
        raise ValueError(f"steps_per_file must be >= 1, got {steps_per_file}")
    loader = _DatasetLoader(paths, tuple(columns) if columns else None, steps_per_file, open_files)
    return gpq.deferred_source(
        name,
        paths=paths,
        form=form,
        loader=loader,
        session=session,
        steps_per_file=steps_per_file,
        open_files=open_files,
    )


def read_parquet_partition(partition: Partition, columns: Sequence[str] | None = None) -> ak.Array:
    """Read one partition (resolving blind ones at read time), restricted to ``columns``."""
    part = gpq.resolve_partition(partition)
    arr = ak.from_parquet(part.uri, columns=list(columns) if columns else None)
    return arr[part.entry_start : part.entry_stop]


def _resolve_behavior(behavior: Any) -> Any:
    """Return a behavior dict, or an importable "module:attr" reference.

    Behavior dicts often contain lambdas, which do not pickle to process workers.
    """
    if isinstance(behavior, str):
        mod_name, _, attr = behavior.partition(":")
        return getattr(importlib.import_module(mod_name), attr)
    return behavior


def _keyed_behavior(behavior: Any) -> Any:
    """Return what a task key holds for ``behavior``: a reference's target, in canonical order."""
    resolved = _resolve_behavior(behavior)
    return _canonical(resolved) if resolved and isinstance(behavior, str) else resolved


def _session_behavior(session: Session, behavior: Any) -> Any:
    """Return ``behavior``, or when ``None`` the session backend's, as a plan ships it."""
    chosen = getattr(session.backend, "_behavior", None) if behavior is None else behavior
    return shipped_behavior(chosen) if chosen and isinstance(chosen, Mapping) else behavior


# ---- deferred writing ------------------------------------------------------------------------
@dataclass(frozen=True)
class _WritePart:
    """The picklable per-partition write task: compiled IR in, one parquet part out."""

    checkpointable: ClassVar[bool] = True
    checkpoint_resolve: ClassVar[Mapping[str, Callable[[Any], Any]]] = {
        "behavior": _keyed_behavior
    }

    compiled: CompiledGraph
    source_name: str
    #: shipped to `read_partition` verbatim; `None` = the source's own selection. The driver's own
    #: "everything" sentinel `()` is translated to `None` where it is computed, so a source that
    #: DECLARES `()` gets `()`.
    columns: tuple[str, ...] | None
    destination: str
    prefix: str
    steps_per_file: int
    bases: tuple[tuple[Any, int], ...]
    column: str = "data"
    reader: Any = None  # a graphed.write.PartitionedSource (picklable)
    rows: tuple[tuple[Any, int], ...] = ()  # each eagerly partitioned file's row count
    behavior: Any = None  # a behavior dict or an importable "module:attr" reference
    memory_data: ak.Array | None = None  # in-memory source payload (bounded by the dataset)
    memory_rows: int = 0
    externals: tuple[tuple[str, Callable[..., object]], ...] = ()

    def bind_services(self, endpoints: Mapping[str, str]) -> _WritePart:
        return replace(self, externals=bind_externals(self.externals, endpoints))

    def part_paths(self, partition: Partition) -> list[str]:
        return _part_paths(self, partition)

    def __call__(self, partition: Partition, resources: WorkerResources) -> list[str]:
        if self.reader is not None:
            chunk = self.reader.read_partition(partition, self.columns, resources)
        elif self.memory_data is not None:
            chunk = self.memory_data[partition.entry_start : partition.entry_stop]
        else:  # pragma: no cover - every source is a protocol reader or in-memory
            raise TypeError("write task has neither a partition reader nor in-memory data")
        backend = AwkwardBackend(behavior=_resolve_behavior(self.behavior))
        (out,) = evaluate_ir(
            self.compiled,
            cast("Backend", backend),
            {self.source_name: chunk},
            externals=dict(self.externals),
            chunked=True,
        )
        payload = _payload(out, self.column)
        (path,) = self.part_paths(partition)
        gw.prepare_part(path)
        ak.to_parquet(payload, path)
        return [path]


def _file_key(partition: Partition) -> Any:
    return (partition.uri, partition.tree) if partition.tree else partition.uri


def _file_tables(partitions: Sequence[Partition], steps_per_file: int) -> dict[str, Any]:
    """Return a reader-backed writer's ``bases`` and ``rows`` over its source's ``partitions``.

    Every listing of a file has one first partition (blind step 0, or an eager range from 0), so
    each gets a key and a repeated one is refused; a file's last partition ends at its row count,
    against which an eager step is reconstructed.
    """
    files = [
        _file_key(p)
        for p in partitions
        if p.blind_step == 0 or (not p.is_blind and p.entry_start == 0)
    ]
    rows = {_file_key(p): p.entry_stop for p in partitions if not p.is_blind}
    return {
        "bases": tuple(gw.file_bases(files, steps_per_file).items()),
        "rows": tuple(rows.items()),
    }


def _part_paths(writer: _WritePart | _VariedWritePart, partition: Partition) -> list[str]:
    """Return a deferred writer's one part for ``partition``: its ``part_paths`` hook."""
    if writer.reader is None:
        index = _memory_step(partition, writer.memory_rows, writer.steps_per_file)
    elif partition.is_blind:
        index = gw.blind_part_index(partition, dict(writer.bases))
    else:
        key = _file_key(partition)
        index = dict(writer.bases)[key] + _memory_step(
            partition, dict(writer.rows)[key], writer.steps_per_file
        )
    return [gpq.part_path(writer.destination, index, prefix=writer.prefix)]


def _payload(value: object, column: str) -> ak.Array:
    """Return What a part holds: a record as is, anything else as the one field ``column``."""
    result = ak.Array(value)
    return result if result.fields else ak.Array({column: result})


@dataclass(frozen=True)
class _ArrowParquet:
    """`parquet_write`'s codec: ``ak.to_arrow_table`` then ``pq.write_table``.

    Each takes its own options, and the part's key-value metadata REPLACES the schema's
    (``to_arrow_table`` always stamps ``ak:parameters``, whichever ``extensionarray``).
    """

    column: str
    arrow_options: tuple[tuple[str, Any], ...]
    parquet_options: tuple[tuple[str, Any], ...]

    def __call__(self, value: object, path: str, kv: Mapping[str, str] | None) -> None:
        table = ak.to_arrow_table(_payload(value, self.column), **dict(self.arrow_options))
        if kv is not None:
            table = table.replace_schema_metadata(kv)
        fs, where = gw.part_fs(path)
        gpq._pq().write_table(table, where, **{"filesystem": fs, **dict(self.parquet_options)})


def parquet_write(
    array: Array,
    destination: str,
    *,
    name: Callable[[Partition], str],
    metadata: Mapping[str, Any] | None = None,
    arrow_options: Mapping[str, Any] | None = None,
    parquet_options: Mapping[str, Any] | None = None,
    column: str = "data",
) -> gw.PartWrite:
    """Return a parquet part per task of ``array``, for :func:`graphed.aggregate_plan` ``writes=``.

    Each part is ``ak.to_arrow_table(array_chunk, **arrow_options)`` written by
    ``pyarrow.parquet.write_table(table, path, **parquet_options)`` at
    :func:`graphed.write.join_part` ``(destination, name(partition))``; a ``destination``
    containing ``://`` is an fsspec URL, written through its filesystem. A non-record array is
    written as the field ``column``. ``metadata`` (see :class:`~graphed.write.PartWrite`) replaces
    the schema's key-value metadata when given. Column order is the array's: sort a record's fields
    in the graph (``rec[sorted(rec.fields)]``) to write them sorted.
    """
    refuse_container("graphed.awkward.parquet_write", array, *(metadata or {}).values())
    codec = _ArrowParquet(
        column, tuple((arrow_options or {}).items()), tuple((parquet_options or {}).items())
    )
    return gw.PartWrite(
        array=array, destination=destination, name=name, codec=codec, metadata=metadata
    )


def _memory_step(partition: Partition, n: int, steps: int) -> int:
    for s in range(steps):
        if ((s * n) // steps, ((s + 1) * n) // steps) == (
            partition.entry_start,
            partition.entry_stop,
        ):
            return s
    raise ValueError(f"{partition} does not match any of {steps} steps over {n} rows")


@dataclass(frozen=True)
class _ValueSpec:
    """A structurally-varied leaf.

    Its on-disk name, jaggedness (record-time), and per-label output node ids (nominal is the XOR
    reference AND, on an all-zero-delta collapse, the §7.2 dedup source).
    """

    path: str
    flat: str
    jagged: bool
    members: tuple[tuple[str, int], ...]
    #: labels stored as-is (`"base"`): a type change from nominal, or a leaf `_xor_representable`
    #: refuses.
    verbatim: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _MaskSpec:
    """A supplied selection level.

    Its on-disk `entry` token, manifest value, depth, the field its level->=1 mask filters
    (`None` = the record's own axis), and per-label output node ids.
    """

    entry_flat: str
    manifest: int | list[Any]
    depth: int
    field_path: str | None
    members: tuple[tuple[str, int], ...]
    #: labels whose mask is option-typed: stored as-is (`"option"`), since packbits has no `None`.
    option: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _VariedWritePart:
    """The picklable per-partition varied-write task (§6.4).

    Widens `_WritePart`: one `compile_ir` over base + per-label values + per-label masks, resolved
    BY NODE ID through `rank` (§7.2), XOR/ packbits-encoded on the evaluated buffers, and written
    as one augmented row-level record.
    """

    checkpointable: ClassVar[bool] = True
    checkpoint_resolve: ClassVar[Mapping[str, Callable[[Any], Any]]] = {
        "behavior": _keyed_behavior
    }

    compiled: CompiledGraph
    source_name: str
    #: as `_WritePart.columns`: verbatim to `read_partition`, `None` = the source's own selection.
    columns: tuple[str, ...] | None
    destination: str
    prefix: str
    steps_per_file: int
    bases: tuple[tuple[Any, int], ...]
    labels: tuple[str, ...]
    #: §7.2: record node id -> position in `evaluate_ir`'s DEDUPLICATED output list.
    #: Driver-derived, shipped in the closure — the worker cannot recompute it from the reduced
    #: bytes alone.
    rank: tuple[tuple[int, int], ...]
    base_id: int
    value_specs: tuple[_ValueSpec, ...]
    mask_specs: tuple[_MaskSpec, ...]
    wrapped: bool
    reader: Any = None
    rows: tuple[tuple[Any, int], ...] = ()  # as `_WritePart.rows`
    behavior: Any = None
    memory_data: ak.Array | None = None
    memory_rows: int = 0
    manifest: Mapping[str, Any] | None = (
        None  # C4 fills the KV manifest; None writes plain parquet
    )
    externals: tuple[tuple[str, Callable[..., object]], ...] = ()

    def bind_services(self, endpoints: Mapping[str, str]) -> _VariedWritePart:
        return replace(self, externals=bind_externals(self.externals, endpoints))

    def _chunk(self, partition: Partition, resources: WorkerResources) -> Any:
        if self.reader is not None:
            return self.reader.read_partition(partition, self.columns, resources)
        if self.memory_data is not None:
            return self.memory_data[partition.entry_start : partition.entry_stop]
        raise TypeError("write task has neither a partition reader nor in-memory data")

    def part_paths(self, partition: Partition) -> list[str]:
        return _part_paths(self, partition)

    def __call__(self, partition: Partition, resources: WorkerResources) -> list[str]:
        chunk = self._chunk(partition, resources)
        backend = AwkwardBackend(behavior=_resolve_behavior(self.behavior))
        values = evaluate_ir(
            self.compiled,
            cast("Backend", backend),
            {self.source_name: chunk},
            externals=dict(self.externals),
            chunked=True,
        )
        rank = dict(self.rank)

        def resolved(node_id: int) -> ak.Array:
            return ak.Array(values[rank[node_id]])

        base = resolved(self.base_id)
        cols = {**self._value_columns(resolved), **self._mask_columns(resolved, base)}

        if self.wrapped:
            fields: dict[str, ak.Array] = {_BASE_FIELD: base}
        else:
            fields = {name: base[name] for name in base.fields}
        payload = ak.zip({**fields, **cols}, depth_limit=1)
        (path,) = self.part_paths(partition)
        gw.prepare_part(path)
        _write_augmented(payload, path, self.manifest)
        return [path]

    def _value_columns(self, resolved: Callable[[int], ak.Array]) -> dict[str, ak.Array]:
        """Return each varied leaf's per-label column: an XOR delta vs nominal, or the value."""
        cols: dict[str, ak.Array] = {}
        for spec in self.value_specs:
            nominal_value = resolved(dict(spec.members)["nominal"])
            if spec.jagged:
                nominal_counts = _counts(nominal_value)
                for label, node_id in spec.members:
                    value = resolved(node_id)
                    if label == "nominal" or value.ndim == 1:
                        continue
                    if not np.array_equal(_counts(value), nominal_counts):
                        # §6.4d: differing per-label offsets have no representable XOR delta —
                        # execution-time (offsets are data), naming the label and the field.
                        raise GraphedError(
                            f"variation {label!r} changes the multiplicity of field "
                            f"{spec.path!r}: its per-label offsets differ from nominal's, so it "
                            "has no same-shaped delta (§6.4d — a multiplicity-changing stored "
                            "variation is Phase 2)"
                        )
            for label, node_id in spec.members:
                if label == "nominal":
                    continue
                value = resolved(node_id)
                cols[f"{_VARY_PREFIX}{label}__{spec.flat}"] = (
                    value
                    if label in spec.verbatim
                    else _encode_delta(nominal_value, value, spec.jagged)
                )
        return cols

    def _mask_columns(
        self, resolved: Callable[[int], ak.Array], base: ak.Array
    ) -> dict[str, ak.Array]:
        """Return each supplied level's per-label mask column, checked against ``base``."""
        cols: dict[str, ak.Array] = {}
        for mask_spec in self.mask_specs:
            field_values = base if mask_spec.field_path is None else base[mask_spec.field_path]
            for label, node_id in mask_spec.members:
                mask = resolved(node_id)
                if len(mask) != len(base):  # §6.4a(2b) row-count, execution-time
                    raise GraphedError(
                        f"the level-{mask_spec.entry_flat} select= mask for {label!r} spans "
                        f"{len(mask)} rows but the record spans {len(base)} (§6.4a(2b))"
                    )
                valid = ~ak.is_none(mask, axis=0)  # a None row has no objects to check
                # level->=1 STRUCTURAL: the mask's offsets must match the field's (§6.4a)
                if mask_spec.depth >= 1 and not np.array_equal(
                    np.asarray(ak.num(mask[valid])), np.asarray(ak.num(field_values[valid]))
                ):
                    raise GraphedError(
                        f"the level-{mask_spec.entry_flat} select= mask for {label!r} does not "
                        "share the offsets of the field it filters (§6.4a level->=1 structural "
                        "check)"
                    )
                cols[f"{_VARY_PREFIX}{label}__mask__{mask_spec.entry_flat}"] = (
                    mask if label in mask_spec.option else _encode_mask(mask)
                )
        return cols


def _write_varied(
    record: Any,
    destination: str,
    select: Any,
    *,
    steps_per_file: int | None,
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
    behavior = _session_behavior(session, behavior)

    # --- record-time entry checks (raise at the call; `compute=False` still raises) ---
    level0 = entries.get(0)
    _lineage_refusal(
        _record_context(record), _record_context(level0) if level0 is not None else None
    )
    _check_depth(session, entries)
    _check_bare_key(session, record, tuple(entries))
    varied = _varied_leaves(session, record, labels)
    _check_collision(varied)

    superset = None if level0 is None else _superset(session, level0, labels)
    base = nominal_record if superset is None else nominal_record[superset]
    value_specs, value_outputs = _value_specs(session, record, labels, varied, superset)
    mask_specs, mask_outputs = _mask_specs(session, entries, labels, superset)
    outputs: list[Any] = [base, *value_outputs, *mask_outputs]

    compiled = compile_ir(session, *outputs)
    refuse_chunk_partials(compiled, as_outputs=True)
    _refuse_optimizer_merge(session, outputs, compiled)

    manifest = _build_manifest(
        labels, value_specs, mask_specs, _record_types(session, record, labels)
    )
    rank = {node_id: i for i, node_id in enumerate(dict.fromkeys(out.node_id for out in outputs))}
    wrapped = _tt(session, nominal_record).ndim > 1

    ((source_id, data),) = gw.sources_read(outputs).items()
    steps_per_file = gw.steps_for(data, steps_per_file)
    source_name = session.source_name(source_id)
    source_form = session.form_of(source_id)
    assert isinstance(source_form, AwkwardForm)
    columns = gw.declared_columns(data, outputs)
    if columns is None:
        # `()` is this computation's OWN "read everything"; `read_partition` spells that `None`.
        columns = _evaluation_columns_union(outputs, source_id, source_name, source_form) or None

    return _write(
        _VariedWritePart,
        session,
        data,
        executor,
        compute=compute,
        compiled=compiled,
        source_name=source_name,
        columns=columns,
        destination=destination,
        prefix=prefix,
        steps_per_file=steps_per_file,
        labels=labels,
        rank=tuple(rank.items()),
        base_id=base.node_id,
        value_specs=tuple(value_specs),
        mask_specs=tuple(mask_specs),
        wrapped=wrapped,
        behavior=behavior,
        manifest=manifest,
        externals=shipped_evaluators(session, compiled),
    )


def _superset(session: Session, level0: Any, labels: Sequence[str]) -> Any:
    """Return the superset rows: the level-0 OR over the per-label masks (graph ops, §6.4a)."""
    return functools.reduce(
        operator.or_,
        (
            fill_none(m, True)
            if _is_option(session, m)
            else m  # a None row stays, as a None record
            for m in (member_of(level0, label) for label in labels)
        ),
    )


def _value_specs(
    session: Session, record: Any, labels: Sequence[str], varied: Sequence[str], superset: Any
) -> tuple[list[_ValueSpec], list[Any]]:
    """Return a `_ValueSpec` per varied leaf, and the per-label leaves the write evaluates."""
    nodes = session._nodes()
    value_specs: list[_ValueSpec] = []
    outputs: list[Any] = []
    for path in varied:
        parts = path.split(".")
        leaves: dict[str, Any] = {}
        for label in labels:
            member = member_of(record, label)
            key = _canonical_leaf(nodes, session, member, path)
            # a scalar leaf has no value node: project it out of the record
            leaves[label] = (
                session._proxies([key], [member])[0]
                if isinstance(key, int)
                else functools.reduce(operator.getitem, parts, member)
            )
        # a zip broadcast the leaf: encode what the reader's record holds
        if any(
            _type_of(session, leaves[label])
            != functools.reduce(
                operator.getitem, parts, _tt(session, member_of(record, label))
            ).layout.form.type
            for label in labels
        ):
            leaves = {
                label: functools.reduce(operator.getitem, parts, member_of(record, label))
                for label in labels
            }
        nominal_type = _type_of(session, leaves["nominal"])
        members: list[tuple[str, int]] = []
        verbatim: set[str] = set()
        for label in labels:
            leaf = leaves[label]
            if _type_of(session, leaf) != nominal_type or not _xor_representable(nominal_type):
                verbatim.add(label)
            leaf = leaf if superset is None else leaf[superset]
            members.append((label, leaf.node_id))
            outputs.append(leaf)
        value_specs.append(
            _ValueSpec(
                path,
                _flat_field(path),
                _tt(session, leaves["nominal"]).ndim > 1,
                tuple(members),
                frozenset(verbatim),
            )
        )
    return value_specs, outputs


def _mask_specs(
    session: Session, entries: Mapping[_SelectKey, Any], labels: Sequence[str], superset: Any
) -> tuple[list[_MaskSpec], list[Any]]:
    """Return a `_MaskSpec` per supplied level in §6.4e order, and the per-label masks."""
    mask_specs: list[_MaskSpec] = []
    outputs: list[Any] = []
    for key in sorted(entries, key=_level_sort_key):
        mask = entries[key]
        field_path = None if isinstance(key, int) else key[0]
        members: list[tuple[str, int]] = []
        option: set[str] = set()
        for label in labels:
            member = member_of(mask, label)
            if _is_option(session, member):
                option.add(label)
            member = member if superset is None else member[superset]
            members.append((label, member.node_id))
            outputs.append(member)
        mask_specs.append(
            _MaskSpec(
                _entry_flat(key),
                _entry_manifest(key),
                _entry_depth(key),
                field_path,
                tuple(members),
                frozenset(option),
            )
        )
    return mask_specs, outputs


def _record_types(session: Session, record: Any, labels: Sequence[str]) -> dict[str, Any]:
    """Return each label's record form the reader cannot rebuild from the stored columns."""
    nominal_form = _tt(session, member_of(record, "nominal")).layout.form
    record_types = {
        label: form.to_dict()
        for label in labels
        if not (form := _tt(session, member_of(record, label)).layout.form).type.is_equal_to(
            nominal_form.type, all_parameters=True
        )
    }
    # its stored fields drop the record's own
    if nominal_form.is_record and nominal_form.parameters:
        record_types["nominal"] = nominal_form.to_dict()
    return record_types


def _write(
    task: type[_WritePart] | type[_VariedWritePart],
    session: Session,
    data: Any,
    executor: Any | None,
    *,
    compute: bool,
    **fields: Any,
) -> list[str] | Plan[list[str]]:
    """Return the write plan of ``task`` over ``data``'s partitions, or run it when ``compute``.

    ANY source describing its own partitioning (parquet datasets, the ROOT reader integration, ...)
    is written partition by partition — its whole-dataset loader is never invoked; in-memory data
    is split into ``steps_per_file`` eager steps.
    """
    source_name, steps_per_file = fields["source_name"], fields["steps_per_file"]
    if isinstance(data, PartitionedSource):
        partitions = data.partitions(steps_per_file)
        writer = task(**_file_tables(partitions, steps_per_file), reader=data, **fields)
    else:
        whole = ak.Array(data() if callable(data) else data)
        n = len(whole)
        partitions = tuple(
            Partition(f"memory://{source_name}", "", start, stop)
            for start, stop in gw.eager_ranges(n, steps_per_file)
        )
        writer = task(bases=(), memory_data=whole, memory_rows=n, **fields)

    plan = gw.write_plan(partitions, writer, plan_services(session, fields["compiled"]))
    if not compute:
        return plan
    runner = executor if executor is not None else SequentialRunner()
    return list(runner.run(plan).value)


@overload
def to_parquet(
    array: Any,
    destination: str,
    *,
    select: Any = ...,
    steps_per_file: int = ...,
    compute: Literal[True] = ...,
    executor: Any | None = ...,
    prefix: str = ...,
    column: str = ...,
    behavior: Any = ...,
) -> list[str]: ...
@overload
def to_parquet(
    array: Any,
    destination: str,
    *,
    select: Any = ...,
    steps_per_file: int = ...,
    compute: Literal[False],
    executor: Any | None = ...,
    prefix: str = ...,
    column: str = ...,
    behavior: Any = ...,
) -> Plan[list[str]]: ...
@overload
def to_parquet(
    array: Any,
    destination: str,
    *,
    select: Any = ...,
    steps_per_file: int = ...,
    compute: bool,
    executor: Any | None = ...,
    prefix: str = ...,
    column: str = ...,
    behavior: Any = ...,
) -> list[str] | Plan[list[str]]: ...
def to_parquet(
    array: Any,
    destination: str,
    *,
    select: Any = None,
    steps_per_file: int | None = None,
    compute: bool = True,
    executor: Any | None = None,
    prefix: str = "part",
    column: str = "data",
    behavior: Any = None,
) -> list[str] | Plan[list[str]]:
    """Write the deferred array to parquet parts, one per partition (R15.4 semantics).

    With ``compute=False`` returns the task graph of write tasks; with ``compute=True`` runs that
    SAME plan (``SequentialRunner`` by default; pass any R7 executor). The array must be recorded
    over exactly one source; the per-task read list comes from the recorded graph's projection. A
    ``destination`` containing ``://`` is an fsspec URL: the parts are written on its filesystem.
    ``steps_per_file`` parts are written per input file; ``None`` takes the source's own default.

    ``select=`` turns this into a §6.4 variation-aware write: ``array`` is the PRE-selection record
    and ``select`` carries the per-level ``Varied`` mask(s) (a single row mask ⇔ ``{0: mask}``, or
    a ``{0: event_mask, ("Jet", 1): jet_mask}`` mapping). The writer materializes the SUPERSET of
    rows passing any universe's selection, augments the nominal record with reconstruction data,
    and returns the ordinary part-path list — a §2.3d *accepting* verb. Without ``select=`` the
    path is unchanged (byte-identical, §6.4g).
    """
    if select is not None:
        return _write_varied(
            array,
            destination,
            select,
            steps_per_file=steps_per_file,
            compute=compute,
            executor=executor,
            prefix=prefix,
            behavior=behavior,
        )
    session: Session = array.session
    behavior = _session_behavior(session, behavior)
    sources = gw.sources_read([array])
    if len(sources) != 1:
        raise TypeError(
            f"to_parquet needs an array recorded over exactly one source, got {len(sources)}"
        )
    ((node_id, data),) = sources.items()
    steps_per_file = gw.steps_for(data, steps_per_file)
    source_name = session.source_name(node_id)
    source_form = session.form_of(node_id)
    assert isinstance(source_form, AwkwardForm)  # this backend recorded the source
    columns = gw.declared_columns(data, (array,))
    if columns is None:
        # `()` is this computation's OWN "read everything"; `read_partition` spells that `None`.
        columns = _evaluation_columns(array, node_id, source_name, source_form) or None
    compiled = compile_ir(session, array)
    refuse_chunk_partials(compiled, as_outputs=True)
    return _write(
        _WritePart,
        session,
        data,
        executor,
        compute=compute,
        compiled=compiled,
        source_name=source_name,
        columns=columns,
        destination=destination,
        prefix=prefix,
        steps_per_file=steps_per_file,
        column=column,
        behavior=behavior,
        externals=shipped_evaluators(session, compiled),
    )
