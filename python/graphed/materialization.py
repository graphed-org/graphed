"""Evaluate recorded arrays in this process, reading only the source columns they use."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .array import Array
from .core.execution import LocalResources
from .projection import read_columns
from .session import Session
from .write import PartitionedSource, declared_columns, sources_read


def _replace_arrays(item: Any, leaf: Callable[[Array], Any]) -> Any:
    """Return ``item`` with each Array replaced by ``leaf(array)``.

    Through lists, tuples (named ones keep their type) and mappings; other items pass through.
    """
    if isinstance(item, Array):
        return leaf(item)
    if isinstance(item, Mapping):
        return {key: _replace_arrays(v, leaf) for key, v in item.items()}
    if isinstance(item, list):
        return [_replace_arrays(v, leaf) for v in item]
    if isinstance(item, tuple):
        values = [_replace_arrays(v, leaf) for v in item]
        return type(item)(*values) if hasattr(item, "_fields") else tuple(values)
    return item


def _projected_reads(session: Session, arrays: Sequence[Array]) -> dict[int, object]:
    """Each partitioned source ``arrays`` read, read with the columns a plan over them would read.

    A source whose read list is every column, or whose partitions the backend cannot concatenate,
    is left to its lazy loader.
    """
    concat = getattr(session.backend, "concat", None)
    reads: dict[int, object] = {}
    resources = LocalResources()
    try:
        for nid, data in sources_read(arrays).items():
            if not isinstance(data, PartitionedSource):
                continue
            columns = declared_columns(data, arrays)
            if columns is None:
                columns = read_columns(arrays, nid)
            if columns is None:
                continue
            partitions = data.partitions(1)
            if len(partitions) == 1:
                reads[nid] = data.read_partition(partitions[0], columns, resources)
            elif concat is not None:
                reads[nid] = concat(
                    [data.read_partition(p, columns, resources) for p in partitions]
                )
    finally:
        resources.close()
    return reads


def materialize(arrays: Any) -> Any:
    """Evaluate ``arrays`` in this process.

    An :class:`Array`, or any nesting of lists, tuples (named ones keep their type) and mappings
    holding them (other items pass through), recorded in any number of Sessions. Returns the same
    structure with each Array replaced by its value; the Arrays of one Session evaluate every node
    they share once, and read only the source columns they use, as a plan over them would.
    Evaluation only: running on an executor is a plan's job
    (:func:`graphed.aggregate_plan`, :func:`graphed.collate`).
    """
    found: dict[Session, list[Array]] = {}
    _replace_arrays(arrays, lambda a: found.setdefault(a.session, []).append(a))
    memos = {session: _projected_reads(session, outs) for session, outs in found.items()}
    return _replace_arrays(arrays, lambda a: a.session.materialize(a, memo=memos[a.session]))
