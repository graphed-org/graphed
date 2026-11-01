"""Debug execution that maps any failure back to the user's analysis line (plan M6).

``run`` evaluates the lowered graph op-by-op (so it localizes a failure even when it happens deep
inside a fused stage), wrapping the first failing op in a ``StageError`` carrying that op's exact
provenance, its input forms, and the partition. At ``opt_level=0`` it also runs inter-op
consistency assertions. This is a *debug* runner, not the M7 executor; it exists to prove the
error-surfacing contract, including across a process boundary (the resulting ``StageError`` is
picklable).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import cast

from graphed import Array, Session
from graphed.core import Partition
from graphed.materialization import _partition_chunk, _Sample, _sample_of
from graphed.varied import refuse_container

from .errors import SourceFrame, StageError
from .lowering import LoweredGraph, LoweredOp, lower


def _given(value: object) -> object:
    return value


def _chain(lowered: LoweredGraph, node_id: int) -> tuple[SourceFrame, ...]:
    """Return the user-frame lineage.

    The failing op first (the analysis line), then back along its primary input to a source — a
    readable 'how this value was built' trail.
    """
    frames: list[SourceFrame] = []
    cur: int | None = node_id
    seen: set[int] = set()
    while cur is not None and cur not in seen:
        seen.add(cur)
        op = lowered.op_for(cur)
        frames.append(op.provenance)
        cur = op.input_ids[0] if op.input_ids else None
    return tuple(frames)


def _stage_error(
    lowered: LoweredGraph, node_id: int, partition: str, exc: BaseException
) -> StageError:
    failing = lowered.op_for(node_id)
    input_forms = tuple(lowered.op_for(i).form for i in failing.input_ids)
    return StageError(
        op=failing.op,
        frames=_chain(lowered, node_id),
        input_forms=input_forms,
        partition=partition,
        cause_type=type(exc).__name__,
        cause_message=str(exc),
        opt_level=lowered.opt_level,
    )


def _assert_consistent(lowered: LoweredGraph, node_id: int, value: object, partition: str) -> None:
    """Check inter-op consistency, only at opt_level=0.

    A debug scaffold; richer dtype/shape contracts are a tracked improvement.
    """
    if value is None:
        raise _stage_error(
            lowered,
            node_id,
            partition,
            AssertionError("op produced None (inter-op consistency check)"),
        )


def _walk(
    session: Session,
    array: Array,
    opt_level: int,
    chunks: Mapping[int, Callable[[], object]],
    where: str,
    memo: dict[int, object] | None = None,
) -> object:
    lowered = lower(session, array, opt_level=opt_level)
    backend = session.backend

    def on_source(nid: int) -> object:
        try:
            return chunks[nid]() if nid in chunks else session.source_value(nid)
        except StageError:
            raise
        except Exception as exc:
            raise _stage_error(lowered, nid, where, exc) from exc

    def on_op(nid: int, name: str, ins: list[object], params: Mapping[str, object]) -> object:
        try:
            value = backend.eval_stage(name, ins, params)
        except StageError:
            raise
        except Exception as exc:
            raise _stage_error(lowered, nid, where, exc) from exc
        if opt_level == 0:
            _assert_consistent(lowered, nid, value, where)
        return value

    def on_external(nid: int, fn: object, ins: list[object]) -> object:
        try:
            return fn(*ins)  # type: ignore[operator]
        except StageError:
            raise
        except Exception as exc:
            raise _stage_error(lowered, nid, where, exc) from exc

    return session.walk(array, source=on_source, op=on_op, external=on_external, memo=memo)


def run(
    session: Session,
    array: Array,
    *,
    opt_level: int = 1,
    partition: Partition | None = None,
    label: str | None = None,
) -> object:
    """Execute the analysis reaching ``array`` on the session's source data.

    ``partition`` runs that one chunk (``plan.tasks[3].partition``) instead of the whole source.
    ``label`` names the data in a failure, by default ``partition``.

    Raise a `StageError` (mapped to the user's analysis line) on the first failing op.
    """
    if isinstance(partition, str):
        raise TypeError(
            f"run(partition=) selects a graphed.core.Partition, not {partition!r}; name the data "
            "in a failure with label=, or evaluate a sample with graphed.materialize(x, entries=N)"
        )
    chunks = {} if partition is None else _partition_chunk([array], partition)
    where = label if label is not None else "sample" if partition is None else str(partition)
    return _walk(session, array, opt_level, chunks, where)


def _rows(value: object) -> int | str:
    """Return how many rows ``value`` has; a scalar's length means nothing, so it shows ``-``."""
    return len(value) if getattr(value, "ndim", 0) else "-"  # type: ignore[arg-type]


def trace(array: Array, *, entries: int) -> None:
    """Print how many rows each recorded line keeps of the first ``entries`` input events.

    One row per line, for its last operation, marking the first that empties the data. This finds
    a cut that removes everything. A failing operation raises its `StageError`.
    """
    refuse_container("graphed.debug.trace", array)
    session = array.session
    sample = cast(_Sample, _sample_of(entries, None))
    reads = sample.reads(session, [array])
    chunks = {nid: partial(_given, value) for nid, value in reads.items()}
    values: dict[int, object] = {}
    _walk(session, array, 0, chunks, sample.where, values)
    # a dict keeps the first position of a line and the last operation recorded on it
    lines: dict[tuple[str, int], LoweredOp] = {}
    for op in lower(session, array, opt_level=0).ops:
        lines[op.provenance.filename, op.provenance.lineno] = op
    emptied = False
    for (filename, lineno), op in lines.items():
        rows_in = next(
            (rows for rows in (_rows(values[i]) for i in op.input_ids) if rows != "-"), "-"
        )
        rows_out = _rows(values[op.node_id])
        first_empty = not emptied and rows_out == 0 and rows_in not in ("-", 0)
        emptied = emptied or first_empty
        mark = "   <-- removed every event" if first_empty else ""
        print(
            f"{Path(filename).name}:{lineno:<5} {op.op:<16} {rows_in:>9} -> {rows_out:<9} "
            f"{op.provenance.source}{mark}"
        )
