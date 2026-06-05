"""Debug execution that maps any failure back to the user's analysis line (plan M6).

`run` evaluates the lowered graph op-by-op (so it localizes a failure even when it happens deep
inside a fused stage), wrapping the first failing op in a `StageError` carrying that op's exact
provenance, its input forms, and the partition. At ``opt_level=0`` it also runs inter-op consistency
assertions. This is a *debug* runner, not the M7 executor; it exists to prove the error-surfacing
contract, including across a process boundary (the resulting `StageError` is picklable).
"""

from __future__ import annotations

from collections.abc import Mapping

from graphed import Array, Session

from .errors import SourceFrame, StageError
from .lowering import LoweredGraph, lower


def _chain(lowered: LoweredGraph, node_id: int) -> tuple[SourceFrame, ...]:
    """The user-frame lineage: the failing op first (the analysis line), then back along its primary
    input to a source — a readable 'how this value was built' trail."""
    frames: list[SourceFrame] = []
    cur: int | None = node_id
    seen: set[int] = set()
    while cur is not None and cur not in seen:
        seen.add(cur)
        op = lowered.op_for(cur)
        frames.append(op.provenance)
        cur = op.input_ids[0] if op.input_ids else None
    return tuple(frames)


def _stage_error(lowered: LoweredGraph, node_id: int, partition: str, exc: BaseException) -> StageError:
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
    """An inter-op consistency check, only at opt_level=0 (a debug scaffold; richer dtype/shape
    contracts are a tracked improvement)."""
    if value is None:
        raise _stage_error(
            lowered, node_id, partition, AssertionError("op produced None (inter-op consistency check)")
        )


def run(session: Session, array: Array, *, opt_level: int = 1, partition: str = "sample") -> object:
    """Execute the analysis reaching ``array`` on the session's source data, raising a `StageError`
    (mapped to the user's analysis line) on the first failing op."""
    lowered = lower(session, array, opt_level=opt_level)
    backend = session.backend

    def on_source(nid: int) -> object:
        try:
            return session.source_value(nid)
        except StageError:
            raise
        except Exception as exc:
            raise _stage_error(lowered, nid, partition, exc) from exc

    def on_op(nid: int, name: str, ins: list[object], params: Mapping[str, object]) -> object:
        try:
            value = backend.eval_stage(name, ins, params)
        except StageError:
            raise
        except Exception as exc:
            raise _stage_error(lowered, nid, partition, exc) from exc
        if opt_level == 0:
            _assert_consistent(lowered, nid, value, partition)
        return value

    def on_external(nid: int, fn: object, ins: list[object]) -> object:
        try:
            return fn(*ins)  # type: ignore[operator]
        except StageError:
            raise
        except Exception as exc:
            raise _stage_error(lowered, nid, partition, exc) from exc

    return session.walk(array, source=on_source, op=on_op, external=on_external)
