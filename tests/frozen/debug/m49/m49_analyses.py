"""Shared fixtures for the m49 debug suite: a poisoned toy backend, the merged-key topology, and the
module-level worker the process-boundary test spawns.

Named ``m49_analyses`` and not ``analyses``: the ``debug`` subtree is collected WHOLE in one pytest
process (``scripts/run-tests.sh``) and ``m6/analyses.py`` is live, so a bare name would bind to
whichever sibling directory imported first.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import graphed.core
from graphed import Array, Session, aggregate_plan
from graphed.core import Partition
from graphed.core.execution import Plan

POISON_OP = "mul"
POISON_MESSAGE = "poisoned stage member"
#: The failing key's label, and a label put on every OTHER key: an attribution that reports the
#: first/any entry rather than the failing key's reports ``OTHER_LABEL``.
LABEL = "jes_up"
OTHER_LABEL = "zz_elsewhere"


class PoisonError(RuntimeError):
    """The worker-side failure the wrap either attributes or re-raises untouched."""


class PoisonBackend:
    """A toy backend (no numpy, no awkward) whose ``POISON_OP`` always raises."""

    def op_form(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> str:
        return op

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        if op == POISON_OP:
            raise PoisonError(POISON_MESSAGE)
        return inputs[0]

    def boundary_ops(self) -> frozenset[str]:
        return frozenset({"source"})

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        return used

    def external_payload(self, op: str, params: Mapping[str, object]) -> None:
        return None


@dataclass
class ListSource:
    """A ``PartitionedSource`` over an in-memory list."""

    data: tuple[int, ...] = tuple(range(1, 13))

    def __call__(self) -> list[int]:
        raise AssertionError("the whole-dataset loader must never run during a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(
            Partition.blind("toy://list", "", s, steps_per_file) for s in range(steps_per_file)
        )

    def read_partition(
        self, partition: Partition, columns: object, resources: object
    ) -> list[int]:
        part = partition.resolve(len(self.data))
        return list(self.data[part.entry_start : part.entry_stop])


class Resources:
    def open_once(self, uri: str, opener: Callable[[str], object]) -> object:
        return opener(uri)


def merged_key_program() -> tuple[Session, Array, Array, ListSource]:
    """Two ops recorded at DIFFERENT user lines that the reducer merges onto ONE reduced key.

    ``poisoned + 0.0`` is an additive identity, so equality saturation quotients it onto ``poisoned``
    and keeps the earlier node. Returns ``(session, output, poisoned, source)``; ``poisoned`` holds
    the LOWER of the two record ids.
    """
    session = Session(PoisonBackend())
    src = ListSource()
    x = session.source("x", form="f", data=src)
    poisoned = x * 2.0
    identity = poisoned + 0.0
    return session, identity, poisoned, src


def reduce_first(values: list[object]) -> object:
    return values[0]


def combine_first(a: object, b: object) -> object:
    return a


def empty_none() -> object:
    return None


# ---- §8.2(i) payload construction -------------------------------------------------------------


def _key_order(key: tuple[int, int | None]) -> tuple[int, int]:
    """§8.2(i)'s sort: a bare ``sorted()`` over the keys is a ``TypeError`` once one reduced id
    carries both an indexed and a ``None`` entry."""
    return (key[0], -1 if key[1] is None else key[1])


def _is_key(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], int)
        and (value[1] is None or isinstance(value[1], int))
    )


def _frame_assoc(value: object) -> list[tuple[Any, Any]] | None:
    """``value`` read as §8.2(i)'s per-key frame association list, or ``None`` if it is not one."""
    items: object = value.items() if isinstance(value, Mapping) else value
    if isinstance(items, str | bytes) or not isinstance(items, Iterable):
        return None
    pairs = list(items)
    if not pairs or not all(isinstance(p, tuple) and len(p) == 2 for p in pairs):
        return None
    return pairs if all(_is_key(k) for k, _ in pairs) else None


def _artifact_values(compiled: object) -> Iterable[object]:
    top = list(vars(compiled).values())
    yield from top
    for value in top:
        if isinstance(value, tuple):
            yield from value
        elif hasattr(value, "__dict__"):
            yield from vars(value).values()


def artifact_frames(compiled: object) -> dict[tuple[int, int | None], Any]:
    """§8.2(i)'s per-key frame association list, read off the ``CompiledGraph``.

    Located by LAYOUT rather than by name — the field's spelling is pinned by the accessor anchor in
    ``tests/frozen/frontend/m49`` — and frames travel through as opaque plain data.
    """
    found = [assoc for assoc in map(_frame_assoc, _artifact_values(compiled)) if assoc is not None]
    assert len(found) == 1, (
        "expected exactly one per-key frame association list "
        "(((reduced_node_id, member_index | None), frame), ...) on the CompiledGraph per §8.2(i); "
        f"found {len(found)} across fields {sorted(vars(compiled))}"
    )
    return dict(found[0])


def failing_keys(compiled: object) -> set[tuple[int, int | None]]:
    """The keys of the compiled store's output nodes — where ``POISON_OP`` raises in this fixture."""
    outputs = set(graphed.core.GraphStore.deserialize(bytes(compiled.ir)).outputs())  # type: ignore[attr-defined]
    return {key for key in artifact_frames(compiled) if key[0] in outputs}


def _payload(
    compiled: object, labels_for: Callable[[tuple[int, int | None]], tuple[str, ...] | None]
) -> tuple[Any, ...]:
    """§8.2(i)'s association list: one entry per key, frames COPIED off the artifact, sorted."""
    frames = artifact_frames(compiled)
    return tuple(
        (key, (labels, frames[key]))
        for key in sorted(frames, key=_key_order)
        if (labels := labels_for(key)) is not None
    )


def hook_labelling_every_key(compiled: object) -> tuple[Any, ...]:
    """Every key gets an entry; only the failing key gets ``LABEL``."""
    failing = failing_keys(compiled)
    return _payload(compiled, lambda key: (LABEL,) if key in failing else (OTHER_LABEL,))


def hook_skipping_the_failing_key(compiled: object) -> tuple[Any, ...]:
    """A populated payload that carries NO entry for the key that fails."""
    failing = failing_keys(compiled)
    return _payload(compiled, lambda key: None if key in failing else (OTHER_LABEL,))


def plan_for(
    hook: Callable[[object], object] | None = None,
) -> tuple[Session, Array, Array, Plan[object]]:
    session, out, poisoned, _src = merged_key_program()
    plan = aggregate_plan(
        out,
        reduce=reduce_first,
        combine=combine_first,
        empty=empty_none,
        steps_per_file=2,
        on_compiled=hook,
    )
    return session, out, poisoned, plan


def run_labelled_failure_and_raise() -> None:
    """Module-level worker for ``multiprocessing``: build + run in THIS process and let the attributed
    ``StageError`` propagate, so the parent receives it pickled across the process boundary."""
    _session, _out, _poisoned, plan = plan_for(hook_labelling_every_key)
    plan.process(plan.tasks[0].partition, Resources())
