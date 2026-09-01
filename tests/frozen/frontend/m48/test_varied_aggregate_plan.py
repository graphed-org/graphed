"""§7.2's `aggregate_plan` SEAM — (alpha) the compiled artifact, (beta) its return channel.

`aggregate_plan` compiles internally, so the `CompiledGraph` the §7.2 merge refusal and m49's
`variation_labels` key on is unreachable from the caller. The seam is ADDITIVE: m5's frozen call
shape (`tests/frozen/frontend/m5/test_aggregate_plan.py`) keeps passing `reduce`/`combine`/`empty`
as plain callables. The anchor is over an UNVARIED program — the seam is what m48 adds, not the
expansion.
"""

from __future__ import annotations

from typing import Any

from vary_fixtures import ToyBackend, add_pairs, m5_two_outputs, sum_each

from graphed import CompiledGraph, Session, aggregate_plan
from graphed.aggregate import _PartitionReduce
from graphed.core import GraphStore
from graphed.core.execution import SequentialRunner

M5_VALUE = [sum(2 * v for v in range(1, 13)), sum(3 * v for v in range(1, 13))]  # [156, 234]


def _build(on_compiled: Any = None) -> Any:
    s = Session(ToyBackend())
    out1, out2, _src = m5_two_outputs(s)
    kwargs = {} if on_compiled is None else {"on_compiled": on_compiled}
    return aggregate_plan(
        out1, out2, reduce=sum_each, combine=add_pairs, empty=lambda: [0, 0], steps_per_file=4, **kwargs
    )


def test_hook_fires_exactly_once_with_the_compiled_graph() -> None:
    seen: list[CompiledGraph] = []
    outputs: list[list[int]] = []

    def hook(compiled: CompiledGraph) -> None:
        seen.append(compiled)
        outputs.append(GraphStore.deserialize(compiled.ir).outputs())

    _build(hook)
    assert len(seen) == 1, "the seam must not compile twice (the §3.3 budget is one reduction)"
    assert isinstance(seen[0], CompiledGraph)  # the artifact itself, not a list of output ids
    # `CompiledGraph` exposes only ir/source_names/evaluate, so the outputs are read by
    # deserializing — the route §7.2's merge refusal uses to count DISTINCT compiled outputs.
    assert len(outputs[0]) == 2


def test_the_hooked_plan_runs_to_the_hookless_value() -> None:
    """m5's own assertion is the positive control: the seam changes no result."""
    assert SequentialRunner().run(_build(lambda compiled: None)).value == M5_VALUE
    assert SequentialRunner().run(_build()).value == M5_VALUE


def test_the_hook_return_value_is_carried_onto_the_shipped_closure() -> None:
    """(beta) is the RETURN channel: the payload it carries is produced inside the call, so no
    call-time parameter could supply it. `()` is a well-typed NON-DEFAULT value of §8.2(i)'s
    declared `tuple[..., ...] | None` field — `None` is indistinguishable from the default."""
    process = _build(lambda compiled: ()).process
    # `Plan.process` is declared `Callable[[Partition, WorkerResources], R]`, which has no such
    # field; the narrowing is also the runtime discriminator of the concrete closure.
    assert isinstance(process, _PartitionReduce)
    assert process.variation_labels == ()
    assert _build().process.variation_labels is None  # untouched when no hook is supplied
