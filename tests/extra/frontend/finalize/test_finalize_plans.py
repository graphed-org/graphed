"""``aggregate_plan(finalize=)`` and ``collate`` carry each plan's finalize to the run's value."""

from __future__ import annotations

import pickle
from typing import Any

import m69b_opt_fixtures as fx
import numpy as np

from graphed import aggregate_plan, collate
from graphed.core import Partition, Plan, SequentialRunner

OTHER = (Partition.blind("mem://other", "t", 0, 1),)


def _plan(**kw: Any) -> Plan[Any]:
    _s, events = fx.record()
    return aggregate_plan(
        events.x, reduce=fx.sums, combine=fx.add, empty=fx.zero, steps_per_file=2, **kw
    )


def test_aggregate_plan_finalizes_the_combined_value() -> None:
    calls: list[Any] = []

    def finalize(v: Any) -> Any:
        calls.append(v)
        return ("final", float(np.sum(v)))

    assert _plan().finalize is None
    assert SequentialRunner().run(_plan()).value.tolist() == [10.0]
    assert SequentialRunner().run(_plan(finalize=finalize)).value == ("final", 10.0)
    assert len(calls) == 1


def test_collate_finalizes_each_name_by_its_own_plan() -> None:
    plain = collate({"a": _plan(), "b": _plan(partitions=())})
    assert plain.finalize is None
    collated = collate(
        {
            "a": _plan(finalize=lambda v: ("a", v.tolist())),
            "b": _plan(partitions=OTHER),
            "c": _plan(partitions=(), finalize=lambda v: ("c", v)),
        }
    )
    value = SequentialRunner().run(collated).value
    assert value["a"] == ("a", [10.0]) and value["b"].tolist() == [10.0]
    assert "c" not in value


def test_a_lambda_finalize_pickles_with_its_plan() -> None:
    shipped = pickle.loads(pickle.dumps(_plan(finalize=lambda v: ("final", v.tolist()))))
    assert SequentialRunner().run(shipped).value == ("final", [10.0])
