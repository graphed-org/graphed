"""Externals sharing a payload and params, on a flat and a jagged input, check their own claims."""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import Array, Session, aggregate_plan
from graphed.awkward import AwkwardBackend, AwkwardForm, gak
from graphed.core import Partition, SequentialRunner
from graphed.debug import StageError
from graphed.execute import node_key
from graphed.preserve import record_external
from graphed.preserve.externals import ExternalPlugin, sha256_bytes

correctionlib = pytest.importorskip("correctionlib")

CSET = json.dumps(
    {
        "schema_version": 2,
        "corrections": [
            {
                "name": "a",
                "version": 1,
                "inputs": [{"name": "pt", "type": "real"}],
                "output": {"name": "sf", "type": "real"},
                "data": {
                    "nodetype": "formula",
                    "expression": "2*x",
                    "parser": "TFormula",
                    "variables": ["pt"],
                },
            }
        ],
    }
).encode()
EVENTS = ak.Array({"pt": [[10.0, 20.0], [], [30.0]]})
FLAT = [20.0, 40.0, 60.0]
JAGGED = [[20.0, 40.0], [], [60.0]]


@dataclass
class _Events:
    data: ak.Array

    def __call__(self) -> ak.Array:
        return self.data

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(
            Partition.blind("mem://events", "", s, steps_per_file) for s in range(steps_per_file)
        )

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> ak.Array:
        part = partition.resolve(len(self.data))
        return self.data[part.entry_start : part.entry_stop]


def _samples() -> list[bytes]:
    return [b"x"]


def _twice(_resource: object, _params: object, inputs: list[Any]) -> Any:
    return inputs[0] * 2.0


DOUBLE = ExternalPlugin(
    kind="double",
    content_hash=sha256_bytes,
    evaluate=_twice,
    samples=_samples,
    output_dtype="float64",
    broadcasts=True,  # a whole-type claim: the inputs' broadcast, so flat and jagged claim apart
)


def _payload() -> bytes:
    # a fresh object per call, so one pickled copy is one shipped evaluator; recorded lines
    # (provenance) must not spell it
    return ("payload-" + "308").encode()


def _by_value() -> Callable[[Any], Any]:
    def _double(v: Any, _tag: str = "by-value-308") -> Any:  # counts its shipped copies
        return v * 2.0

    return _double


_DOUBLE = _by_value()  # a local def, which a plan ships by value


Record = Callable[[Session, Array], Array]
KINDS: dict[str, Record] = {
    "correction": lambda s, x: gak.apply_correction(CSET, "a", [x], None, args=["$0"]),
    "plugin": lambda s, x: record_external(s, DOUBLE, _payload(), [x]),
    "map": lambda s, x: x.map(_DOUBLE),
}


def _record(kind: str | Record, jagged_first: bool) -> tuple[Session, Array, Array]:
    session = Session(AwkwardBackend())
    form = AwkwardForm(ak.Array(EVENTS.layout.to_typetracer(forget_length=True)))
    pt = session.source("events", form=form, data=_Events(EVENTS)).pt
    record = KINDS[kind] if isinstance(kind, str) else kind
    if jagged_first:
        jag = record(session, pt)
        flat = record(session, gak.flatten(pt))
    else:
        flat = record(session, gak.flatten(pt))
        jag = record(session, pt)
    return session, flat, jag


def _fold(values: list[Any]) -> list[Any]:
    return [ak.to_list(v) for v in values]


def _add(a: list[Any], b: list[Any]) -> list[Any]:
    return [x + y for x, y in zip(a, b, strict=True)]


def _empty() -> list[Any]:
    return [[], []]


def _plan(*outputs: Array, **options: Any) -> Any:
    return aggregate_plan(*outputs, reduce=_fold, combine=_add, empty=_empty, **options)


ORDERS = pytest.mark.parametrize("jagged_first", [False, True], ids=["flat_first", "jagged_first"])


@ORDERS
@pytest.mark.parametrize("kind", sorted(KINDS))
def test_materialize_checks_each_node_against_its_own_claim(kind: str, jagged_first: bool) -> None:
    _session, flat, jag = _record(kind, jagged_first)
    assert flat.node_id != jag.node_id
    assert ak.to_list(graphed.materialize(flat)) == FLAT
    assert ak.to_list(graphed.materialize(jag)) == JAGGED


@ORDERS
def test_head_checks_each_node_against_its_own_claim(jagged_first: bool) -> None:
    _session, flat, jag = _record("correction", jagged_first)
    assert ak.to_list(graphed.head(flat, 3)) == FLAT
    assert ak.to_list(graphed.head(jag, 3)) == JAGGED


@ORDERS
@pytest.mark.parametrize("kind", sorted(KINDS))
def test_a_plan_runs_each_node_with_its_own_claim(kind: str, jagged_first: bool) -> None:
    _session, flat, jag = _record(kind, jagged_first)
    plan = _plan(flat, jag)
    shipped = pickle.loads(pickle.dumps(plan))
    assert SequentialRunner().run(shipped).value == [FLAT, JAGGED]


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_the_later_node_gets_its_own_entry_and_the_payload_ships_once(kind: str) -> None:
    _session, flat, jag = _record(kind, jagged_first=False)
    plan = _plan(flat, jag)
    keys = [key for key, _fn in plan.process.externals]
    nodes = graphed.core.GraphStore.deserialize(bytes(plan.process.ir)).nodes()
    later = [n for n in nodes if n["kind"] == "external"][-1]
    assert keys[1:] == [node_key(later)]
    payload = {"correction": b"2*x", "plugin": b"payload-308", "map": b"by-value-308"}[kind]
    assert pickle.dumps(plan).count(payload) == 1


def _flattens(_resource: object, _params: object, inputs: list[Any]) -> Any:
    return ak.flatten(inputs[0], axis=None) * 2.0


FLATTENS = ExternalPlugin(
    kind="flattens",
    content_hash=sha256_bytes,
    evaluate=_flattens,
    samples=_samples,
    output_dtype="float64",
    broadcasts=True,
)


def test_a_value_breaking_its_own_nodes_claim_is_still_refused_naming_that_node() -> None:
    _session, flat, jag = _record(lambda s, x: record_external(s, FLATTENS, b"x", [x]), False)
    assert ak.to_list(graphed.materialize(flat)) == FLAT
    with pytest.raises(StageError) as info:
        graphed.materialize(jag)
    assert f"node {jag.node_id} " in info.value.cause_message
    assert "declares array output_type 'var * float64'" in info.value.cause_message


def _first(values: list[Any]) -> Any:
    return ak.to_list(values[0])


def _concat(a: list[Any], b: list[Any]) -> list[Any]:
    return a + b


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_a_repartitions_map_side_carries_each_nodes_own_evaluator(kind: str) -> None:
    _session, flat, jag = _record(kind, jagged_first=False)
    both = gak.unflatten(flat, gak.num(jag, axis=1)) + jag
    plan = graphed.shuffle_plan(both.repartition(n=1), reduce=_first, combine=_concat, empty=list)
    assert SequentialRunner().run(plan).value == [[40.0, 80.0], [], [120.0]]


UNCHECKED = ExternalPlugin(
    kind="unchecked",
    content_hash=sha256_bytes,
    evaluate=_twice,
    samples=_samples,
    array_value=False,  # no stated type: its form stays its first input's, unchecked
)


def test_siblings_with_no_claim_share_the_one_evaluator() -> None:
    session, flat, jag = _record(
        lambda s, x: record_external(s, UNCHECKED, _payload(), [x]), False
    )
    fns = [fn for fn, _inputs in session._externals.values()]
    assert fns[0] is not fns[1]
    plan = _plan(flat, jag)
    assert len(plan.process.externals) == 1
    assert SequentialRunner().run(plan).value == [FLAT, JAGGED]


@ORDERS
@pytest.mark.parametrize("given", [False, True], ids=["plan_ir", "outputs"])
def test_replay_runs_each_node_with_its_own_claim(jagged_first: bool, given: bool) -> None:
    _session, flat, jag = _record("correction", jagged_first)
    plan = _plan(flat, jag)
    outputs = (flat, jag) if given else ()
    assert _fold(graphed.debug.replay(plan, 0, *outputs).value) == [FLAT, JAGGED]


def test_an_override_of_the_key_binds_every_node_of_it() -> None:
    _session, flat, jag = _record("plugin", jagged_first=False)
    auto = _plan(flat, jag)
    (key, _fn), _own = auto.process.externals
    plan = _plan(flat, jag, externals={key: lambda v: v * 3.0})
    assert SequentialRunner().run(plan).value == [[30.0, 60.0, 90.0], [[30.0, 60.0], [], [90.0]]]
