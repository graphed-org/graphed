"""M39 — the repartition frontend surface (plan §3.1), backend-agnostic.

The factorization rule (memory ``frontend-array-factorization``): ``Array`` stays idiom-neutral. A
keyed repartition is a graphed capability -> a neutral MODULE verb ``graphed.repartition``; count/size
rebalancing is *physical* (moves rows, no idiom) -> it stays on ``Array.repartition``. Every form
records an ``Exchange`` node carrying its scheme params. graphed itself imports no numpy/awkward, so
these run over a toy backend.
"""

from __future__ import annotations

from shuffle_backends import ListSource, ToyBackend

import graphed
from graphed import Array, Session, compile_ir
from graphed.core import GraphStore


def _exchange_nodes(session: Session, result: Array) -> list[dict]:
    ir = bytes(compile_ir(session, result).ir)
    return [n for n in GraphStore.deserialize(ir).nodes() if n["kind"] == "exchange"]


def _src(session: Session) -> Array:
    return session.source("x", form="f", data=ListSource([{"__joinkey__": i, "v": i} for i in range(8)]))


def test_graphed_repartition_is_a_neutral_module_verb() -> None:
    # a join/keyed-shuffle is NEITHER an awkward nor a numpy idiom -> a module function, not on Array.
    assert callable(graphed.repartition)


def test_repartition_by_key_records_a_hash_exchange() -> None:
    s = Session(ToyBackend())
    out = graphed.repartition(_src(s), by="__joinkey__")
    assert isinstance(out, Array) and out.session is s  # idiom-neutral proxy
    xchg = _exchange_nodes(s, out)
    assert len(xchg) == 1
    assert xchg[0]["params"]["scheme"] in {"hash", "range"}
    assert xchg[0]["params"]["key"] == "__joinkey__", "the routing key field must be recorded"


def test_array_repartition_by_size_records_a_coalesce_exchange() -> None:
    # count/size rebalancing stays on Array (physical, backend-neutral); target_bytes -> coalesce.
    s = Session(ToyBackend())
    out = _src(s).repartition(target_bytes=2048)
    assert isinstance(out, Array)
    xchg = _exchange_nodes(s, out)
    assert len(xchg) == 1
    assert xchg[0]["params"]["scheme"] == "coalesce"
    assert xchg[0]["params"]["target_bytes"] == 2048


def test_array_repartition_by_count_records_the_partition_count() -> None:
    s = Session(ToyBackend())
    out = _src(s).repartition(n=4)
    xchg = _exchange_nodes(s, out)
    assert len(xchg) == 1
    assert xchg[0]["params"]["parts"] == 4, "count rebalance must record the target partition count"


def test_repartition_params_drive_structural_identity() -> None:
    # two different targets are two different Exchange nodes (params in the structural key, not ignored)
    s = Session(ToyBackend())
    src = _src(s)
    a = graphed.repartition(src, by="__joinkey__")
    b = src.repartition(n=4)
    assert bytes(compile_ir(s, a).ir) != bytes(compile_ir(s, b).ir)
