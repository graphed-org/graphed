"""M39 — GIR1 codec extension for ``Exchange`` (plan §2.3, T_EXCHANGE=5).

The tag is APPENDED (``T_EXCHANGE=5``) and the magic stays ``b"GIR1"``: existing graphs (tags 0..4)
serialize byte-identically, so the M8 determinism gate is untouched; an Exchange-bearing blob
round-trips byte-identically; and a corrupt/unknown node tag is rejected loudly (never mis-parsed).
"""

from __future__ import annotations

import pytest

from graphed.core import GraphStore


def _exchange_graph() -> tuple[GraphStore, list[int]]:
    g = GraphStore()
    src = g.add_source("events", {"uri": "f.root"})
    pt = g.add_op("pt", [src])
    xchg = g.add_exchange([pt], {"scheme": "hash", "key": "__joinkey__", "parts": 8})
    out = g.add_reduction("sum", [xchg])
    return g, [out]


def _legacy_graph() -> bytes:
    g = GraphStore()
    src = g.add_source("events", {"uri": "f.root"})
    out = g.add_reduction("sum", [g.add_op("pt", [src])])
    return g.serialize(outputs=[out])


def test_magic_is_unchanged_even_with_an_exchange() -> None:
    # the tag is APPENDED, not a format break: a blob CONTAINING an Exchange still starts with GIR1.
    g, outs = _exchange_graph()
    blob = g.serialize(outputs=outs)
    assert blob[:4] == b"GIR1", "the magic must stay GIR1 (append a tag, never bump the magic)"


def test_exchange_blob_roundtrips_byte_identically() -> None:
    g, outs = _exchange_graph()
    blob = g.serialize(outputs=outs)
    back = GraphStore.deserialize(blob)
    assert back.node_count() == g.node_count()
    # freeze-M39-1 (owner-sanctioned refreeze, 2026-07-02): compare deserialize-vs-deserialize —
    # the M8 pattern (m8/test_ir_serialization.py) — because the blob-marked `back` can never equal
    # the unmarked builder `g` under the M22 read-only-serialize pin (dispute: .graphed/M39/disputes/).
    assert back.to_dot() == GraphStore.deserialize(blob).to_dot()
    assert back.serialize() == blob, "re-serialize of a decoded Exchange graph must be byte-identical"


def test_exchange_scheme_survives_the_roundtrip() -> None:
    g, outs = _exchange_graph()
    back = GraphStore.deserialize(g.serialize(outputs=outs))
    xchg = next(n for n in back.nodes() if n["kind"] == "exchange")
    assert xchg["params"] == {"scheme": "hash", "key": "__joinkey__", "parts": 8}


def test_legacy_graphs_are_byte_stable_and_unaffected() -> None:
    # a graph with no Exchange node must serialize identically to a second identical build (the M8
    # determinism gate is untouched by adding the tag).
    assert _legacy_graph() == _legacy_graph()


def test_a_corrupt_node_tag_is_rejected_not_misparsed() -> None:
    g, outs = _exchange_graph()
    blob = bytearray(g.serialize(outputs=outs))
    # byte 8 is the first node's tag (4 magic + 4 node-count). Poison it with an out-of-range tag.
    blob[8] = 99
    with pytest.raises(ValueError):
        GraphStore.deserialize(bytes(blob))
